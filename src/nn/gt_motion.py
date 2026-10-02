"""
Modelo de movimento das Partes 4 e 5 (Trilha A), treinado so em
trajetorias do ground truth (src/nn/train.py):

    GTMotionModel                 um estado recorrente por track; recebe
                                  [caixa_t | caixa_t - caixa_{t-1} |
                                  observado (| confianca | dt)] e preve a
                                  caixa do quadro seguinte como residuo
    train_motion_epoch,
    evaluate_motion               loops de treino / avaliacao
    gradient_norm_through_time    norma de dL_t/dh_{t-k} (horizonte
                                  analitico da Parte 4)
    GTRNNMotion                   adaptador para o Tracker (src/nn/tracking.py)

E a versao do MotionModel em que os checkpoints reports/results/4_*.pt
foram treinados. A Parte 2 usa outro MotionModel (src/nn/models.py),
com a velocidade relativa ao tamanho da caixa como entrada e as
deteccoes reais do SDP no treino. As celulas e o Recurrent sao os mesmos.

Caixas dentro do modelo: (cx, cy, w, h) NORMALIZADAS pelo tamanho da
imagem, em [0, 1] (src/dataset/gt_trajectories.py).
"""

import numpy as np
import torch
from torch import nn

from src.nn.boxes import xyxy_to_cxcywh, cxcywh_to_xyxy
from src.nn.models import Recurrent, detach_state
from src.nn.tracking import MotionModelBase


# ============================================================
# MODELO
# ============================================================


class GTMotionModel(nn.Module):
    """
    Um estado recorrente por track. A cada quadro a celula recebe

        x_t = [ caixa_t (4) | caixa_t - caixa_{t-1} (4) | observado (1)
                | confianca (1, opcional) | dt (1, opcional) ]

    e preve a caixa do quadro seguinte como um RESIDUO sobre a entrada:

        caixa_{t+1} = caixa_t + head(h_t)[:4]

    Opcionalmente preve tambem log sigma^2 por coordenada (4), para a
    log-verossimilhanca gaussiana -- o portao de associacao adaptativo.

    Regimes de treino (Parte 3, Eixo 2), via `sampling_prob` em forward:
        0.0  teacher forcing: a entrada e sempre a caixa verdadeira;
        p    scheduled sampling: com prob. p a entrada e a PROPRIA
             previsao anterior (destacada do grafo);
        1.0  free-running: so a primeira caixa e verdadeira.
    Passos com `observed = 0` (oclusao) SEMPRE usam a propria previsao,
    em qualquer regime -- e o que acontece na inferencia.

    Args:
        cell, hidden_size, num_layers, bidirectional: ver Recurrent.
        predict_uncertainty: acrescenta as 4 saidas de log-variancia.
        use_conf, use_dt: incluir confianca / dt na entrada.
    """

    def __init__(self, cell="gru", hidden_size=64, num_layers=1, bidirectional=False,
                 predict_uncertainty=False, use_conf=True, use_dt=True):
        super().__init__()

        self.kind = cell
        self.hidden_size = hidden_size
        self.predict_uncertainty = predict_uncertainty
        self.use_conf = use_conf
        self.use_dt = use_dt
        self.bidirectional = bidirectional

        self.input_size = 4 + 4 + 1 + int(use_conf) + int(use_dt)

        self.rnn = Recurrent(cell, self.input_size, hidden_size, num_layers, bidirectional)

        out_size = 8 if predict_uncertainty else 4
        self.head = nn.Linear(self.rnn.output_size, out_size)

        # comeca prevendo "fica parado": residuo ~ 0
        with torch.no_grad():
            self.head.weight.mul_(0.1)
            self.head.bias.zero_()

    # --------------------------------------------------------

    def features(self, box, prev_box, observed, conf=None, dt=None):
        """Monta x_t. Todos (..., k); devolve (..., input_size)."""

        parts = [box, box - prev_box, observed]

        if self.use_conf:
            parts.append(conf if conf is not None else torch.ones_like(observed))

        if self.use_dt:
            parts.append(dt if dt is not None else torch.ones_like(observed))

        return torch.cat(parts, dim=-1)

    def _split_head(self, out):

        if self.predict_uncertainty:
            delta, log_var = out[..., :4], out[..., 4:]
            return delta, log_var.clamp(-10.0, 5.0)

        return out, None

    def init_state(self, batch_size, device=None):
        return self.rnn.init_state(batch_size, device)

    # --------------------------------------------------------

    def step(self, box, prev_box, observed, state, conf=None, dt=None):
        """
        Um passo online para (um lote de) tracks. Todos (B, k).

        Returns:
            {"box": (B, 4) caixa prevista para o proximo quadro,
             "log_var": (B, 4) ou None, "state": novo estado}
        """

        x = self.features(box, prev_box, observed, conf, dt)
        h, state = self.rnn.step(x, state)

        delta, log_var = self._split_head(self.head(h))

        return {"box": box + delta, "log_var": log_var, "state": state}

    def forward(self, boxes, observed=None, conf=None, dt=None, sampling_prob=0.0,
                state=None, keep_hidden=False):
        """
        Sequencias inteiras (treino / avaliacao offline).

        Args:
            boxes: (B, T, 4) caixas normalizadas.
            observed: (B, T, 1) 1 = observacao disponivel (padrao: tudo).
            conf, dt: (B, T, 1) opcionais.
            sampling_prob: ver docstring da classe.
            state: estado inicial (BPTT truncado: o final do bloco anterior).
            keep_hidden: guarda h_t com retain_grad (analise de gradiente).

        Returns:
            {"box": (B, T, 4) previsao da caixa de t+1 feita em t,
             "log_var": (B, T, 4) ou None,
             "input": (B, T, 4) o que de fato entrou em cada passo,
             "state": estado final, "hiddens": lista de h_t}
        """

        B, T, _ = boxes.shape
        device = boxes.device

        if observed is None:
            observed = torch.ones(B, T, 1, device=device)

        if self.bidirectional:
            # offline: tudo de uma vez, sem scheduled sampling
            prev = torch.cat([boxes[:, :1], boxes[:, :-1]], dim=1)
            x = self.features(boxes, prev, observed, conf, dt)
            out, state, hiddens = self.rnn(x, state, keep_hidden)
            delta, log_var = self._split_head(self.head(out))
            return {"box": boxes + delta, "log_var": log_var, "input": boxes,
                    "state": state, "hiddens": hiddens}

        if state is None:
            state = self.rnn.init_state(B, device)

        state = list(state)

        preds, log_vars, inputs, hiddens = [], [], [], []

        prev_input = boxes[:, 0]
        prev_pred = boxes[:, 0]

        for t in range(T):

            gt_box = boxes[:, t]
            obs_t = observed[:, t]

            if t == 0:
                box_in = gt_box
            else:
                # sem observacao -> propria previsao; com observacao ->
                # verdadeira, exceto se o scheduled sampling sortear
                use_pred = obs_t < 0.5

                if sampling_prob > 0:
                    coin = torch.rand(B, 1, device=device) < sampling_prob
                    use_pred = use_pred | coin

                box_in = torch.where(use_pred, prev_pred.detach(), gt_box)

            x = self.features(
                box_in, prev_input, obs_t,
                conf[:, t] if conf is not None else None,
                dt[:, t] if dt is not None else None,
            )

            h, state = self.rnn.step(x, state)

            if keep_hidden:
                h.retain_grad()
                hiddens.append(h)

            delta, log_var = self._split_head(self.head(h))
            pred = box_in + delta

            preds.append(pred)
            inputs.append(box_in)

            if log_var is not None:
                log_vars.append(log_var)

            prev_input = box_in
            prev_pred = pred

        return {
            "box": torch.stack(preds, dim=1),
            "log_var": torch.stack(log_vars, dim=1) if log_vars else None,
            "input": torch.stack(inputs, dim=1),
            "state": state,
            "hiddens": hiddens,
        }


# ============================================================
# TREINO E AVALIACAO
# ============================================================


def _to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def train_motion_epoch(model, dataloader, optimizer, loss_fn, device,
                       sampling_prob=0.0, clip_grad=None, tbptt=None):
    """
    Uma epoca do GTMotionModel.

    Batches (ver src/dataset/gt_trajectories.py): dict com
        boxes (B, T, 4), observed (B, T, 1), conf (B, T, 1),
        dt (B, T, 1), valid (B, T) -- padding.

    A previsao feita em t e comparada com a caixa verdadeira de t+1;
    `loss_fn(pred, target, log_var, mask)` (src/nn/loss.py).

    Args:
        sampling_prob: regime (0 teacher forcing ... 1 free-running).
        clip_grad: norma maxima do gradiente (None = sem clipping).
        tbptt: comprimento da janela de BPTT truncado (None = a
            sequencia inteira). O estado atravessa as janelas destacado.

    Returns:
        (perda media, norma media do gradiente antes do clipping)
    """

    model.train()

    total_loss = 0.0
    total_norm = 0.0
    n_steps = 0

    for batch in dataloader:

        batch = _to_device(batch, device)

        boxes = batch["boxes"]
        B, T, _ = boxes.shape

        window = T if tbptt is None else int(tbptt)
        state = None

        for start in range(0, T - 1, window):

            end = min(start + window, T)

            conf = batch.get("conf")
            dt = batch.get("dt")

            out = model(
                boxes[:, start:end],
                observed=batch["observed"][:, start:end],
                conf=conf[:, start:end] if conf is not None else None,
                dt=dt[:, start:end] if dt is not None else None,
                sampling_prob=sampling_prob,
                state=state,
            )

            # previsao em t (start..end-1) vs. verdade em t+1 (ate T-1)
            n = min(end, T - 1) - start

            if n <= 0:
                break

            pred = out["box"][:, :n]
            target = boxes[:, start + 1:start + 1 + n]
            mask = batch["valid"][:, start + 1:start + 1 + n]
            log_var = out["log_var"][:, :n] if out["log_var"] is not None else None

            loss = loss_fn(pred, target, log_var=log_var, mask=mask)

            optimizer.zero_grad()
            loss.backward()

            norm = float(torch.nn.utils.clip_grad_norm_(
                model.parameters(), clip_grad if clip_grad is not None else float("inf")
            ))

            optimizer.step()

            state = detach_state(out["state"])

            total_loss += loss.item()
            total_norm += norm
            n_steps += 1

    return total_loss / max(n_steps, 1), total_norm / max(n_steps, 1)


@torch.no_grad()
def evaluate_motion(model, dataloader, loss_fn, device, sampling_prob=0.0):
    """
    Perda media e erro L1 medio (em unidades normalizadas) do
    GTMotionModel num loader, no regime dado (na inferencia real os
    passos ocluidos sao sempre free-running -- `observed` cuida disso).
    """

    model.eval()

    total_loss = 0.0
    total_l1 = 0.0
    n_batches = 0

    for batch in dataloader:

        batch = _to_device(batch, device)
        boxes = batch["boxes"]

        out = model(
            boxes,
            observed=batch["observed"],
            conf=batch.get("conf"),
            dt=batch.get("dt"),
            sampling_prob=sampling_prob,
        )

        pred = out["box"][:, :-1]
        target = boxes[:, 1:]
        mask = batch["valid"][:, 1:]
        log_var = out["log_var"][:, :-1] if out["log_var"] is not None else None

        total_loss += loss_fn(pred, target, log_var=log_var, mask=mask).item()

        l1 = (pred - target).abs().mean(dim=-1)
        total_l1 += float((l1 * mask).sum() / mask.sum().clamp(min=1))

        n_batches += 1

    return total_loss / max(n_batches, 1), total_l1 / max(n_batches, 1)


# ============================================================
# HORIZONTE DE MEMORIA  --  norma de dL_t / dh_{t-k}  (Parte 4)
# ============================================================


def gradient_norm_through_time(model, batch, loss_fn, device, t_loss=None, sampling_prob=0.0):
    """
    A curva do gradiente que some (slide 52), no SEU modelo e nos SEUS
    dados: roda a sequencia guardando os h_t, computa a perda SO no
    passo t_loss, faz backward e le ||dL_t / dh_{t-k}|| para k = 0..t.

    Args:
        model: GTMotionModel (causal).
        batch: um batch como o de train_motion_epoch.
        loss_fn: a mesma do treino.
        t_loss: passo onde a perda e medida (padrao: o ultimo valido).

    Returns:
        np.ndarray (t_loss + 1,) com a norma para k = 0..t_loss
        (indice k = distancia no tempo).
    """


    model.eval()
    batch = _to_device(batch, device)

    boxes = batch["boxes"]
    B, T, _ = boxes.shape

    if t_loss is None:
        t_loss = T - 2

    model.zero_grad()

    out = model(
        boxes,
        observed=batch["observed"],
        conf=batch.get("conf"),
        dt=batch.get("dt"),
        sampling_prob=sampling_prob,
        keep_hidden=True,
    )

    pred = out["box"][:, t_loss:t_loss + 1]
    target = boxes[:, t_loss + 1:t_loss + 2]
    mask = batch["valid"][:, t_loss + 1:t_loss + 2]
    log_var = out["log_var"][:, t_loss:t_loss + 1] if out["log_var"] is not None else None

    loss = loss_fn(pred, target, log_var=log_var, mask=mask)
    loss.backward()

    norms = np.zeros(t_loss + 1)

    for k in range(t_loss + 1):
        grad = out["hiddens"][t_loss - k].grad
        norms[k] = float(grad.norm(dim=-1).mean()) if grad is not None else 0.0

    model.zero_grad()

    return norms


# ============================================================
# ADAPTADOR PARA O TRACKER
# ============================================================


class GTRNNMotion(MotionModelBase):
    """
    Adaptador para o GTMotionModel (Partes 4 e 5, Trilha A). Um estado
    recorrente POR TRACK.

    A cada quadro a celula recebe a ultima observacao (caixa, confianca,
    dt, flag de observado) e preve a caixa do quadro seguinte. Sob
    oclusao (`coast`) a entrada e a propria previsao anterior, com a
    flag de observado em zero: o estado roda para frente sem observacao.

    Args:
        model: GTMotionModel treinado.
        image_size: (largura, altura) para normalizar as caixas.
        device: torch device.
    """

    def __init__(self, model, image_size, device="cpu"):

        import torch

        self.torch = torch
        self.model = model.to(device).eval()
        self.device = device
        self.image_size = np.asarray(image_size, dtype=np.float64)

    def _normalize(self, box):
        cxcywh = xyxy_to_cxcywh(np.asarray(box, dtype=np.float64))
        scale = np.concatenate([self.image_size, self.image_size])
        return cxcywh / scale

    def _denormalize(self, cxcywh):
        scale = np.concatenate([self.image_size, self.image_size])
        return cxcywh_to_xyxy(np.asarray(cxcywh, dtype=np.float64) * scale)

    def _step(self, state, box_n, score, dt, observed):

        torch = self.torch

        with torch.no_grad():

            box_t = torch.as_tensor(box_n, dtype=torch.float32, device=self.device)[None]
            prev_t = torch.as_tensor(state["last_input"], dtype=torch.float32, device=self.device)[None]

            out = self.model.step(
                box=box_t,
                prev_box=prev_t,
                conf=torch.full((1, 1), float(score), device=self.device),
                dt=torch.full((1, 1), float(dt), device=self.device),
                observed=torch.full((1, 1), float(observed), device=self.device),
                state=state["h"],
            )

        state["h"] = out["state"]
        state["last_input"] = box_n
        state["pred_n"] = out["box"][0].cpu().numpy()
        state["log_var"] = out["log_var"][0].cpu().numpy() if out.get("log_var") is not None else None
        return state

    def init(self, box, score, t):

        box_n = self._normalize(box)

        state = {
            "h": self.model.init_state(1, self.device),
            "last_input": box_n,
            "t_last": t,
        }

        return self._step(state, box_n, score, dt=1.0, observed=1.0)

    def predict(self, state):
        return self._denormalize(state["pred_n"])

    def update(self, state, box, score, t):

        dt = float(t - state["t_last"])
        state["t_last"] = t

        return self._step(state, self._normalize(box), score, dt=dt, observed=1.0)

    def coast(self, state, t):

        # Alimenta a propria previsao (free-running), sem observacao
        return self._step(state, state["pred_n"], score=0.0, dt=1.0, observed=0.0)


