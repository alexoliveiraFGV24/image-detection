"""
Modelos temporais, escritos do zero (sem nn.RNN / nn.LSTM / nn.GRU):

    RNNCell, LSTMCell, GRUCell   as celulas dos slides 38, 54-55 e 71
    Recurrent                    roda uma celula (ou pilha delas) numa
                                 sequencia; opcionalmente bidirecional;
                                 guarda os h_t para a analise de gradiente
    MotionModel                  Parte 2, Trilha A: um estado recorrente
                                 por track que recebe a ultima observacao
                                 (caixa, confianca, dt, flag de observado)
                                 e preve a caixa do quadro seguinte
                                 (+ incerteza opcional)
    CropEncoder, AppearanceModel Parte 2, Trilha B: embedding por recorte
                                 + agregador recorrente do estado de
                                 aparencia da track
    crop_boxes                   recorta caixas de uma imagem (grid_sample)

Mais os loops de treino (teacher forcing -> scheduled sampling ->
free-running, BPTT truncado, gradient clipping) e a medida analitica do
horizonte de memoria (norma de dL_t/dh_{t-k}) da Parte 4.

Caixas dentro dos modelos: (cx, cy, w, h) NORMALIZADAS pelo tamanho da
imagem, em [0, 1]. A conversao mora em src/nn/tracking.py::RNNMotion
e em src/dataset/trajectories.py.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F


# ============================================================
# CELULAS
# ============================================================


class RNNCell(nn.Module):
    """
    RNN simples (slide 38):

        h_t = tanh(W_hh h_{t-1} + W_xh x_t + b) = tanh(W [h_{t-1}; x_t] + b)

    O estado e so h. Backpropagation de h_t para h_{t-1} multiplica por
    W_hh^T (e pela derivada da tanh) a cada passo -- e dai que vem o
    gradiente que some (slides 51-52).
    """

    n_gates = 1

    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.linear = nn.Linear(input_size + hidden_size, hidden_size)

    def init_state(self, batch_size, device=None):
        return torch.zeros(batch_size, self.hidden_size, device=device)

    @staticmethod
    def hidden(state):
        return state

    def forward(self, x, state):
        h = torch.tanh(self.linear(torch.cat([state, x], dim=-1)))
        return h, h


class LSTMCell(nn.Module):
    """
    LSTM (slides 54-63), com a notacao dos slides:

        [f, i, s, c~] = [sigma, sigma, sigma, tanh](W [h_{t-1}; x_t] + b)
        c_t = f * c_{t-1} + i * c~          (forget / ignore)
        h_t = s * tanh(c_t)                 (select)

    O estado e (h, c). De c_{t-1} para c_t so ha multiplicacao
    elemento a elemento por f (slide 64): sem W^T repetido, o gradiente
    atravessa muitos passos. O vies do forget gate comeca em 1 para a
    celula nascer "lembrando".
    """

    n_gates = 4

    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.linear = nn.Linear(input_size + hidden_size, 4 * hidden_size)

        with torch.no_grad():
            self.linear.bias[:hidden_size].fill_(1.0)     # forget gate

    def init_state(self, batch_size, device=None):
        zeros = torch.zeros(batch_size, self.hidden_size, device=device)
        return (zeros, zeros.clone())

    @staticmethod
    def hidden(state):
        return state[0]

    def forward(self, x, state):

        h_prev, c_prev = state

        gates = self.linear(torch.cat([h_prev, x], dim=-1))
        f, i, s, c_tilde = gates.chunk(4, dim=-1)

        f = torch.sigmoid(f)
        i = torch.sigmoid(i)
        s = torch.sigmoid(s)
        c_tilde = torch.tanh(c_tilde)

        c = f * c_prev + i * c_tilde
        h = s * torch.tanh(c)

        return h, (h, c)


class GRUCell(nn.Module):
    """
    GRU (slide 71):

        z_t = sigma(W_z [h_{t-1}; x_t])                 (update)
        r_t = sigma(W_r [h_{t-1}; x_t])                 (reset)
        h~  = tanh(W_h [r_t * h_{t-1}; x_t])
        h_t = z_t * h_{t-1} + (1 - z_t) * h~

    Sem estado de celula separado: a propria h faz a "esteira".
    """

    n_gates = 3

    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.gates = nn.Linear(input_size + hidden_size, 2 * hidden_size)     # z, r
        self.candidate = nn.Linear(input_size + hidden_size, hidden_size)     # h~

    def init_state(self, batch_size, device=None):
        return torch.zeros(batch_size, self.hidden_size, device=device)

    @staticmethod
    def hidden(state):
        return state

    def forward(self, x, state):

        h_prev = state

        z, r = self.gates(torch.cat([h_prev, x], dim=-1)).chunk(2, dim=-1)
        z = torch.sigmoid(z)
        r = torch.sigmoid(r)

        h_tilde = torch.tanh(self.candidate(torch.cat([r * h_prev, x], dim=-1)))

        h = z * h_prev + (1.0 - z) * h_tilde

        return h, h


CELLS = {
    "rnn": RNNCell,
    "lstm": LSTMCell,
    "gru": GRUCell,
}


def cell_parameter_count(kind, input_size, hidden_size):
    """Numero de parametros de uma celula (bias incluido)."""

    gates = CELLS[kind].n_gates
    return gates * ((input_size + hidden_size) * hidden_size + hidden_size)


def hidden_size_for_budget(kind, input_size, n_params):
    """
    Maior hidden_size cuja celula `kind` cabe em `n_params` parametros.
    E o "mesmo orcamento aproximado de parametros" do Eixo 1 da Parte 3:
    a LSTM tem 4 matrizes, a GRU 3, a RNN 1 -- com o mesmo H a LSTM
    teria 4x mais parametros que a RNN.

        g * (H^2 + (I + 1) H) = P   ->   H = (-(I+1) + sqrt((I+1)^2 + 4P/g)) / 2
    """

    g = CELLS[kind].n_gates
    a = input_size + 1

    h = (-a + math.sqrt(a * a + 4.0 * n_params / g)) / 2.0

    return max(1, int(math.floor(h)))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# CAMADA RECORRENTE
# ============================================================


def detach_state(state):
    """Corta o grafo no estado (BPTT truncado) -- lida com h ou (h, c)."""

    if state is None:
        return None

    if isinstance(state, (tuple, list)):
        return type(state)(detach_state(s) for s in state)

    return state.detach()


class Recurrent(nn.Module):
    """
    Roda uma pilha de celulas sobre uma sequencia (B, T, I).

    Args:
        cell: "rnn" | "lstm" | "gru".
        input_size, hidden_size, num_layers.
        bidirectional: se True, uma segunda pilha le a sequencia de tras
            para frente e as saidas sao concatenadas (slide 80). So faz
            sentido offline (Eixo 4 da Parte 3).

    forward(x, state=None, keep_hidden=False) devolve
        outputs  (B, T, H * dirs)
        state    estado final (lista por camada; par (fwd, bwd) se bi)
        hiddens  lista de h_t da ultima camada forward, um por passo,
                 com retain_grad() se keep_hidden -- e o que a analise
                 de gradiente da Parte 4 le.
    """

    def __init__(self, cell="gru", input_size=8, hidden_size=64, num_layers=1, bidirectional=False):
        super().__init__()

        self.kind = cell
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        cell_class = CELLS[cell]

        self.forward_cells = nn.ModuleList([
            cell_class(input_size if k == 0 else hidden_size, hidden_size)
            for k in range(num_layers)
        ])

        if bidirectional:
            self.backward_cells = nn.ModuleList([
                cell_class(input_size if k == 0 else hidden_size, hidden_size)
                for k in range(num_layers)
            ])
        else:
            self.backward_cells = None

    @property
    def output_size(self):
        return self.hidden_size * (2 if self.bidirectional else 1)

    def init_state(self, batch_size, device=None):

        forward = [c.init_state(batch_size, device) for c in self.forward_cells]

        if not self.bidirectional:
            return forward

        backward = [c.init_state(batch_size, device) for c in self.backward_cells]
        return (forward, backward)

    def step(self, x, state):
        """
        Um passo (online, so a direcao forward). x: (B, I).
        Returns: (h da ultima camada (B, H), novo estado).
        """

        if self.bidirectional:
            raise RuntimeError("step() e causal; modelo bidirecional so roda offline (forward)")

        new_state = []
        h = x

        for cell, s in zip(self.forward_cells, state):
            h, s_new = cell(h, s)
            new_state.append(s_new)

        return h, new_state

    def _run_direction(self, cells, x, state, keep_hidden):

        B, T, _ = x.shape
        hiddens = []

        layer_input = x

        for layer, (cell, s) in enumerate(zip(cells, state)):

            outputs = []

            for t in range(T):

                h, s = cell(layer_input[:, t], s)

                if keep_hidden and layer == len(cells) - 1:
                    h.retain_grad()
                    hiddens.append(h)

                outputs.append(h)

            layer_input = torch.stack(outputs, dim=1)
            state[layer] = s

        return layer_input, state, hiddens

    def forward(self, x, state=None, keep_hidden=False):

        B, T, _ = x.shape

        if state is None:
            state = self.init_state(B, x.device)

        if not self.bidirectional:
            out, state, hiddens = self._run_direction(self.forward_cells, x, list(state), keep_hidden)
            return out, state, hiddens

        fwd_state, bwd_state = state

        out_f, fwd_state, hiddens = self._run_direction(self.forward_cells, x, list(fwd_state), keep_hidden)
        out_b, bwd_state, _ = self._run_direction(self.backward_cells, x.flip(1), list(bwd_state), False)

        out = torch.cat([out_f, out_b.flip(1)], dim=-1)

        return out, (fwd_state, bwd_state), hiddens


# ============================================================
# TRILHA A  --  RNN como modelo de movimento
# ============================================================


class MotionModel(nn.Module):
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
# TRILHA B  --  RNN como memoria de aparencia
# ============================================================


def crop_boxes(images, boxes, batch_index=None, size=(32, 32)):
    """
    Recorta e redimensiona caixas de imagens com grid_sample (sem
    torchvision.ops).

    Args:
        images: (B, C, H, W) float.
        boxes: (N, 4) xyxy em PIXELS.
        batch_index: (N,) de qual imagem vem cada caixa (padrao: 0).
        size: (h, w) do recorte.

    Returns:
        (N, C, h, w)
    """

    if images.dim() == 3:
        images = images[None]

    B, C, H, W = images.shape
    N = boxes.shape[0]

    if N == 0:
        return images.new_zeros((0, C, *size))

    boxes = boxes.to(images.dtype)

    if batch_index is None:
        batch_index = torch.zeros(N, dtype=torch.long, device=images.device)

    x1, y1, x2, y2 = boxes.unbind(dim=1)

    # theta mapeia a grade [-1, 1] do recorte para a caixa, em coords
    # normalizadas da imagem
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    sx = (x2 - x1).clamp(min=1.0) / W
    sy = (y2 - y1).clamp(min=1.0) / H

    theta = torch.zeros(N, 2, 3, dtype=images.dtype, device=images.device)
    theta[:, 0, 0] = sx
    theta[:, 0, 2] = 2.0 * cx / W - 1.0
    theta[:, 1, 1] = sy
    theta[:, 1, 2] = 2.0 * cy / H - 1.0

    grid = F.affine_grid(theta, (N, C, *size), align_corners=False)

    return F.grid_sample(images[batch_index], grid, align_corners=False, padding_mode="zeros")


class CropEncoder(nn.Module):
    """
    Encoder pequeno, do zero, para recortes (3 x 32 x 32 por padrao):
    tres blocos conv-BN-ReLU-pool, media global, FC -> D, normalizado.
    """

    def __init__(self, embedding_dim=64, in_channels=3, width=32):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(in_channels, width),
            block(width, 2 * width),
            block(2 * width, 4 * width),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4 * width, embedding_dim)

        self.embedding_dim = embedding_dim

    def forward(self, crops):
        x = self.pool(self.features(crops)).flatten(1)
        return F.normalize(self.fc(x), dim=-1)


class AppearanceModel(nn.Module):
    """
    Cada deteccao vira um embedding D-dimensional (CropEncoder); um
    agregador recorrente mantem o estado de aparencia da track,
    atualizado a cada observacao. Sem observacao (oclusao) o estado e
    mantido -- a aparencia nao muda so porque a pessoa esta escondida.

    A saida por passo e o embedding AGREGADO (normalizado), comparado
    por cosseno na associacao; a perda contrastiva / triplet
    (src/nn/loss.py) age sobre ele e sobre o instantaneo.

    Args:
        embedding_dim: D.
        cell, hidden_size, num_layers: o agregador.
        encoder: um nn.Module recorte -> (N, D) (padrao: CropEncoder).
    """

    def __init__(self, embedding_dim=64, cell="gru", hidden_size=64, num_layers=1, encoder=None):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.encoder = encoder if encoder is not None else CropEncoder(embedding_dim)
        self.rnn = Recurrent(cell, embedding_dim, hidden_size, num_layers, bidirectional=False)
        self.head = nn.Linear(hidden_size, embedding_dim)

    def init_state(self, batch_size, device=None):
        return self.rnn.init_state(batch_size, device)

    def encode(self, crops):
        """(N, 3, h, w) -> (N, D) embedding instantaneo, normalizado."""
        return self.encoder(crops)

    def _aggregate(self, e, observed, state):

        h, new_state = self.rnn.step(e, state)

        # mantem o estado antigo onde nao houve observacao
        keep = 1.0 - observed

        if isinstance(new_state[0], tuple):
            mixed = [
                tuple(observed * n + keep * o for n, o in zip(ns, os_))
                for ns, os_ in zip(new_state, state)
            ]
        else:
            mixed = [observed * n + keep * o for n, o in zip(new_state, state)]

        h_mixed = self.rnn.forward_cells[-1].hidden(mixed[-1])

        return F.normalize(self.head(h_mixed), dim=-1), mixed

    def step(self, crop, state, observed=None):
        """
        Um passo online. crop: (B, 3, h, w); observed: (B, 1).

        Returns:
            {"embedding": (B, D) agregado, "instant": (B, D), "state"}
        """

        e = self.encode(crop)

        if observed is None:
            observed = torch.ones(e.shape[0], 1, device=e.device)

        emb, state = self._aggregate(e, observed, state)

        return {"embedding": emb, "instant": e, "state": state}

    def forward(self, crops, observed=None, state=None):
        """
        crops: (B, T, 3, h, w); observed: (B, T, 1).

        Returns:
            {"embedding": (B, T, D), "instant": (B, T, D), "state"}
        """

        B, T = crops.shape[:2]

        instant = self.encode(crops.flatten(0, 1)).view(B, T, -1)

        if observed is None:
            observed = torch.ones(B, T, 1, device=crops.device)

        if state is None:
            state = self.rnn.init_state(B, crops.device)

        state = list(state)
        outputs = []

        for t in range(T):
            emb, state = self._aggregate(instant[:, t], observed[:, t], state)
            outputs.append(emb)

        return {"embedding": torch.stack(outputs, dim=1), "instant": instant, "state": state}


# ============================================================
# LOOPS DE TREINO
# ============================================================


def _to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def train_motion_epoch(model, dataloader, optimizer, loss_fn, device,
                       sampling_prob=0.0, clip_grad=None, tbptt=None):
    """
    Uma epoca do MotionModel.

    Batches (ver src/dataset/trajectories.py): dict com
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
    MotionModel num loader, no regime dado (na inferencia real os
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


def train_appearance_epoch(model, dataloader, optimizer, loss_fn, device, clip_grad=None):
    """
    Uma epoca do AppearanceModel.

    Batches: dict com crops (B, T, 3, h, w), observed (B, T, 1),
    labels (B,) identidade, valid (B, T).

    `loss_fn(embeddings, labels)` recebe os embeddings agregados de
    todos os passos validos, (N, D), e os rotulos (N,).
    """

    model.train()

    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:

        batch = _to_device(batch, device)

        out = model(batch["crops"], observed=batch["observed"])

        valid = batch["valid"].bool()
        labels = batch["labels"][:, None].expand(-1, valid.shape[1])

        embeddings = out["embedding"][valid]
        instant = out["instant"][valid]
        labels = labels[valid]

        loss = loss_fn(embeddings, labels) + loss_fn(instant, labels)

        optimizer.zero_grad()
        loss.backward()

        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ============================================================
# HORIZONTE DE MEMORIA  --  norma de dL_t / dh_{t-k}  (Parte 4)
# ============================================================


def gradient_norm_through_time(model, batch, loss_fn, device, t_loss=None, sampling_prob=0.0):
    """
    A curva do gradiente que some (slide 52), no SEU modelo e nos SEUS
    dados: roda a sequencia guardando os h_t, computa a perda SO no
    passo t_loss, faz backward e le ||dL_t / dh_{t-k}|| para k = 0..t.

    Args:
        model: MotionModel (causal).
        batch: um batch como o de train_motion_epoch.
        loss_fn: a mesma do treino.
        t_loss: passo onde a perda e medida (padrao: o ultimo valido).

    Returns:
        np.ndarray (t_loss + 1,) com a norma para k = 0..t_loss
        (indice k = distancia no tempo).
    """

    import numpy as np

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
