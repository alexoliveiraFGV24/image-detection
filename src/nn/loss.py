"""
Funcoes de perda.

Trilha A (caixas, com mascara de padding / passos validos):
    l1_loss, smooth_l1_loss      L1 / smooth-L1 sobre a caixa (cx, cy, w, h)
    gaussian_nll_loss            log-verossimilhanca gaussiana com a
                                 incerteza prevista (portao adaptativo)
    giou_loss                    1 - GIoU (opcional, complementa a L1)
    make_box_loss                fabrica: nome -> funcao com a assinatura
                                 loss(pred, target, log_var=None, mask=None)

Trilha B (embeddings e identidades):
    contrastive_loss             pares: mesma id puxa, id diferente empurra
                                 alem de uma margem
    triplet_loss                 batch-hard: ancora, positivo mais longe,
                                 negativo mais perto

Todas devolvem um escalar conectado ao grafo, mesmo com mascara vazia.
"""

import torch
import torch.nn.functional as F

from src.nn.boxes import cxcywh_to_xyxy


# ============================================================
# TRILHA A  --  perdas sobre a caixa
# ============================================================


def _masked_mean(per_step, mask):
    """
    Media de `per_step` (B, T) sobre os passos onde mask (B, T) = 1.
    Sem mascara, media simples. Mascara vazia -> zero conectado ao grafo.
    """

    if mask is None:
        return per_step.mean()

    mask = mask.to(per_step.dtype)
    denominator = mask.sum()

    if float(denominator) == 0:
        return per_step.sum() * 0.0

    return (per_step * mask).sum() / denominator


def l1_loss(pred, target, log_var=None, mask=None):
    """
    L1 sobre as 4 coordenadas normalizadas (media por passo valido).
    `log_var` e ignorado (assinatura comum a todas as perdas de caixa).
    """

    per_step = (pred - target).abs().mean(dim=-1)
    return _masked_mean(per_step, mask)


def smooth_l1_loss(pred, target, log_var=None, mask=None, beta=0.02):
    """
    Smooth-L1 (Huber): quadratica ate `beta`, linear depois. Em
    coordenadas normalizadas, beta=0.02 ~ 2.5 px numa imagem de 128.
    """

    per_step = F.smooth_l1_loss(pred, target, reduction="none", beta=beta).mean(dim=-1)
    return _masked_mean(per_step, mask)


def gaussian_nll_loss(pred, target, log_var=None, mask=None):
    """
    Log-verossimilhanca gaussiana negativa por coordenada:

        NLL = 0.5 * [ (target - pred)^2 / sigma^2 + log sigma^2 ]

    com log sigma^2 previsto pelo modelo. Onde o modelo esta inseguro
    (oclusao longa) ele pode "abrir" sigma e pagar menos pelo erro --
    e essa sigma vira o portao adaptativo da associacao (Trilha A).
    Se `log_var` for None, recai na L2 com sigma = 1.
    """

    if log_var is None:
        log_var = torch.zeros_like(pred)

    per_coord = 0.5 * ((target - pred) ** 2 * torch.exp(-log_var) + log_var)
    per_step = per_coord.mean(dim=-1)

    return _masked_mean(per_step, mask)


def giou_loss(pred, target, log_var=None, mask=None):
    """
    1 - GIoU entre caixas (cx, cy, w, h). Diferente da L1, e invariante
    a escala e penaliza caixas que nem se tocam pela area do envelope.
    """

    p = cxcywh_to_xyxy(pred)
    t = cxcywh_to_xyxy(target)

    ix1 = torch.maximum(p[..., 0], t[..., 0])
    iy1 = torch.maximum(p[..., 1], t[..., 1])
    ix2 = torch.minimum(p[..., 2], t[..., 2])
    iy2 = torch.minimum(p[..., 3], t[..., 3])

    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)

    area_p = (p[..., 2] - p[..., 0]).clamp(min=0) * (p[..., 3] - p[..., 1]).clamp(min=0)
    area_t = (t[..., 2] - t[..., 0]).clamp(min=0) * (t[..., 3] - t[..., 1]).clamp(min=0)

    union = area_p + area_t - inter
    iou = inter / union.clamp(min=1e-9)

    ex1 = torch.minimum(p[..., 0], t[..., 0])
    ey1 = torch.minimum(p[..., 1], t[..., 1])
    ex2 = torch.maximum(p[..., 2], t[..., 2])
    ey2 = torch.maximum(p[..., 3], t[..., 3])

    enclosing = (ex2 - ex1).clamp(min=0) * (ey2 - ey1).clamp(min=0)

    giou = iou - (enclosing - union) / enclosing.clamp(min=1e-9)

    return _masked_mean(1.0 - giou, mask)


BOX_LOSSES = {
    "l1": l1_loss,
    "smooth_l1": smooth_l1_loss,
    "gaussian_nll": gaussian_nll_loss,
    "giou": giou_loss,
}


def make_box_loss(name="smooth_l1", giou_weight=0.0, **kwargs):
    """
    Fabrica de perda de caixa com a assinatura
    ``loss(pred, target, log_var=None, mask=None)``.

    Args:
        name: "l1" | "smooth_l1" | "gaussian_nll" | "giou".
        giou_weight: se > 0, soma giou_weight * (1 - GIoU) a perda.
        **kwargs: repassados (ex.: beta da smooth-L1).
    """

    base = BOX_LOSSES[name]

    def loss(pred, target, log_var=None, mask=None):

        value = base(pred, target, log_var=log_var, mask=mask, **kwargs)

        if giou_weight > 0:
            value = value + giou_weight * giou_loss(pred, target, mask=mask)

        return value

    loss.__name__ = name if giou_weight == 0 else f"{name}+{giou_weight}giou"

    return loss


# ============================================================
# TRILHA B  --  perdas sobre embeddings
# ============================================================


def _pairwise_cosine_distance(embeddings):
    """(N, D) normalizados -> (N, N) com 1 - cos."""

    e = F.normalize(embeddings, dim=-1)
    return 1.0 - e @ e.t()


def contrastive_loss(embeddings, labels, margin=0.5):
    """
    Perda contrastiva sobre todos os pares do batch (distancia de cosseno):

        mesma identidade:   d^2
        identidades !=  :   max(0, margin - d)^2

    Args:
        embeddings: (N, D).
        labels: (N,) identidade de cada embedding.
        margin: distancia minima desejada entre identidades diferentes.
    """

    N = embeddings.shape[0]

    if N < 2:
        return embeddings.sum() * 0.0

    d = _pairwise_cosine_distance(embeddings)

    same = labels[:, None] == labels[None, :]
    off_diagonal = ~torch.eye(N, dtype=torch.bool, device=embeddings.device)

    positive = same & off_diagonal
    negative = ~same

    pull = (d ** 2)[positive]
    push = (F.relu(margin - d) ** 2)[negative]

    terms = []

    if pull.numel():
        terms.append(pull.mean())

    if push.numel():
        terms.append(push.mean())

    if not terms:
        return embeddings.sum() * 0.0

    return sum(terms) / len(terms)


def triplet_loss(embeddings, labels, margin=0.3):
    """
    Triplet batch-hard (Hermans et al., 2017): para cada ancora, o
    positivo MAIS DISTANTE e o negativo MAIS PROXIMO do batch:

        max(0, d(a, p_hard) - d(a, n_hard) + margin)

    Ancoras sem positivo ou sem negativo no batch sao ignoradas.
    """

    N = embeddings.shape[0]

    if N < 2:
        return embeddings.sum() * 0.0

    d = _pairwise_cosine_distance(embeddings)

    same = labels[:, None] == labels[None, :]
    off_diagonal = ~torch.eye(N, dtype=torch.bool, device=embeddings.device)

    positive = same & off_diagonal
    negative = ~same

    has_both = positive.any(dim=1) & negative.any(dim=1)

    if not has_both.any():
        return embeddings.sum() * 0.0

    hardest_positive = torch.where(positive, d, torch.full_like(d, -1.0)).max(dim=1).values
    hardest_negative = torch.where(negative, d, torch.full_like(d, 4.0)).min(dim=1).values

    loss = F.relu(hardest_positive - hardest_negative + margin)

    return loss[has_both].mean()


EMBEDDING_LOSSES = {
    "contrastive": contrastive_loss,
    "triplet": triplet_loss,
}


def make_embedding_loss(name="triplet", **kwargs):
    """Fabrica: ``loss(embeddings, labels)``."""

    base = EMBEDDING_LOSSES[name]

    def loss(embeddings, labels):
        return base(embeddings, labels, **kwargs)

    loss.__name__ = name

    return loss
