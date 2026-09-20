"""
Cria um otimizador (e, opcionalmente, um scheduler) a partir do nome.

    model = MotionModel("gru", hidden_size=64).to(device)
    optimizer = create_optimizer("Adam", model, lr=1e-3)
    scheduler = create_scheduler("cosine", optimizer, T_max=NUM_EPOCHS)

O gradient clipping NAO mora aqui: e um argumento (`clip_grad`) dos
loops de treino em src/nn/models.py, porque a Parte 3 (Eixo 2) compara
ligado / desligado e precisa reportar a norma antes do corte.
"""

import torch


otimizadores = {
    "SGD": torch.optim.SGD,
    "Nesterov": lambda params, **kw: torch.optim.SGD(params, nesterov=True, momentum=kw.pop("momentum", 0.9), **kw),
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
    "Adagrad": torch.optim.Adagrad,
    "RMSProp": torch.optim.RMSprop,
}


schedulers = {
    "step": torch.optim.lr_scheduler.StepLR,
    "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
    "plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "exponential": torch.optim.lr_scheduler.ExponentialLR,
}


def create_optimizer(nome, model, **params):
    """
    Cria um otimizador com base no nome e parametros fornecidos.

    Args:
        nome (str): "SGD" | "Nesterov" | "Adam" | "AdamW" | "Adagrad" | "RMSProp".
        model (torch.nn.Module ou iteravel de parametros).
        **params: parametros do otimizador (lr, weight_decay, ...).

    Returns:
        torch.optim.Optimizer
    """

    if nome not in otimizadores:
        raise ValueError(f"Otimizador '{nome}' nao encontrado. Opcoes: {list(otimizadores.keys())}")

    parameters = model.parameters() if isinstance(model, torch.nn.Module) else model

    return otimizadores[nome](parameters, **params)


def create_scheduler(nome, optimizer, **params):
    """
    Cria um scheduler de taxa de aprendizado.

    Args:
        nome (str): "step" | "cosine" | "plateau" | "exponential".
    """

    if nome not in schedulers:
        raise ValueError(f"Scheduler '{nome}' nao encontrado. Opcoes: {list(schedulers.keys())}")

    return schedulers[nome](optimizer, **params)
