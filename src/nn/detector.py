"""
Segunda fonte de deteccoes da Parte 1: um detector pre-treinado do
torchvision (Faster R-CNN, COCO), em modo de inferencia, classe `person`.

O enunciado proibe torchvision.ops.nms. O Faster R-CNN do torchvision
aplica NMS em dois lugares:

  * dentro da RPN, para reduzir as propostas -- e parte da arquitetura do
    detector (permitido: "torchvision.models.detection"), fica como esta;
  * no final (RoIHeads.postprocess_detections), sobre as deteccoes de
    cada classe -- esse e o NMS que transforma "caixas" em "deteccoes".
    Ele e DESLIGADO aqui (nms_thresh = 1.0: o NMS do torchvision so
    suprime caixas com IoU > limiar, e IoU <= 1 sempre) e substituido
    pelo nosso (src/nn/boxes.py::nms) em `apply_nms`.

Propostas: 300 por imagem no teste (rpn post-NMS), como no artigo
original do Faster R-CNN (Ren et al., 2015); o padrao do torchvision e
1000. A cabeca do v2 (4 convolucoes sobre cada proposta 7x7) custa mais
que o backbone; com 300 propostas o custo cai ~30% e, no quadro mais
denso do MOT17-04 (45 pessoas), as deteccoes finais sao as mesmas.

As deteccoes cruas (pre-NMS, score >= 0.05, so `person`) ficam em cache
em disco por sequencia: rodar o detector nas 5.316 imagens de treino
custa horas em CPU, e o notebook so precisa disso uma vez.

Linha de comando (o que gerou o cache usado nos notebooks):

    python -m src.nn.detector --sequences MOT17-09 MOT17-04 --threads 4
"""

import argparse
import os
import time

import numpy as np

from src.nn.boxes import FRAME, X1, Y2, CONF, nms, split_by_frame, empty_table, make_table


COCO_PERSON = 1

ARCHITECTURES = {
    "fasterrcnn_resnet50_fpn_v2": ("fasterrcnn_resnet50_fpn_v2", "FasterRCNN_ResNet50_FPN_V2_Weights"),
    "fasterrcnn_resnet50_fpn": ("fasterrcnn_resnet50_fpn", "FasterRCNN_ResNet50_FPN_Weights"),
    "fasterrcnn_mobilenet_v3_large_fpn": ("fasterrcnn_mobilenet_v3_large_fpn", "FasterRCNN_MobileNet_V3_Large_FPN_Weights"),
}

DEFAULT_ARCH = "fasterrcnn_resnet50_fpn_v2"
RPN_PROPOSALS = 300


def load_detector(arch=DEFAULT_ARCH, device="cpu", score_thresh=0.05, max_detections=1000,
                  proposals=RPN_PROPOSALS):
    """
    Faster R-CNN pre-treinado no COCO, em eval, com o NMS final desligado.

    Args:
        arch: chave de ARCHITECTURES.
        score_thresh: score minimo das caixas que saem do detector.
        max_detections: teto de caixas por imagem (sem NMS final, cada
            pessoa gera varias caixas quase iguais -- o teto padrao de
            100 cortaria pessoas em cenas densas).
        proposals: propostas da RPN por imagem no teste (ver o cabecalho).
    """

    import torchvision.models.detection as detection

    builder_name, weights_name = ARCHITECTURES[arch]

    weights = getattr(detection, weights_name).DEFAULT
    model = getattr(detection, builder_name)(weights=weights)

    model.roi_heads.nms_thresh = 1.0                 # NMS final desligado
    model.roi_heads.score_thresh = score_thresh
    model.roi_heads.detections_per_img = max_detections
    model.rpn._post_nms_top_n["testing"] = proposals

    return model.to(device).eval()


def detect_images(model, paths, device="cpu"):
    """
    Roda o detector numa lista de imagens.

    Returns:
        lista (uma por imagem) de arrays (K, 5) = (x1, y1, x2, y2, score)
        so da classe person, SEM NMS.
    """

    import torch
    from PIL import Image
    import torchvision.transforms.functional as TF

    outputs = []

    with torch.inference_mode():

        for path in paths:

            image = TF.to_tensor(Image.open(path).convert("RGB")).to(device)
            out = model([image])[0]

            person = out["labels"] == COCO_PERSON
            boxes = out["boxes"][person].cpu().numpy()
            scores = out["scores"][person].cpu().numpy()

            outputs.append(np.column_stack([boxes, scores]).astype(np.float32))

    return outputs


def cache_path(seq_name, arch=DEFAULT_ARCH, cache_dir="../data/detections"):
    return os.path.join(cache_dir, arch, f"{seq_name}.npy")


def run_on_sequence(seq, model=None, arch=DEFAULT_ARCH, device="cpu", cache_dir="../data/detections",
                    chunk=25, verbose=True):
    """
    Deteccoes cruas (pre-NMS) de uma sequencia inteira, com cache.

    Se o cache existe, so carrega. Senao roda o detector quadro a quadro
    e grava um arquivo parcial a cada `chunk` quadros -- se o processo
    cair, a proxima chamada continua de onde parou.

    Returns:
        tabela (N, 7) = (frame, -1, x1, y1, x2, y2, score), sem NMS.
    """

    path = cache_path(seq.name, arch, cache_dir)

    if os.path.exists(path):
        return np.load(path).astype(np.float64)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    partial = path.replace(".npy", ".partial.npz")

    rows = []
    start = 0
    elapsed = 0.0

    if os.path.exists(partial):
        data = np.load(partial)
        rows = [data["rows"]]
        start = int(data["next_frame"])
        elapsed = float(data["elapsed"])

    if model is None:
        model = load_detector(arch, device)

    for t0 in range(start, seq.n_frames, chunk):

        frames = list(range(t0, min(t0 + chunk, seq.n_frames)))

        tic = time.time()
        outputs = detect_images(model, [seq.image_path(t) for t in frames], device)
        elapsed += time.time() - tic

        for t, out in zip(frames, outputs):
            rows.append(make_table(np.full(len(out), t), -np.ones(len(out)), out[:, :4], out[:, 4]).astype(np.float32))

        np.savez(partial, rows=np.concatenate(rows) if rows else np.zeros((0, 7), np.float32),
                 next_frame=frames[-1] + 1, elapsed=elapsed)

        if verbose:
            done = frames[-1] + 1
            print(f"{seq.name}: {done}/{seq.n_frames} quadros, {elapsed / done:.2f} s/quadro", flush=True)

    table = np.concatenate(rows) if rows else np.zeros((0, 7), np.float32)
    np.save(path, table.astype(np.float32))

    with open(path.replace(".npy", ".time.txt"), "w") as fh:
        fh.write(f"{elapsed:.1f} s para {seq.n_frames} quadros ({elapsed / seq.n_frames:.3f} s/quadro)\n")

    if os.path.exists(partial):
        os.remove(partial)

    return table.astype(np.float64)


def apply_nms(raw, iou_threshold=0.5, min_score=0.0):
    """
    O NOSSO NMS, quadro a quadro, sobre as deteccoes cruas.

    Args:
        raw: tabela (N, 7) sem NMS.
        iou_threshold: limiar de supressao.
        min_score: deteccoes com score < isto saem antes do NMS.

    Returns:
        tabela (M, 7).
    """

    raw = np.asarray(raw, dtype=np.float64).reshape(-1, 7)
    raw = raw[raw[:, CONF] >= min_score]

    if len(raw) == 0:
        return empty_table()

    kept = []

    for _, rows in split_by_frame(raw).items():
        keep = nms(rows[:, X1:Y2 + 1], rows[:, CONF], iou_threshold=iou_threshold, score_threshold=-np.inf)
        kept.append(rows[keep])

    out = np.concatenate(kept)

    return out[np.argsort(out[:, FRAME], kind="stable")]


def read_time(seq_name, arch=DEFAULT_ARCH, cache_dir="../data/detections"):
    """Segundos por quadro registrados quando o cache foi gerado (ou None)."""

    path = cache_path(seq_name, arch, cache_dir).replace(".npy", ".time.txt")

    if not os.path.exists(path):
        return None

    text = open(path).read()
    return float(text.split("(")[1].split(" ")[0])


def main(argv=None):

    import torch

    from src.dataset.mot17 import MOT17Sequence, SEQUENCES

    parser = argparse.ArgumentParser(description="Roda o Faster R-CNN do torchvision nas sequencias do MOT17 (cache)")
    parser.add_argument("--sequences", nargs="+", default=SEQUENCES)
    parser.add_argument("--arch", default=DEFAULT_ARCH, choices=sorted(ARCHITECTURES))
    parser.add_argument("--root", default="data")
    parser.add_argument("--cache-dir", default="data/detections")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args(argv)

    if args.threads:
        torch.set_num_threads(args.threads)

    model = load_detector(args.arch, args.device)

    for name in args.sequences:
        seq = MOT17Sequence(name, args.root)
        table = run_on_sequence(seq, model, args.arch, args.device, args.cache_dir)
        print(f"{name}: {len(table)} caixas cruas de pessoa -> {cache_path(name, args.arch, args.cache_dir)}", flush=True)


if __name__ == "__main__":
    main()
