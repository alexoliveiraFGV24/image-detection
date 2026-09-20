"""
metrics.py -- entregavel do PA2: implementacao propria de IDF1, ID
switches e fragmentacoes (mais mAP por quadro, erro de contagem de
identidades e MOTA).

A implementacao mora em ``src/nn/metrics.py`` (para ser importada pelos
notebooks junto com o resto de ``src``); este arquivo a re-exporta e
oferece uma linha de comando que avalia dois arquivos no formato do
MOTChallenge:

    python metrics.py --gt data/MOT17Labels/train/MOT17-02-FRCNN/gt/gt.txt \\
                      --pred resultados/MOT17-02-FRCNN.txt [--iou 0.5]

Uso em codigo:

    from metrics import idf1, id_switches, evaluate_tracking
    stats = evaluate_tracking(pred_table, gt_table, iou_threshold=0.5)

onde as tabelas sao arrays (N, 7) = (frame, id, x1, y1, x2, y2, conf).
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from src.nn.metrics import (          # noqa: F401  (re-export)
    idf1,
    id_switches,
    identity_count_error,
    mota,
    evaluate_tracking,
    average_precision,
    mean_average_precision,
    detection_counts,
    greedy_match,
    hungarian_match,
    frame_matching,
)
from src.nn.boxes import mot_to_table, nms, box_iou_matrix   # noqa: F401


def load_mot(path, gt=False, keep_classes=(1,), min_visibility=0.0):
    """
    Le um arquivo do MOTChallenge e devolve a tabela (N, 7).

    Para o gt.txt (gt=True): mantem so as linhas com conf = 1 (as
    consideradas na avaliacao), classe em `keep_classes` (1 = pedestre)
    e visibilidade >= min_visibility; `conf` recebe a visibilidade.
    """

    rows = np.loadtxt(path, delimiter=",", ndmin=2)

    if gt:
        keep = rows[:, 6] == 1
        if rows.shape[1] > 7:
            keep &= np.isin(rows[:, 7], keep_classes)
        if rows.shape[1] > 8:
            keep &= rows[:, 8] >= min_visibility
        rows = rows[keep]
        return mot_to_table(rows, conf_column=8 if rows.shape[1] > 8 else 6)

    return mot_to_table(rows, conf_column=6)


def main(argv=None):

    parser = argparse.ArgumentParser(description="IDF1 / ID switches / fragmentacoes (implementacao propria)")
    parser.add_argument("--gt", required=True, help="gt.txt no formato do MOTChallenge")
    parser.add_argument("--pred", required=True, help="trajetorias previstas (frame, id, left, top, w, h, conf, ...)")
    parser.add_argument("--iou", type=float, default=0.5, help="limiar de IoU (padrao 0.5)")
    parser.add_argument("--min-visibility", type=float, default=0.0, help="ignora GT com visibilidade abaixo disto")
    parser.add_argument("--no-map", action="store_true", help="nao calcula o mAP de deteccao")
    parser.add_argument("--json", default=None, help="grava o resultado neste arquivo")

    args = parser.parse_args(argv)

    gt = load_mot(args.gt, gt=True, min_visibility=args.min_visibility)
    pred = load_mot(args.pred, gt=False)

    stats = evaluate_tracking(pred, gt, iou_threshold=args.iou, detection_map=not args.no_map)

    text = json.dumps(stats, indent=2)
    print(text)

    if args.json:
        with open(args.json, "w") as fh:
            fh.write(text)


if __name__ == "__main__":
    main()
