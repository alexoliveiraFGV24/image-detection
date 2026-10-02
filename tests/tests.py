"""
Testes unitarios.

    python -m unittest tests.tests -v        (na raiz do repositorio)

Os tres casos construidos a mao da Parte 0 (metrica) estao aqui E no
notebook reports/0_sintetic_tests.ipynb -- o notebook mostra, o teste
garante que continuam passando quando o codigo mudar.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from src.nn.boxes import box_iou_matrix, nms, mot_to_table, table_to_mot
from src.nn.metrics import (
    idf1, id_switches, identity_count_error, evaluate_tracking,
    average_precision, greedy_match, hungarian_match,
)
from src.nn.tracking import Tracker, StaticMotion, KalmanMotion
from src.dataset.sintetic import generate_video, simulate_detector
from src.nn.models import (
    RNNCell, LSTMCell, GRUCell, MotionModel, hidden_size_for_budget, cell_parameter_count,
)
from src.nn.loss import smooth_l1_loss, gaussian_nll_loss, triplet_loss, contrastive_loss
from src.nn.metrics import reacquisition_events, keep_rate_by_gap
from src.nn.tracking import track_sequence
from src.nn.detector import apply_nms
from src.dataset.mot17 import remove_distractor_matches, occlusion_durations, consecutive_iou
from src.analysis.memory import crossing
from src.analysis.stress import degrade_detections


# ------------------------------------------------------------
# helpers: trajetorias sinteticas em linha reta, sem sobreposicao
# ------------------------------------------------------------


def straight_tracks(n_ids=2, n_frames=40, spacing=30.0, size=10.0, speed=1.0):
    """Tabela GT com `n_ids` objetos andando em linhas paralelas."""

    rows = []

    for i in range(n_ids):
        for t in range(n_frames):
            x = 5.0 + t * speed
            y = 5.0 + i * spacing
            rows.append((t, i + 1, x, y, x + size, y + size, 1.0))

    return np.asarray(rows, dtype=np.float64)


# ============================================================
# caixas
# ============================================================


class TestBoxes(unittest.TestCase):

    def test_iou_matrix(self):
        a = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], float)
        b = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [100, 100, 110, 110]], float)
        iou = box_iou_matrix(a, b)
        self.assertEqual(iou.shape, (2, 3))
        self.assertAlmostEqual(iou[0, 0], 1.0)
        self.assertAlmostEqual(iou[0, 1], 25.0 / 175.0)
        self.assertAlmostEqual(iou[0, 2], 0.0)
        self.assertAlmostEqual(iou[1, 0], 0.0)

    def test_iou_empty(self):
        self.assertEqual(box_iou_matrix(np.zeros((0, 4)), np.zeros((3, 4))).shape, (0, 3))

    def test_nms(self):
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60], [0, 0, 10, 10]], float)
        scores = np.array([0.9, 0.8, 0.7, 0.95])
        kept = nms(boxes, scores, iou_threshold=0.5)
        # a mais confiante das sobrepostas (idx 3) e a isolada (idx 2)
        self.assertEqual(list(kept), [3, 2])

    def test_nms_score_threshold(self):
        boxes = np.array([[0, 0, 10, 10], [50, 50, 60, 60]], float)
        kept = nms(boxes, [0.9, 0.1], iou_threshold=0.5, score_threshold=0.5)
        self.assertEqual(list(kept), [0])

    def test_mot_roundtrip(self):
        table = straight_tracks(2, 5)
        mot = table_to_mot(table)
        back = mot_to_table(mot)
        np.testing.assert_allclose(back[:, :6], table[:, :6])
        self.assertTrue((mot[:, 0] >= 1).all())     # 1-indexado


class TestMatching(unittest.TestCase):

    def test_greedy_and_hungarian_agree_on_easy(self):
        iou = np.array([[0.9, 0.1], [0.2, 0.8]])
        self.assertEqual(sorted(greedy_match(iou, 0.5)), [(0, 0), (1, 1)])
        self.assertEqual(sorted(hungarian_match(iou, 0.5)), [(0, 0), (1, 1)])

    def test_hungarian_beats_greedy(self):
        # guloso pega (0,0)=0.6 e deixa (1,1)=0.3 abaixo do limiar;
        # Hungarian pega (0,1)=0.55 e (1,0)=0.5: soma maior, 2 pares
        iou = np.array([[0.6, 0.55], [0.5, 0.3]])
        self.assertEqual(len(greedy_match(iou, 0.4)), 1)
        self.assertEqual(len(hungarian_match(iou, 0.4)), 2)

    def test_threshold(self):
        iou = np.array([[0.3]])
        self.assertEqual(greedy_match(iou, 0.5), [])
        self.assertEqual(hungarian_match(iou, 0.5), [])


# ============================================================
# metricas de rastreamento: os tres casos do enunciado
# ============================================================


class TestTrackingMetrics(unittest.TestCase):

    def setUp(self):
        self.T = 40
        self.k = 20
        self.gt = straight_tracks(n_ids=2, n_frames=self.T)

    def test_a_prediction_equals_gt(self):
        """(a) predicao = ground truth => IDF1 = 1 e zero switches."""
        stats = evaluate_tracking(self.gt.copy(), self.gt, detection_map=True)
        self.assertAlmostEqual(stats["IDF1"], 1.0)
        self.assertEqual(stats["IDSW"], 0)
        self.assertEqual(stats["FRAG"], 0)
        self.assertEqual(stats["count_error"], 0)
        self.assertAlmostEqual(stats["mAP"], 1.0)
        self.assertAlmostEqual(stats["MOTA"], 1.0)

    def test_b_swapped_identities(self):
        """(b) duas identidades trocadas a partir do quadro k => 2 switches, IDF1 = 0.5."""
        pred = self.gt.copy()
        late = pred[:, 0] >= self.k
        pred[late & (self.gt[:, 1] == 1), 1] = 2
        pred[late & (self.gt[:, 1] == 2), 1] = 1

        sw = id_switches(pred, self.gt)
        self.assertEqual(sw["IDSW"], 2)             # uma troca por identidade, no quadro k
        self.assertEqual(sw["FRAG"], 0)
        self.assertEqual({e[0] for e in sw["switch_events"]}, {self.k})

        f1 = idf1(pred, self.gt)
        self.assertAlmostEqual(f1["IDF1"], 0.5)     # cada gt so cobra metade

    def test_c_track_split(self):
        """(c) uma track partida em duas no meio => 1 switch, IDF1 = 0.75 (!= de b)."""
        pred = self.gt.copy()
        pred[(pred[:, 0] >= self.k) & (self.gt[:, 1] == 1), 1] = 3

        sw = id_switches(pred, self.gt)
        self.assertEqual(sw["IDSW"], 1)
        self.assertEqual(sw["FRAG"], 0)

        f1 = idf1(pred, self.gt)
        self.assertAlmostEqual(f1["IDF1"], 0.75)
        self.assertEqual(f1["assignment"][2], 2)
        self.assertIn(f1["assignment"][1], (1, 3))

        count = identity_count_error(pred, self.gt)
        self.assertEqual(count["count_error"], 1)
        self.assertAlmostEqual(count["id_ratio"], 1.5)

    def test_fragmentation(self):
        """Um buraco no meio de uma track (sem trocar de id) e fragmentacao, nao switch."""
        pred = self.gt.copy()
        hole = (pred[:, 1] == 1) & (pred[:, 0] >= 10) & (pred[:, 0] < 15)
        pred = pred[~hole]

        sw = id_switches(pred, self.gt)
        self.assertEqual(sw["IDSW"], 0)
        self.assertEqual(sw["FRAG"], 1)
        self.assertEqual(sw["FN"], 5)

    def test_idf1_missing_and_extra(self):
        """Deteccoes a mais so baixam IDP; a menos, so IDR."""
        pred = self.gt.copy()
        pred = pred[~((pred[:, 1] == 2) & (pred[:, 0] >= 30))]     # perde 10 quadros
        f1 = idf1(pred, self.gt)
        self.assertAlmostEqual(f1["IDP"], 1.0)
        self.assertAlmostEqual(f1["IDR"], 70 / 80)

    def test_average_precision_perfect_and_half(self):
        gt = self.gt
        self.assertAlmostEqual(average_precision(gt.copy(), gt, 0.5), 1.0)
        half = gt[gt[:, 1] == 1].copy()
        self.assertAlmostEqual(average_precision(half, gt, 0.5), 0.5)


# ============================================================
# gerador, detector simulado, rastreador
# ============================================================


class TestSynthetic(unittest.TestCase):

    def test_generator_shapes_and_gt(self):
        v = generate_video(n_objects=6, speed=1.0, n_frames=12, seed=1)
        self.assertEqual(v.frames.shape, (12, 128, 128))
        self.assertEqual(v.label_maps.shape, (12, 128, 128))
        self.assertTrue((v.frames >= 0).all() and (v.frames <= 1).all())
        self.assertEqual(len(np.unique(v.gt[:, 1])), 6)
        self.assertTrue((v.gt[:, 6] >= 0).all() and (v.gt[:, 6] <= 1).all())

    def test_generator_is_deterministic(self):
        a = generate_video(n_objects=5, speed=2.0, n_frames=10, seed=7)
        b = generate_video(n_objects=5, speed=2.0, n_frames=10, seed=7)
        np.testing.assert_array_equal(a.frames, b.frames)
        np.testing.assert_array_equal(a.gt, b.gt)

    def test_depth_order_makes_real_occlusion(self):
        """O ocultado precisa realmente sumir (visibilidade 0) e voltar."""
        v = generate_video(n_objects=6, speed=2.0, occlusion_duration=10, n_frames=50, seed=3)
        occ = v.occlusions[0]
        intervals = v.occlusion_intervals(0.0).get(occ["occludee"])
        self.assertIsNotNone(intervals)
        length = intervals[0][1] - intervals[0][0] + 1
        self.assertTrue(8 <= length <= 13, length)
        # o ocultador fica 100% visivel enquanto cobre o ocultado
        s, e = intervals[0]
        vis_occluder = v.visibility_of(occ["occluder"])[s:e + 1]
        self.assertTrue((vis_occluder > 0.99).all())

    def test_detector_simulator(self):
        v = generate_video(n_objects=6, speed=1.0, n_frames=20, seed=2)
        perfect = simulate_detector(v.gt, min_visibility=0.0, seed=0)
        np.testing.assert_allclose(perfect[:, 2:6], v.gt[:, 2:6])
        self.assertTrue((perfect[:, 1] == -1).all())

        dropped = simulate_detector(v.gt, drop_rate=0.5, min_visibility=0.0, seed=0)
        self.assertLess(len(dropped), len(v.gt))

        with_fp = simulate_detector(v.gt, fp_rate=2.0, min_visibility=0.0, seed=0)
        self.assertGreater(len(with_fp), len(v.gt))

        noisy = simulate_detector(v.gt, noise_std=0.1, min_visibility=0.0, seed=0)
        self.assertTrue((noisy[:, 4] > noisy[:, 2]).all() and (noisy[:, 5] > noisy[:, 3]).all())

    def test_baseline_on_easy_floor(self):
        """Poucas elipses, lentas, sem oclusao, deteccao perfeita => IDF1 ~ 1."""
        v = generate_video(n_objects=4, speed=0.5, occlusion_duration=0, n_frames=40, seed=11)
        det = simulate_detector(v.gt, min_visibility=0.0)
        for motion in (StaticMotion(), KalmanMotion()):
            pred = Tracker(motion, iou_threshold=0.3, max_age=5).run(det)
            stats = evaluate_tracking(pred, v.gt, detection_map=False)
            self.assertGreater(stats["IDF1"], 0.95, motion)

    def test_tracker_birth_and_death(self):
        gt = straight_tracks(n_ids=1, n_frames=20, speed=0.0)     # objeto parado
        det = gt.copy()
        det[:, 1] = -1
        det = det[(det[:, 0] < 5) | (det[:, 0] >= 12)]     # buraco de 7 quadros

        pred = Tracker(StaticMotion(), iou_threshold=0.3, max_age=3).run(det)
        self.assertEqual(len(np.unique(pred[:, 1])), 2)     # morreu e nasceu de novo

        pred = Tracker(StaticMotion(), iou_threshold=0.3, max_age=10).run(det)
        self.assertEqual(len(np.unique(pred[:, 1])), 1)     # sobreviveu ao buraco

    def test_static_loses_moving_object_kalman_keeps_it(self):
        """No buraco, a caixa parada fica para tras de um objeto que anda; a de velocidade constante nao."""
        gt = straight_tracks(n_ids=1, n_frames=20, speed=1.5)
        det = gt.copy()
        det[:, 1] = -1
        det = det[(det[:, 0] < 6) | (det[:, 0] >= 12)]     # 6 quadros sem observacao, 9 px andados

        static = Tracker(StaticMotion(), iou_threshold=0.3, max_age=10).run(det)
        kalman = Tracker(KalmanMotion(), iou_threshold=0.3, max_age=10).run(det)

        self.assertEqual(len(np.unique(static[:, 1])), 2)
        self.assertEqual(len(np.unique(kalman[:, 1])), 1)


# ============================================================
# modelos e perdas
# ============================================================


class TestModels(unittest.TestCase):

    def test_cells_shapes_and_budget(self):
        for cell_class in (RNNCell, LSTMCell, GRUCell):
            cell = cell_class(8, 16)
            state = cell.init_state(3)
            h, state = cell(torch.randn(3, 8), state)
            self.assertEqual(h.shape, (3, 16))
            self.assertEqual(cell.hidden(state).shape, (3, 16))

        for kind in ("rnn", "lstm", "gru"):
            H = hidden_size_for_budget(kind, 11, 20000)
            self.assertLessEqual(cell_parameter_count(kind, 11, H), 20000)
            self.assertGreater(cell_parameter_count(kind, 11, H + 1), 20000)

    def test_motion_model_forward_and_step_agree(self):
        torch.manual_seed(0)
        model = MotionModel("lstm", hidden_size=16, predict_uncertainty=True).eval()

        boxes = torch.rand(2, 6, 4)
        out = model(boxes)
        self.assertEqual(out["box"].shape, (2, 6, 4))
        self.assertEqual(out["log_var"].shape, (2, 6, 4))

        # passo a passo (teacher forcing) da o mesmo que a sequencia
        state = model.init_state(2)
        prev = boxes[:, 0]
        ones = torch.ones(2, 1)
        for t in range(6):
            step = model.step(boxes[:, t], prev, ones, state, conf=ones, dt=ones)
            torch.testing.assert_close(step["box"], out["box"][:, t])
            state = step["state"]
            prev = boxes[:, t]

    def test_free_running_uses_own_prediction(self):
        torch.manual_seed(0)
        model = MotionModel("gru", hidden_size=16).eval()
        boxes = torch.rand(1, 5, 4)
        out = model(boxes, sampling_prob=1.0)
        # a partir de t=1 a entrada e a previsao anterior, nao o GT
        torch.testing.assert_close(out["input"][:, 1:], out["box"][:, :-1])

    def test_losses(self):
        pred = torch.zeros(2, 3, 4, requires_grad=True)
        target = torch.ones(2, 3, 4)
        mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

        loss = smooth_l1_loss(pred, target, mask=mask)
        loss.backward()
        self.assertGreater(loss.item(), 0)
        self.assertEqual(float(pred.grad[0, 2].abs().sum()), 0.0)     # passo mascarado sem gradiente

        nll = gaussian_nll_loss(pred, target, log_var=torch.zeros(2, 3, 4), mask=mask)
        self.assertAlmostEqual(nll.item(), 0.5, places=5)

        emb = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        self.assertGreaterEqual(float(triplet_loss(emb, labels)), 0.0)
        self.assertGreaterEqual(float(contrastive_loss(emb, labels)), 0.0)

        # embeddings perfeitos: perda zero
        perfect = torch.nn.functional.one_hot(labels, 16).float()
        self.assertAlmostEqual(float(triplet_loss(perfect, labels, margin=0.3)), 0.0)


# ============================================================
# Parte 1: MOT17, recaptura, NMS sobre deteccoes cruas
# ============================================================


class TestPart1(unittest.TestCase):

    def test_distractor_filter(self):
        """Previsao sobre um distrator (classe 7) sai; sobre um pedestre ou um carro (classe 3) fica."""
        # gt.txt cru: frame (1-idx), id, left, top, w, h, conf, class, vis
        gt_all = np.array([
            [1, 1, 0, 0, 10, 20, 1, 1, 1.0],      # pedestre avaliado
            [1, 2, 50, 0, 10, 20, 0, 7, 1.0],     # pessoa estatica (distrator)
            [1, 3, 100, 0, 30, 20, 0, 3, 1.0],    # carro: nao e distrator
        ])
        pred = np.array([
            [0, 10, 0, 0, 10, 20, 0.9],
            [0, 11, 50, 0, 60, 20, 0.9],
            [0, 12, 100, 0, 130, 20, 0.9],
        ])
        kept = remove_distractor_matches(pred, gt_all)
        self.assertEqual(sorted(kept[:, 1].tolist()), [10, 12])

    def test_reacquisition_events(self):
        gt = straight_tracks(n_ids=1, n_frames=30)
        pred = gt[(gt[:, 0] < 10) | (gt[:, 0] >= 14)].copy()       # buraco de 4 quadros
        events = [e for e in reacquisition_events(pred, gt) if e["gap"] > 0]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["gap"], 4)
        self.assertTrue(events[0]["kept"])

        pred[pred[:, 0] >= 14, 1] = 9                                # volta com outro id
        events = [e for e in reacquisition_events(pred, gt) if e["gap"] > 0]
        self.assertFalse(events[0]["kept"])
        rates = {r["gap_lo"]: r["keep_rate"] for r in keep_rate_by_gap(reacquisition_events(pred, gt))}
        self.assertEqual(rates[0], 1.0)
        self.assertEqual(rates[3], 0.0)

    def test_apply_nms_per_frame(self):
        raw = np.array([
            [0, -1, 0, 0, 10, 10, 0.9],
            [0, -1, 1, 1, 11, 11, 0.8],      # duplicata da primeira
            [1, -1, 1, 1, 11, 11, 0.8],      # outro quadro: nao compete com o quadro 0
            [1, -1, 50, 50, 60, 60, 0.2],    # abaixo do min_score
        ])
        out = apply_nms(raw, iou_threshold=0.5, min_score=0.5)
        self.assertEqual(out[:, 0].tolist(), [0, 1])
        self.assertEqual(out[:, 6].tolist(), [0.9, 0.8])

    def test_track_sequence_processes_empty_frames(self):
        """Quadros sem deteccao fazem a track envelhecer (e morrer)."""
        gt = straight_tracks(n_ids=1, n_frames=20, speed=0.0)
        det = gt[(gt[:, 0] < 5) | (gt[:, 0] >= 12)].copy()
        det[:, 1] = -1
        pred = track_sequence(det, 20, iou_threshold=0.3, max_age=3)
        self.assertEqual(len(np.unique(pred[:, 1])), 2)

    def test_occlusion_durations_and_consecutive_iou(self):
        gt = straight_tracks(n_ids=1, n_frames=20, speed=0.0)
        gt[(gt[:, 0] >= 5) & (gt[:, 0] < 9), 6] = 0.0               # 4 quadros invisivel, volta
        self.assertEqual(occlusion_durations(gt, max_visibility=0.25).tolist(), [4])
        np.testing.assert_allclose(consecutive_iou(gt), 1.0)        # parado: IoU 1 entre quadros


class TestPart4(unittest.TestCase):

    def test_crossing_interpolates_50_percent(self):
        rates = [{"gap_lo": 1, "gap_hi": 1, "keep_rate": 0.9}, {"gap_lo": 3, "gap_hi": 3, "keep_rate": 0.1}]
        self.assertAlmostEqual(crossing(rates), 2.0)

    def test_free_running_model_keeps_shape(self):
        model = MotionModel(hidden_size=8, use_conf=False, use_dt=False)
        boxes = torch.rand(2, 10, 4)
        observed = torch.ones(2, 10, 1)
        observed[:, 4:] = 0
        self.assertEqual(model(boxes, observed=observed)["box"].shape, (2, 10, 4))


class TestPart5(unittest.TestCase):

    def test_degrade_detections(self):
        det = straight_tracks(n_ids=3, n_frames=50, speed=1.0)
        det[:, 1] = -1
        det = det[np.argsort(det[:, 0], kind="stable")]
        self.assertTrue(np.allclose(degrade_detections(det, image_size=(1000, 1000), n_frames=50), det))
        out = degrade_detections(det, drop=0.5, fp_per_frame=2.0, image_size=(1000, 1000), n_frames=50, seed=0)
        n_kept = len(out) - np.sum(~np.isin(out[:, 2], det[:, 2]))
        self.assertLess(n_kept, len(det))
        self.assertGreater(len(out), n_kept)
        self.assertTrue(np.all(out[:, 4] <= 1000) and np.all(out[:, 2] >= 0))


if __name__ == "__main__":
    unittest.main()
