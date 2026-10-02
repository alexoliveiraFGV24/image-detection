# Identidade ao longo do tempo: detecção, recorrência e rastreamento (PA2 — Aprendizado Profundo)

Este repositório resolve o **Programming Assignment 2** da disciplina de Aprendizado
Profundo (FGV): fazer as arquiteturas das aulas de detecção e de RNN produzirem rótulos
**identity-aware** — se um objeto aparece no quadro 3 e reaparece no quadro 40, ele sai
com o mesmo identificador — **sem rastreador pronto** (nada de SORT/DeepSORT/ByteTrack,
`motmetrics` ou `torchvision.ops.nms`). O modelo temporal, as perdas, a associação, a
gestão de tracks, o NMS e as métricas (IDF1, ID switches, fragmentações) são de nossa
autoria.

Dataset: **MOT17 / MOTChallenge** (pedestres, caixas e identidades anotadas quadro a
quadro; detecções públicas DPM / Faster R-CNN / SDP).

> **Estado atual:** Partes 0 (testes sintéticos), 1 (baseline por quadro no MOT17) e 2
> (memória temporal, Trilha A) concluídas; `src/` contém os modelos, perdas,
> otimizadores, métricas, o rastreador e o detector usados pelas partes seguintes.

---

## 1. Instalação

```bash
git clone <url-do-repositorio>
cd image-detection
```

Crie e ative um ambiente virtual (o projeto foi desenvolvido com **Python 3.11**):

```bash
# Windows (PowerShell)
python -m venv venv
venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

Instale as dependências (PyTorch CPU, torchvision, numpy, scipy, pandas, matplotlib,
scikit-learn, jupyter):

```bash
pip install -r requirements.txt
```

As versões estão fixadas (`torch==2.14.0+cpu`, `torchvision==0.29.0+cpu`, via o índice
CPU do PyTorch declarado no próprio `requirements.txt`). Com GPU, instale primeiro o
`torch`/`torchvision` da sua versão de CUDA seguindo <https://pytorch.org> e depois o
restante do arquivo — os notebooks detectam `cuda` automaticamente.

## 2. Dados

1. **Anotações** (~10 MB, sem conta): <https://motchallenge.net/data/MOT17Labels.zip>.
   Bastam para a métrica, a associação com as detecções públicas e a Trilha A (movimento).
   Descompacte em `data/MOT17Labels/`.
2. **Quadros** (~1,9 GB, sem conta): <https://motchallenge.net/data/MOT17Det.zip>.
   São as mesmas imagens do `MOT17.zip` (5,9 GB), mas uma vez por sequência — o pacote
   completo repete cada sequência três vezes, uma por detector público. Necessários para o
   detector do torchvision (Parte 1), para a Trilha B (aparência) e para as figuras.
   Descompacte em `data/MOT17Det/`. (O `MOT17.zip` em `data/MOT17/` também funciona.)

A pasta `data/` está no `.gitignore`. A estrutura esperada é a original:

```
data/MOT17Labels/train/MOT17-02-FRCNN/gt/gt.txt      # frame, id, left, top, w, h, conf, class, visibility
data/MOT17Labels/train/MOT17-02-FRCNN/det/det.txt    # detecções públicas (idem -DPM, -SDP)
data/MOT17Labels/train/MOT17-02-FRCNN/seqinfo.ini
data/MOT17Det/train/MOT17-02/img1/000001.jpg         # quadros
data/detections/fasterrcnn_resnet50_fpn_v2/MOT17-02.npy   # cache do detector do torchvision (gerado)
```

## 3. Como rodar

Todo o trabalho está em notebooks numerados em `reports/` (uma parte do enunciado por
notebook). Eles devem ser executados **com `reports/` como diretório de trabalho**
(caminhos relativos `../data` e `results/`), o que é o padrão ao abri-los no VS Code /
Jupyter. Para rodar sem abrir o Jupyter:

**Parte 0 — testes sintéticos** (gerador, simulador de detector, testes da métrica,
baseline no piso fácil e o gráfico de onde ele quebra; ~2 min em CPU):

```bash
jupyter nbconvert --to notebook --execute --inplace reports/0_sintetic_tests.ipynb
```

**Parte 1 — baseline por quadro no MOT17.** Primeiro o detector do torchvision nas 7
sequências (gera o cache em `data/detections/`; **~4 h em CPU** — num Ryzen 7 5700U,
~2,6 s/quadro com dois processos de 4 threads; com GPU, minutos):

```bash
python -m src.nn.detector --threads 8
```

Depois o notebook (escolha do detector público, grade da regra de associação, avaliação
das duas fontes, gráfico do descolamento e diagnóstico; ~15 min em CPU com o cache
pronto — sem o cache, o próprio notebook roda o detector):

```bash
jupyter nbconvert --to notebook --execute --inplace reports/1_baseline.ipynb
```

**Testes unitários** (caixas, NMS, matching, os três casos da métrica, gerador,
rastreador, células, perdas, distratores do MOT17, recaptura de identidade):

```bash
python -m unittest tests.tests -v
```

**Métrica em arquivos do MOTChallenge** (o entregável `metrics.py`):

```bash
python metrics.py --gt data/MOT17Labels/train/MOT17-09-FRCNN/gt/gt.txt --pred reports/results/1_baseline_tracks/MOT17-09.txt
```

(`reports/results/1_baseline_tracks/` tem as trajetórias do baseline nas sequências de
validação. O CLI aplica o mesmo pré-processamento oficial dos distratores dos notebooks.)

**Parte 2 — treinar e avaliar o modelo temporal** (GRU como modelo de movimento):
treina a GRU nas trajetórias de treino com as detecções reais do SDP e avalia baseline ×
Kalman × GRU nas 7 sequências; ~45 min em CPU. Se `reports/results/2_motion_gru.pt` existe
o notebook só carrega o checkpoint (apague-o para treinar de novo):

```bash
jupyter nbconvert --to notebook --execute --inplace reports/2_temporal_memory.ipynb
```

O checkpoint é carregável fora do notebook (ver `checkpoint.json`):

```python
from src.nn.models import load_motion_model
from src.nn.tracking import track_sequence, RNNMotion
model, _ = load_motion_model("reports/results/2_motion_gru.pt")
tracks = track_sequence(detections, n_frames, motion=RNNMotion(model, image_height, fps),
                        matching="hungarian", iou_threshold=0.3, max_age=40)
```

## 4. Estrutura do repositório

```
image-detection/
├── README.md
├── AI_LOG.md                      # como a IA foi usada (episódios)
├── requirements.txt
├── metrics.py                     # entregável: IDF1, ID switches, fragmentações (+ CLI para arquivos MOT)
├── inferencia.ipynb               # entregável: inferência numa sequência qualquer (a preencher)
├── checkpoint.json                # onde estão os pesos do modelo temporal e como carregá-los
├── assignment/
│   ├── PA2.pdf                    # enunciado
│   ├── 06) Detecção de objetos.pdf
│   └── 07) RNN.pdf                # slides das aulas
├── data/                          # NÃO versionado — MOT17Labels, MOT17Det (quadros), cache do detector
├── src/
│   ├── dataset/
│   │   ├── sintetic.py            # Parte 0: generate_video (elipses com ordem de profundidade e
│   │   │                          #   oclusão de duração controlada), simulate_detector, SyntheticVideoDataset
│   │   ├── mot17.py               # MOT17Sequence (GT, detecções públicas, quadros), split por sequência,
│   │   │                          #   pré-processamento oficial dos distratores, eixos de dificuldade
│   │   └── trajectories.py        # trajetórias do GT + detecções reais casadas (DetectorReplay) em
│   │                              #   janelas para o MotionModel; ruído simulado (DetectionNoise)
│   ├── nn/
│   │   ├── boxes.py               # IoU vetorizado, NMS próprio, conversões, tabelas (frame, id, x1, y1, x2, y2, conf)
│   │   ├── metrics.py             # AP/mAP por quadro; IDF1, ID switches, fragmentações, contagem de ids, MOTA
│   │   ├── tracking.py            # Tracker (associação + nascimento/morte) e modelos de movimento:
│   │   │                          #   StaticMotion (Parte 1), KalmanMotion (baseline), RNNMotion (Parte 2)
│   │   ├── detector.py            # Faster R-CNN do torchvision (COCO, person) com o NMS final trocado
│   │   │                          #   pelo nosso; cache por sequência; CLI
│   │   ├── models.py              # RNNCell/LSTMCell/GRUCell do zero, Recurrent (bidirecional opcional),
│   │   │                          #   MotionModel (Trilha A), CropEncoder/AppearanceModel (Trilha B),
│   │   │                          #   loops de treino (teacher forcing → scheduled sampling → free-running,
│   │   │                          #   BPTT truncado, clipping), curva de ||dL_t/dh_{t-k}||
│   │   ├── loss.py                # L1 / smooth-L1 / NLL gaussiana / GIoU (caixas); contrastiva / triplet (embeddings)
│   │   └── optimizers.py          # create_optimizer, create_scheduler
│   └── plot/plot.py               # cores consistentes por id, tiras de quadros, trajetória com oclusão,
│                                  #   linha do tempo de identidades, curvas de quebra, descolamento
├── reports/
│   ├── 0_sintetic_tests.ipynb     # Parte 0 — testes sintéticos
│   ├── 1_baseline.ipynb           # Parte 1 — baseline por quadro no MOT17
│   ├── 2_temporal_memory.ipynb    # Parte 2 — Trilha A: GRU como modelo de movimento
│   └── results/                   # métricas (JSON/CSV) e checkpoints (.pt) de cada parte
└── tests/tests.py                 # unittest
```

## 5. Convenções

- **Caixas**: `(x1, y1, x2, y2)` em pixels. **Tabelas**: `np.ndarray (N, 7)` com colunas
  `frame, id, x1, y1, x2, y2, conf` — `conf` é a visibilidade no ground truth e a
  confiança em detecções/previsões; detecções sem identidade usam `id = -1`. É o `gt.txt`
  do MOT17 em xyxy (`src/nn/boxes.py::mot_to_table` / `table_to_mot` convertem).
- **Regra de associação** (Parte 1, documentada em `src/nn/tracking.py`): IoU entre a
  caixa prevista de cada track e as detecções do quadro; matching guloso por IoU
  decrescente (ou Hungarian), limiar fixo; detecção sem par → id novo; track sem par
  envelhece e morre após `max_age` quadros; só tracks observadas entram na saída.
- **Split do MOT17** (`src/dataset/mot17.py::SPLIT`), por sequência: treino = 02, 04, 05,
  11, 13; validação = 09 (câmera parada) e 10 (câmera móvel, noite). Toda escolha de
  hiperparâmetro é feita só no treino.
- **Avaliação no MOT17**: GT = pedestres com `conf = 1`; previsões casadas (Hungarian,
  IoU ≥ 0,5) com distratores — classes 2, 7, 8, 12 — são removidas antes de contar, como
  no avaliador oficial.
- **Métricas** (`src/nn/metrics.py`): IDF1 com atribuição global 1-para-1 (Hungarian sobre
  a matriz de quadros casados); ID switch quando a identidade verdadeira troca de parceiro
  em relação ao **último** casamento; fragmentação quando uma identidade casada fica sem
  par e volta. Os três casos à mão do enunciado estão em `tests/tests.py` e no notebook
  da Parte 0: (a) pred = GT ⇒ IDF1 = 1, 0 switches; (b) ids trocadas em k ⇒ 2 switches,
  IDF1 = 0,5; (c) track partida em k ⇒ 1 switch, IDF1 = 0,75.

## 6. Resultados por parte

### Parte 0 — testes sintéticos (`reports/0_sintetic_tests.ipynb`)

- **Gerador**: a duração medida da oclusão total bate com o parâmetro (5,7 / 10,0 / 20,3 /
  30,7 quadros para alvos de 5 / 10 / 20 / 30); o ocultado some do mapa de rótulos
  (visibilidade 0) e volta.
- **Detector simulado**: mAP@[.50:.95] 0,87 (só o portão de visibilidade) → 0,69 → 0,39 →
  0,14 nas intensidades leve / média / pesada.
- **Baseline no piso fácil** (5 elipses, 0,5 px/quadro, sem oclusão, detector perfeito):
  IDF1 = 0,989 ± 0,020 em 10 seeds, zero switches.
- **Onde quebra**: velocidade ≥ 4 px/quadro (IDF1 0,82 → 0,38 em 8 px/quadro, ~19
  switches por identidade) e qualquer oclusão mais longa que `max_age` (switches por
  identidade de 0,13 para 0,4–0,6). O Kalman de velocidade constante ajuda em oclusões
  curtas e perde em oclusões longas / velocidades altas por causa dos ricochetes nas
  bordas — velocidade constante prevê através da parede.

### Parte 2 — memória temporal, Trilha A (`reports/2_temporal_memory.ipynb`)

- **Modelo**: uma GRU (64 unidades, 14 mil parâmetros, células escritas do zero) por track,
  alimentada com a velocidade observada relativa ao tamanho da caixa, a escala, a flag de
  observação e Δt; sob oclusão ela recebe a própria previsão e roda para frente. Prevê o
  deslocamento da caixa do quadro seguinte.
- **Treino**: smooth-L1 no deslocamento, GT como alvo e as **detecções reais do SDP** como
  entrada (o erro do detector é correlacionado no tempo, autocorrelação ~0,45 — ruído
  independente simulado ensinaria a suavizar um tremor que não existe), blocos de ausência
  de até 40 quadros, scheduled sampling 0 → 0,5.
- **Previsão do quadro seguinte** (validação): a GRU é a melhor até ~15 quadros sem
  observação (IoU 0,81 contra 0,79 do Kalman e da última caixa a partir de uma detecção;
  0,36 contra 0,35 e 0,28 após 8–15 quadros).
- **Rastreamento** (IDF1 combinado nas 7 sequências): última caixa 0,594 → Kalman 0,641 /
  GRU 0,632; switches 1.438 → 1.048 / 1.180. A GRU **empata com o Kalman no treino e perde na
  validação** (0,519 vs 0,557): prever um pouco melhor raramente muda um casamento com limiar
  de IoU 0,3. Onde a memória ajuda é onde a Parte 1 apontou: nas câmeras móveis a identidade
  sobrevive a buracos de 3–4 quadros em 74 % dos casos (58 % no baseline), nas paradas a
  buracos de 20–39 quadros em 51 % (16 %), e as mortes por oclusão longa caem à metade.
- **Incerteza + portão adaptativo** (NLL gaussiana; a elipse de busca cresce 20× após 1–2 s
  sem observação): o melhor na validação (0,561) e o que menos inventa identidades, mas pior
  no treino — pelo protocolo, o modelo final é a GRU com smooth-L1.
