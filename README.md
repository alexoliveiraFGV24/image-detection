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

> **Estado atual:** Parte 0 (testes sintéticos) concluída; `src/` contém os modelos,
> perdas, otimizadores, métricas e o rastreador usados pelas partes seguintes.

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

1. Baixe o pacote **só de anotações** do MOT17 (~10 MB, sem conta):
   <https://motchallenge.net/data/MOT17/> → `MOT17Labels.zip`. Ele basta para a métrica,
   a associação e a Trilha A (movimento). O pacote completo (~5,5 GB, com os quadros) só é
   necessário para o detector do torchvision e para a Trilha B (aparência).
2. Descompacte em `data/` (a pasta está no `.gitignore`). A estrutura esperada é a original:

```
data/MOT17Labels/train/MOT17-02-FRCNN/gt/gt.txt      # frame, id, left, top, w, h, conf, class, visibility
data/MOT17Labels/train/MOT17-02-FRCNN/det/det.txt    # detecções públicas
data/MOT17Labels/train/MOT17-02-FRCNN/seqinfo.ini
data/MOT17/train/MOT17-02-FRCNN/img1/000001.jpg      # (pacote completo) quadros
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

**Testes unitários** (caixas, NMS, matching, os três casos da métrica, gerador,
rastreador, células e perdas):

```bash
python -m unittest tests.tests -v
```

**Métrica em arquivos do MOTChallenge** (o entregável `metrics.py`):

```bash
python metrics.py --gt data/MOT17Labels/train/MOT17-02-FRCNN/gt/gt.txt --pred resultados/MOT17-02-FRCNN.txt
```

Os comandos de treino e avaliação do modelo temporal serão adicionados quando as
Partes 1–2 forem concluídas.

## 4. Estrutura do repositório

```
image-detection/
├── README.md
├── AI_LOG.md                      # como a IA foi usada (episódios)
├── requirements.txt
├── metrics.py                     # entregável: IDF1, ID switches, fragmentações (+ CLI para arquivos MOT)
├── inferencia.ipynb               # entregável: inferência numa sequência qualquer (a preencher)
├── checkpoint.json                # link/descrição do checkpoint do modelo temporal (a preencher)
├── assignment/
│   ├── PA2.pdf                    # enunciado
│   ├── 06) Detecção de objetos.pdf
│   └── 07) RNN.pdf                # slides das aulas
├── data/                          # NÃO versionado — MOT17Labels (e MOT17 completo, opcional)
├── src/
│   ├── dataset/
│   │   ├── sintetic.py            # Parte 0: generate_video (elipses com ordem de profundidade e
│   │   │                          #   oclusão de duração controlada), simulate_detector, SyntheticVideoDataset
│   │   └── trajectories.py        # TrajectoryDataset: janelas de trajetórias do GT para o MotionModel
│   ├── nn/
│   │   ├── boxes.py               # IoU vetorizado, NMS próprio, conversões, tabelas (frame, id, x1, y1, x2, y2, conf)
│   │   ├── metrics.py             # AP/mAP por quadro; IDF1, ID switches, fragmentações, contagem de ids, MOTA
│   │   ├── tracking.py            # Tracker (associação + nascimento/morte) e modelos de movimento:
│   │   │                          #   StaticMotion (Parte 1), KalmanMotion (baseline), RNNMotion (Parte 2)
│   │   ├── models.py              # RNNCell/LSTMCell/GRUCell do zero, Recurrent (bidirecional opcional),
│   │   │                          #   MotionModel (Trilha A), CropEncoder/AppearanceModel (Trilha B),
│   │   │                          #   loops de treino (teacher forcing → scheduled sampling → free-running,
│   │   │                          #   BPTT truncado, clipping), curva de ||dL_t/dh_{t-k}||
│   │   ├── loss.py                # L1 / smooth-L1 / NLL gaussiana / GIoU (caixas); contrastiva / triplet (embeddings)
│   │   └── optimizers.py          # create_optimizer, create_scheduler
│   └── plot/plot.py               # cores consistentes por id, tiras de quadros, trajetória com oclusão,
│                                  #   linha do tempo de identidades, curvas de quebra do baseline
├── reports/
│   ├── 0_sintetic_tests.ipynb     # Parte 0 — testes sintéticos
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
