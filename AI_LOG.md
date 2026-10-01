# AI_LOG — como usamos IA neste trabalho

Ferramenta: **Claude Code** (Anthropic, modelo Opus 5), usado como assistente de
programação dentro do repositório — com acesso ao enunciado (`assignment/PA2.pdf`), aos
slides das aulas de detecção e de RNN, às anotações do MOT17 em `data/` e ao nosso
repositório do PA1 (`semantic-segmentation`), cuja estrutura e estilo pedimos que fossem
seguidos.

## Sessão 1 — infraestrutura e Parte 0

O que pedimos: criar do zero os modelos temporais, as métricas, as perdas e os
otimizadores em `src/`, e resolver a Parte 0 em `reports/0_sintetic_tests.ipynb`,
tomando o PA1 como referência de organização.

O que a IA fez, e o que revisamos:

1. **`src/nn/boxes.py`, `metrics.py`, `tracking.py`**: IoU vetorizado, NMS próprio,
   matching guloso e Hungarian, IDF1 (atribuição global via `linear_sum_assignment` do
   scipy, permitido), ID switches e fragmentações no estilo CLEAR MOT, o `Tracker` com a
   regra de associação documentada e modelos de movimento plugáveis (estático, Kalman de
   velocidade constante como baseline, adaptador para a RNN).
2. **`src/nn/models.py`, `loss.py`, `optimizers.py`**: células RNN/LSTM/GRU escritas com
   as equações dos slides (sem `nn.LSTM`/`nn.GRU`), `MotionModel` (Trilha A) com
   teacher forcing → scheduled sampling → free-running, BPTT truncado e incerteza
   gaussiana opcional; `AppearanceModel` (Trilha B) com encoder de recortes e agregador
   recorrente; perdas L1 / smooth-L1 / NLL gaussiana / GIoU e contrastiva / triplet.
3. **`src/dataset/sintetic.py`** e o notebook da Parte 0, executado de ponta a ponta
   (`nbconvert --execute`). As conclusões do notebook foram escritas **depois** de olhar
   os números.

Três episódios em que o diálogo com a IA mudou o projeto:

### Episódio 1 — "duração da oclusão" não é o mesmo que "velocidade relativa"

A primeira versão do gerador fazia o ocultador ultrapassar o ocultado com velocidade
relativa constante, calibrada para que a cobertura total durasse `occlusion_duration`
quadros. Ao imprimir a visibilidade quadro a quadro, a IA notou que a fase de oclusão
**parcial** durava ~5× a total (a visibilidade caía de 0,65 a 0 ao longo de 14 quadros
para uma oclusão total de 10): com movimento relativo linear, a razão entre as duas fases
é fixada pelos raios e não pelo parâmetro. O botão "duração da oclusão" não controlava o
que dizia controlar. A solução foi trocar o movimento relativo por "aproxima rápido →
anda junto por $D$ quadros → afasta rápido" (`approach_speed`), compensando a folga da
margem, e **verificar** a duração no mapa de rótulos (`occlusion_intervals`). A tabela
"os botões fazem o que dizem" do notebook é essa verificação (alvo 5/10/20/30 → medido
5,7/10,0/20,3/30,7).

### Episódio 2 — a detecção "ideal" é a caixa inteira, não a visível

A IA propôs inicialmente que o detector simulado devolvesse a caixa **só da parte
visível** de um objeto parcialmente ocluído (mais realista). No primeiro teste do baseline
com detecções "perfeitas" apareceram 11 FP + 11 FN e IDF1 = 0,90 — sem nenhum ruído. O
motivo: o ground truth (como o do MOT17) é a caixa inteira, e a caixa visível desalinha
a detecção do GT, penalizando o IoU mesmo com rastreamento perfeito. Decidimos que a
oclusão deve virar **só falta de detecção** (portão `min_visibility`), com a caixa
visível como opção explícita e documentada.

### Episódio 3 — o Kalman perde do "última caixa" e isso não é bug

Na varredura da Parte 0 o filtro de Kalman de velocidade constante ficou **pior** que a
associação ingênua em 4 px/quadro (IDF1 0,74 contra 0,82) e não ajudava em oclusões
longas. Suspeitamos de bug. A IA isolou o filtro numa trajetória em linha reta com 20
quadros sem observação: a caixa prevista ainda tinha IoU 0,6 com a verdadeira, e uma
grade de sintonias (ruído de processo / de medida) mostrou que filtros *mais suaves*
pioravam. A causa são os **ricochetes nas bordas** do mundo sintético: velocidade
constante prevê através da parede, e quanto mais rápido o objeto ou mais longa a oclusão,
maior a chance de um ricochete no meio. Mantivemos o comportamento e o explicamos no
notebook — é o gancho para a Parte 2: um modelo de movimento aprendido nessas
trajetórias pode capturar essa regularidade, o filtro não. O teste
`test_static_loses_moving_object_kalman_keeps_it` em `tests/tests.py` documenta o caso
em que o Kalman *deve* ganhar.

Decisões que continuam nossas: escolha da trilha da Parte 2 e do eixo da ablação.

## Sessão 2 — Parte 1 (baseline por quadro no MOT17)

O que pedimos: resolver a Parte 1 em `reports/1_baseline.ipynb`. A IA baixou os quadros
do MOT17 (`MOT17Det.zip`, da fonte oficial), escreveu `src/dataset/mot17.py`,
`src/nn/detector.py`, as análises novas de `src/nn/metrics.py` e `src/plot/plot.py`, e
o notebook, executado de ponta a ponta. O **split** treino/validação e a **fonte
padrão** de detecções foram propostos pela IA com justificativa no notebook (seções 1 e
3) e revisados por nós — são escolhas que teremos de defender na apresentação.

Episódios:

### Episódio 4 — o filtro escondido que zerava o DPM negativo

Na varredura do limiar de score dos detectores públicos, o DPM deu números **idênticos**
com limiar −0,5 e 0,0. A IA desconfiou do empate exato e achou a causa: o `Tracker` tinha
`min_score = 0.0` como padrão, um filtro silencioso que descartava todo score negativo —
inofensivo no sintético (scores ≥ 0,05) e errado no DPM do MOT17, cujos scores começam
em −0,5. O padrão virou `None` (sem filtro) e a varredura foi refeita.

### Episódio 5 — o NMS do torchvision e o custo do detector em CPU

O enunciado proíbe `torchvision.ops.nms`, mas o Faster R-CNN do torchvision chama NMS por
dentro. A IA separou os dois usos: o da RPN (parte da arquitetura do detector, que é
permitido) ficou; o **final**, que transforma caixas em detecções, foi desligado
(`nms_thresh = 1.0`) e substituído pelo nosso, aplicado sobre as caixas cruas em cache.
Rodar o ResNet50-FPN v2 nas 5.316 imagens em CPU levaria ~5 h. Em vez de trocar por um
modelo menor (a MobileNet achava 16 pessoas onde o ResNet achava 26), a IA mediu onde
estava o custo — a cabeça do v2 aplica 4 convoluções a cada uma das 1.000 propostas — e
usou 300 propostas no teste, como no artigo original do Faster R-CNN: as mesmas 45
detecções no quadro mais denso do MOT17-04, ~30 % mais rápido. Também baixou o
`MOT17Det.zip` (1,9 GB) em vez do `MOT17.zip` (5,9 GB): as mesmas imagens, sem a
triplicação por detector.
