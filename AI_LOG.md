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

Decisões que continuam nossas: escolha da trilha da Parte 2, do eixo da ablação, do
detector público padrão e do split por sequência do MOT17.
