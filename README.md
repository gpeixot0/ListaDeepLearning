# Lista de Exercícios — Aprendizado Profundo (UFPB/CI, 2026.2)

**Professor:** Tiago Maritan · **Entrega:** 17/11/2026

Resolução completa em PyTorch, com código, resultados e explicações conceituais em cada notebook.

## Estrutura

```
├── notebooks/
│   ├── Q1_MLP_Regressao.ipynb              # (a) aproximação de função  (b) Diabetes
│   ├── Q2_MLP_Classificacao_Wine.ipynb     # Wine Quality, 3 classes, balanceamento
│   ├── Q3_Q4_CIFAR10_MLP_vs_CNN.ipynb      # Q3 MLP × CNN  +  Q4 (a) filtros (b) ativações/Grad-CAM (c) acertos/erros (d) evolução no treino
│   └── Q5_Autoencoders_FashionMNIST.ipynb  # (a) AE latente 100/50/30/10  (b) denoising AE
├── html/                                   # notebooks executados exportados em HTML (abrem no navegador)
├── figuras/                                # todas as figuras em PNG (para os slides)
├── modelos/
│   ├── cnn_cifar10.pt                      # pesos da CNN treinada na Q3 (usada na Q4)
│   ├── hist_cnn.json                       # histórico de treino dessa CNN
│   └── cnn_checkpoints.pt                  # checkpoints (épocas 0, 1, 2, 5, 10, 25) usados na Q4(d)
└── requirements.txt
```

A pasta `data/` é criada automaticamente na primeira execução (os datasets são baixados das fontes oficiais).

## Como executar

**Google Colab (recomendado, com GPU):** abra o notebook, `Ambiente de execução → Alterar tipo → GPU`, e rode tudo. Os dados são baixados automaticamente das fontes oficiais (torchvision/CIFAR-10, UCI, repositório da Zalando) na pasta `data/` — os notebooks esperam ser executados de dentro de `notebooks/`.

**Local:**
```bash
pip install -r requirements.txt
cd notebooks && jupyter lab
```

Tempos aproximados em CPU (2 núcleos): Q1 ≈ 1 min · Q2 ≈ 3 min · Q5 ≈ 15 min · Q3/Q4 ≈ 45 min, mais ≈ 40 min se a Q4(d) precisar retreinar (se `modelos/cnn_checkpoints.pt` existir, os checkpoints são carregados). Em GPU, todos rodam em poucos minutos.

## Decisões gerais
- **Reprodutibilidade:** semente 42 em tudo (Python, NumPy, PyTorch).
- **Sem vazamento de dados:** normalizações e reamostragens ajustadas **só no treino**; o teste é usado uma única vez, no final (na Q4(d), a acurácia de teste de cada checkpoint é só descritiva: nenhuma escolha é feita com ela).
- **Validação separada** para early stopping/seleção de modelo em todas as questões (na Q1a o enunciado pede monitorar com o conjunto de teste, e assim foi feito).
- **Baselines** sempre reportados (regressão linear, classe majoritária, regressão logística) para dar contexto às métricas da rede.

## Fontes dos dados
- Diabetes: `sklearn.datasets.load_diabetes`
- Wine Quality: UCI ML Repository, id 186 (tinto + branco, 6.497 amostras)
- CIFAR-10: Krizhevsky, 2009 — cs.toronto.edu/~kriz/cifar.html
- Fashion-MNIST: Xiao, Rasul & Vollgraf, 2017 — github.com/zalandoresearch/fashion-mnist
