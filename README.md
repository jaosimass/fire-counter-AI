# Fire Counter AI

Projeto acadêmico da disciplina IAA3 sobre monitoramento florestal. Os experimentos analisam imagens RGB rotuladas como `fire` ou `nofire` e registros de ocorrências de incêndio do Ibama.

## Experimentos

| Abordagem | Entrada | O que o código faz |
| --- | --- | --- |
| [CNN](src/cnn_treino.py) | Pixels de imagens RGB | Classifica a imagem inteira como `fire` ou `nofire`. Inclui treinamento, avaliação e previsão com um modelo já treinado. |
| [RNA e derivadas](src/rna_treino.py) | Seis atributos extraídos das imagens | Classifica as mesmas duas classes com uma rede densa. Um experimento separado compara derivadas numéricas com derivadas analíticas e com o gradiente do TensorFlow. |
| [K-Means](src/main.py) | Estatísticas RGB/HSV e proporção de pixels com cor semelhante à de fogo | Agrupa imagens com características parecidas. Depois do ajuste, interpreta os grupos pela classe majoritária no treino. |
| [Método simbólico](symbolic/regras.py) | Nível de acionamento e tipo de área em registros do Ibama | Aplica regras para atribuir prioridade `ALTA`, `MÉDIA` ou `BAIXA` a cada ocorrência. |
| [Árvore de decisão](decision_tree/arvore.py) | Área atingida e atributos tabulares da ocorrência | Estima o nível de acionamento (`Nível 1`, `Nível 2` ou `Nível 3`). Seleciona hiperparâmetros por F1 macro em validação cruzada e avalia em um teste separado. |
| [Fuzzy e algoritmo genético](src/fuzzy_ga.py) | Médias RGB e luminosidade | Calcula uma pontuação de risco com regras fuzzy. Na etapa genética experimental, o cálculo de aptidão ignora os genes candidatos e mantém fixos os parâmetros fuzzy. |

## Instalação

Os exemplos usam Python 3.12 e partem da raiz do repositório. No Windows, estes comandos funcionam no CMD e no PowerShell, sem ativar o ambiente virtual.

```powershell
git clone https://github.com/jaosimass/fire-counter-AI.git
cd fire-counter-AI
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Instale as dependências extras conforme o experimento. O `requirements.txt` contém NumPy, pandas, Matplotlib, scikit-learn e Pillow.

```powershell
# CNN e RNA
.\.venv\Scripts\python.exe -m pip install tensorflow==2.21.0

# Fuzzy e algoritmo genético
.\.venv\Scripts\python.exe -m pip install scikit-fuzzy pygad networkx
```

O TensorFlow 2.21.0 é a versão registrada na [execução da CNN](results/cnn/resumo_execucao.json). Em Linux ou macOS, crie o ambiente com `python3.12 -m venv .venv` e use `.venv/bin/python` no lugar de `.\.venv\Scripts\python.exe`.

## Dados e divisão

- [Imagens G20GLI](data/raw/g20gli_dataset/) — o clone inclui 1.900 imagens RGB, com 950 em `fire/` e 950 em `nofire/`. Nessa tarefa, `fire = 1` e `nofire = 0`.
- [Divisão da CNN e da RNA](results/cnn/divisao_dados.csv) — 1.330 imagens de treino, 286 de validação e 284 de teste. A CNN usa semente 42 e mantém imagens quase duplicadas, agrupadas por hash perceptual (dHash), no mesmo conjunto. A RNA reutiliza essa divisão.
- [Atributos das imagens](data/processed/atributos_g20gli.csv) — tabela usada pela RNA e pelo protótipo fuzzy. Para gerá-la novamente, execute `.\.venv\Scripts\python.exe src/data_processing.py`.
- **Registros do Ibama** — o método simbólico e a preparação da árvore esperam o arquivo `data/raw/ibama/ROI.csv` com separador `;`. O clone inclui as [tabelas processadas](data/processed/), e o CSV original precisa ser fornecido para repetir a preparação.

## Como executar

### CNN

Preveja uma imagem usando o [modelo versionado](models/cnn_fire_nofire.keras). O comando aceita também imagens próprias ou o caminho de uma pasta.

```powershell
.\.venv\Scripts\python.exe src/cnn_prever.py data/raw/g20gli_dataset/fire/fire_0001.jpg
```

A saída mostra a classe prevista, a confiança do modelo e `P(fire)`. Com o modelo versionado, o limiar padrão é `0,5`.

```powershell
# Treino rápido com uma amostra, duas épocas e saídas em pasta temporária
.\.venv\Scripts\python.exe src/cnn_treino.py --smoke

# Treino completo, com saídas separadas dos artefatos versionados
.\.venv\Scripts\python.exe src/cnn_treino.py --modelo-saida models/cnn_nova.keras --resultados-dir results/cnn_nova

# Regenerar análises a partir das previsões versionadas
.\.venv\Scripts\python.exe src/cnn_analise.py --saida-dir results/cnn_analises
```

Os [resultados da CNN](results/cnn/) incluem o relatório, a matriz de confusão, as curvas de treino, exemplos de erros e a análise por luminosidade. Consulte também a [arquitetura e as limitações](docs/CNN_arquitetura_e_limitacoes.md), o [plano experimental](docs/CNN_plano_experimental.md) e o [notebook da CNN](notebooks/cnn_experimentos.ipynb).

### RNA, limites e derivadas

A rede usa as médias dos canais RGB, os desvios-padrão do vermelho e do verde e a luminosidade. O `StandardScaler` é ajustado apenas no treino. A arquitetura tem duas camadas densas com ReLU (16 e 8 neurônios) e uma saída sigmoid.

```powershell
.\.venv\Scripts\python.exe src/rna_dados.py
.\.venv\Scripts\python.exe src/rna_metricas.py
.\.venv\Scripts\python.exe src/rna_derivadas.py
.\.venv\Scripts\python.exe src/rna_graficos.py
```

Os comandos usam o [modelo](models/rna_fire_nofire.keras), o [histórico](results/rna/historico_treinamento.csv) e os CSVs versionados. O experimento de cálculo aproxima a derivada da sigmoid com valores decrescentes de `h` e compara a derivada da perda em relação a um peso real com o resultado de `tf.GradientTape`. Consulte os [resultados da RNA](results/rna/) e as [notas de limites e derivadas](docs/RNA_limites_derivadas.md).

Para treinar novamente, execute `.\.venv\Scripts\python.exe src/rna_treino.py`. Esse comando sobrescreve o modelo, o escalonador e o histórico nos caminhos padrão.

### K-Means

O script lê `data/train/{fire,nofire}` e `data/test/{fire,nofire}`. Essas pastas precisam ser preparadas. Para usar as imagens do repositório, copie os arquivos das linhas `train` e `test` do [manifesto](results/cnn/divisao_dados.csv) para as pastas correspondentes, preservando a classe. O bloco abaixo prepara a divisão e roda o K-Means no PowerShell, a partir de um clone recém-criado.

```powershell
Import-Csv results/cnn/divisao_dados.csv | Where-Object {
    $_.conjunto -in 'train', 'test'
} | ForEach-Object {
    $origem = Join-Path 'data/raw/g20gli_dataset' $_.arquivo
    $destino = Join-Path "data/$($_.conjunto)" $_.arquivo
    New-Item -ItemType Directory -Force -Path (Split-Path $destino) | Out-Null
    Copy-Item -LiteralPath $origem -Destination $destino
}
.\.venv\Scripts\python.exe src/main.py
```

O ajuste usa apenas os atributos do treino, sem os rótulos. O script gera CSVs de agrupamento, métricas de silhouette, ARI e NMI, matriz de confusão após interpretar os grupos e gráficos em [`results/`](results/). O código usa dois grupos por padrão. Já estão disponíveis o [gráfico de dispersão](results/clusters_rgb.png) e a [matriz de confusão](results/confusion_matrix.png) de uma execução anterior.

### Registros do Ibama

A árvore pode ser executada com o [CSV processado](data/processed/dados_arvore.csv) incluído no repositório.

```powershell
.\.venv\Scripts\python.exe decision_tree/arvore.py
```

Com o `ROI.csv` original na pasta esperada, o comando abaixo gera a [tabela de classificações simbólicas](data/processed/resultados_simbolico.csv).

```powershell
.\.venv\Scripts\python.exe symbolic/regras.py
```

O script `decision_tree/preparar_dados.py` também lê o `ROI.csv` para gerar `dados_arvore.csv`. Na versão atual, falta uma vírgula entre `DENTRO ENTORNO` e `TIPO DETECCAO` na lista de colunas categóricas. Isso precisa ser corrigido para regenerar a tabela quando o CSV contém essas duas colunas separadas.

### Fuzzy e algoritmo genético

```powershell
.\.venv\Scripts\python.exe src/fuzzy_ga.py
```

O script demonstra o cálculo de risco com uma entrada de exemplo e imprime os acertos em uma amostra fixa de 50 imagens durante a etapa genética. O risco é uma pontuação de 0 a 100 calculada pelas regras fuzzy.

## Resultados registrados

CNN e RNA têm resultados versionados no mesmo conjunto de teste, com 284 imagens e `fire` como classe positiva.

| Modelo | Acurácia | Precisão | Recall | F1 | Falsos positivos | Falsos negativos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [CNN](results/cnn/metricas_cnn.json) | 98,59% | 98,59% | 98,59% | 98,59% | 2 | 2 |
| [RNA](results/rna/metricas_rna.csv) | 95,77% | 95,77% | 95,77% | 95,77% | 6 | 6 |

Nos registros do Ibama, a árvore acertou 867 de 1.207 níveis de acionamento no teste (71,83% de acurácia). As regras simbólicas classificaram 6.033 ocorrências por prioridade. As saídas estão em [previsões da árvore](data/processed/previsoes_arvore.csv) e [classificações simbólicas](data/processed/resultados_simbolico.csv).

## Estrutura e testes

| Pasta | Conteúdo |
| --- | --- |
| [`src/`](src/) | Extração de atributos, K-Means, CNN, RNA, análise de derivadas e fuzzy. |
| [`data/`](data/) | Imagens e tabelas processadas. |
| [`decision_tree/`](decision_tree/) e [`symbolic/`](symbolic/) | Árvore de decisão e regras para os registros do Ibama. |
| [`models/`](models/) | Modelos Keras da CNN e da RNA e escalonador da RNA. |
| [`results/`](results/) | Relatórios, gráficos e métricas versionadas. |
| [`docs/`](docs/) e [`notebooks/`](notebooks/) | Documentação e experimentos em notebook. |
| [`tests/`](tests/) | Testes da CNN para divisão de dados, pré-processamento, métricas, análises e modelo. |

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

A suíte usa `unittest`. Os testes que carregam ou treinam o modelo Keras são pulados se o TensorFlow estiver ausente.

## Limites do projeto

Os resultados são de experimentos acadêmicos. Os modelos de imagem classificam a cena inteira e foram avaliados no próprio dataset. Luz alaranjada, fumaça e mudanças de câmera ou região podem afetar o desempenho. As imagens RGB usadas aqui carecem das bandas necessárias para índices como NDVI e NBR. O projeto é destinado a estudo e demonstração. Uso em alertas reais exigiria validação em campo.
