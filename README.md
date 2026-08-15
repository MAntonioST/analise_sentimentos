# 📊 Análise de Sentimentos - B2W

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.0-orange?logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-3.8.1-brightgreen?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-Academic-green)
![Status](https://img.shields.io/badge/Status-Em%20Desenvolvimento-yellow)

> **Pós-Graduação em Inteligência Artificial para Desenvolvedores**  
> **FIAP** — Disciplina: Processamento de Linguagem Natural (PLN)  
> **Autor:** Marco Antonio Teixeira

## Sobre o projeto

Este projeto implementa um pipeline completo de **análise de sentimentos** utilizando avaliações de produtos do dataset B2W.

O objetivo é classificar cada avaliação como:

- **Negativa (`0`)**
- **Positiva (`1`)**

O projeto foi desenvolvido com foco em:

- Processamento de Linguagem Natural;
- Organização modular de código;
- Princípios SOLID;
- Reprodutibilidade dos experimentos;
- Avaliação de modelos de machine learning;
- Geração automática de visualizações;
- Persistência do modelo e do vetorizador treinados.

## Resultado atual

A configuração atual do pipeline utiliza:

- Tokenização com NLTK;
- Remoção de stopwords em português;
- Stemming;
- Vetorização TF-IDF;
- Unigramas e bigramas;
- Vocabulário limitado a 1.000 features;
- Regressão Logística;
- Divisão estratificada entre treino e teste;
- Vetorização sem vazamento de dados.

### Métricas

| Métrica | Resultado |
|---|---:|
| **Acurácia** | **93,18%** |
| F1-score — classe negativa | **0,89** |
| F1-score — classe positiva | **0,95** |
| F1-score macro médio | **0,92** |
| F1-score ponderado | **0,93** |

### Relatório de classificação

| Classe | Precisão | Recall | F1-score |
|---|---:|---:|---:|
| Negativa (`0`) | 0,89 | 0,89 | 0,89 |
| Positiva (`1`) | 0,95 | 0,95 | 0,95 |

A avaliação foi realizada com **23.212 amostras de teste**.

## Pipeline de processamento

O fluxo principal é executado pelo comando:

```bash
python main.py
```

### Etapas do pipeline

```text
Carregamento do dataset
        ↓
Remoção de colunas desnecessárias
        ↓
Remoção de registros nulos
        ↓
Tokenização com NLTK
        ↓
Remoção de stopwords em português
        ↓
Stemming
        ↓
Divisão estratificada entre treino e teste
        ↓
TF-IDF ajustado somente no conjunto de treino
        ↓
Transformação do conjunto de teste
        ↓
Treinamento da Regressão Logística
        ↓
Avaliação do modelo
        ↓
Geração de gráficos
        ↓
Salvamento do modelo e do vetorizador
```

## Prevenção de vazamento de dados

O pipeline realiza a divisão entre treino e teste **antes** da vetorização.

O vetorizador é ajustado apenas com os dados de treino:

```python
X_train = vectorizer.fit_transform(textos_train)
X_test = vectorizer.transform(textos_test)
```

Essa abordagem evita que o vocabulário e os pesos IDF sejam aprendidos a partir dos dados de teste.

O conjunto de teste é utilizado somente para a avaliação final do modelo.

## Tecnologias utilizadas

- **Python 3.10+**
- **pandas**
- **NumPy**
- **scikit-learn 1.4.0**
- **NLTK 3.8.1**
- **SciPy**
- **PyYAML**
- **Matplotlib**
- **Seaborn**
- **WordCloud**
- **Joblib**

## Estrutura do projeto

```text
analise_sentimentos/
│
├── config/
│   └── config.yaml
│
├── data/
│   └── raw/
│       └── b2w.csv
│
├── outputs/
│   ├── figures/
│   │   ├── distribuicao_classes.png
│   │   ├── matriz_confusao.png
│   │   ├── wordcloud_todas_as_avaliações.png
│   │   ├── wordcloud_negativas.png
│   │   └── wordcloud_positivas.png
│   │
│   └── models/
│       ├── sentiment_model.joblib
│       └── vectorizer.joblib
│
├── scripts/
│   └── scripts auxiliares e experimentais
│
├── src/
│   ├── data/
│   │   └── carregamento e acesso aos dados
│   │
│   ├── evaluation/
│   │   └── sentiment_experiments.py
│   │
│   ├── features/
│   │   ├── nltk_processor.py
│   │   ├── stemming.py
│   │   ├── tfidf_vectorizer.py
│   │   └── vectorizer.py
│   │
│   ├── models/
│   │   └── classificadores de sentimentos
│   │
│   ├── pipeline/
│   │   └── orchestrator.py
│   │
│   ├── preprocessing/
│   │   └── preprocessor.py
│   │
│   └── visualization/
│       └── geração de gráficos e nuvens de palavras
│
├── tests/
│   └── testes automatizados
│
├── analisar_imports.py
├── analise_sistema.sh
├── main.py
├── requirements.txt
└── README.md
```

## Principais componentes

### `main.py`

Ponto único de entrada da aplicação.

Responsável por:

- Instanciar os componentes;
- Carregar as configurações;
- Montar as dependências;
- Executar o pipeline completo.

### `src/pipeline/orchestrator.py`

Orquestra todas as etapas do projeto:

- Carregamento;
- Pré-processamento;
- Processamento linguístico;
- Stemming;
- Divisão dos dados;
- Vetorização;
- Treinamento;
- Avaliação;
- Visualização;
- Persistência dos artefatos.

### `src/features/nltk_processor.py`

Realiza:

- Tokenização;
- Conversão para letras minúsculas;
- Remoção de stopwords em português;
- Filtragem de tokens não alfabéticos;
- Aplicação do stemming.

### `src/features/stemming.py`

Contém a lógica específica de stemming dos textos.

### `src/features/vectorizer.py`

Transforma os textos em matrizes numéricas utilizando TF-IDF.

Também oferece:

- `fit_transform()` para os dados de treino;
- `transform()` para os dados de teste;
- Extração dos nomes das features;
- Conversão da matriz esparsa para DataFrame.

### `src/models/`

Contém os modelos utilizados na classificação das avaliações.

O modelo atualmente utilizado é a **Regressão Logística**.

### `src/evaluation/`

Contém experimentos e rotinas auxiliares para avaliação de diferentes configurações de vetorização e classificação.

### `src/visualization/`

Gera:

- Distribuição das classes;
- Nuvem de palavras geral;
- Nuvem de palavras negativas;
- Nuvem de palavras positivas;
- Matriz de confusão.

## Configuração

As principais configurações ficam em:

```text
config/config.yaml
```

Exemplo da configuração de vetorização:

```yaml
vectorization:
  max_features: 1000
  ngram_min: 1
  ngram_max: 2
```

Essa configuração representa:

- Até **1.000 features**;
- Unigramas (`1`);
- Bigramas (`2`).

Exemplos de features:

```text
produto
qualidade
muito bom
não gostei
produto defeituoso
```

## Instalação

### 1. Clonar o repositório

```bash
git clone https://github.com/MAntonioST/analise_sentimentos.git
cd analise_sentimentos
```

### 2. Criar o ambiente virtual

Linux ou macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

### 4. Executar o pipeline

```bash
python main.py
```

Na primeira execução, o NLTK poderá baixar os recursos necessários:

- `punkt`;
- `punkt_tab`;
- `stopwords`.

## Saídas geradas

Após a execução, os gráficos são salvos em:

```text
outputs/figures/
```

O modelo e o vetorizador são salvos em:

```text
outputs/models/
```

Arquivos principais:

```text
outputs/models/sentiment_model.joblib
outputs/models/vectorizer.joblib
```

Esses artefatos podem ser reutilizados posteriormente para realizar previsões em novas avaliações sem treinar o modelo novamente.

## Comandos úteis

### Executar o pipeline

```bash
python main.py
```

### Validar a sintaxe dos módulos

```bash
python -m py_compile src/features/vectorizer.py
python -m py_compile src/pipeline/orchestrator.py
```

### Verificar o estado do Git

```bash
git status --short
```

### Verificar a ordem correta da vetorização

```bash
grep -n "fit_transform\|transform(textos_test)\|train_test_split" \
    src/pipeline/orchestrator.py
```

### Verificar a etapa de stemming

```bash
grep -n "aplicar_stemming\|coluna_texto_stemmed" \
    src/features/nltk_processor.py \
    src/pipeline/orchestrator.py
```

## Princípios de projeto

### SRP — Single Responsibility Principle

Cada componente possui uma responsabilidade específica:

- Carregamento de dados;
- Pré-processamento;
- Processamento linguístico;
- Vetorização;
- Classificação;
- Avaliação;
- Visualização.

### OCP — Open/Closed Principle

O pipeline pode ser estendido com:

- Novos modelos;
- Novos vetorizadores;
- Novas técnicas de pré-processamento;
- Novas visualizações;

sem modificar necessariamente o fluxo principal.

### DIP — Dependency Inversion Principle

O orquestrador recebe os componentes como dependências, permitindo substituir implementações sem alterar a lógica central do pipeline.

## Histórico de experimentos

| Configuração | Acurácia |
|---|---:|
| CountVectorizer + stemming | 88,35% |
| TF-IDF + stemming | 88,48% |
| TF-IDF + stemming + n-grams com 100 features | 88,01% |
| TF-IDF + stemming + n-grams com 1.000 features | 93,15% |
| TF-IDF + stemming + n-grams com split antes da vetorização | **93,18%** |

A última configuração é considerada a referência atual por utilizar uma avaliação sem vazamento de dados.

## Status do projeto

Atualmente, o projeto possui:

- ✅ Dataset B2W integrado;
- ✅ Pré-processamento dos dados;
- ✅ Tokenização com NLTK;
- ✅ Stopwords em português;
- ✅ Stemming;
- ✅ TF-IDF;
- ✅ Suporte a unigramas e bigramas;
- ✅ Regressão Logística;
- ✅ Divisão estratificada treino/teste;
- ✅ Prevenção de vazamento de dados;
- ✅ Avaliação com métricas de classificação;
- ✅ Matriz de confusão;
- ✅ Nuvens de palavras;
- ✅ Persistência do modelo;
- ✅ Persistência do vetorizador;
- ✅ Execução centralizada pelo `main.py`.

## Licença

Este projeto é distribuído como software livre sob os termos da **Licença MIT**.

A Licença MIT permite:

- Uso comercial e acadêmico;
- Cópia e distribuição;
- Modificação do código;
- Criação de trabalhos derivados;
- Uso privado do software.

A única exigência principal é a manutenção do aviso de copyright e do texto da licença nas cópias ou partes substanciais do software.

### Copyright

Copyright (c) 2026 Marco Antonio Teixeira

### Texto da licença

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Autoria e contexto acadêmico

**Autor:** Marco Antonio Teixeira  
**Curso:** Pós-Graduação em IA para Desenvolvedores — FIAP  
**Local:** Mogi das Cruzes, SP — Brasil  

Este projeto foi desenvolvido para fins acadêmicos na disciplina de
Processamento de Linguagem Natural (PLN), como parte da Pós-Graduação em
Inteligência Artificial para Desenvolvedores da FIAP.

## Referências

### Processamento de Linguagem Natural

- [NLTK — Natural Language Toolkit](https://www.nltk.org/)
- [NLTK Book — Natural Language Processing with Python](https://www.nltk.org/book/)
- [NLTK Stopwords](https://www.nltk.org/howto/corpus.html)

### Vetorização TF-IDF e n-grams

- [scikit-learn — TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [scikit-learn — Feature extraction from text](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
- [scikit-learn — Working with text data](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)

### Modelagem e avaliação

- [scikit-learn — LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [scikit-learn — train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
- [scikit-learn — Classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)

### Bibliotecas utilizadas

- [pandas](https://pandas.pydata.org/docs/)
- [NumPy](https://numpy.org/doc/)
- [SciPy](https://docs.scipy.org/doc/scipy/)
- [PyYAML](https://pyyaml.org/wiki/PyYAMLDocumentation)
- [Matplotlib](https://matplotlib.org/stable/)
- [Seaborn](https://seaborn.pydata.org/)
- [WordCloud](https://amueller.github.io/word_cloud/)

### Dataset

O projeto utiliza o dataset de avaliações de produtos B2W disponível no
diretório:

```text
data/raw/b2w.csv
```

A referência original do dataset deve ser mantida conforme a fonte de onde o
arquivo foi obtido. Caso o dataset tenha sido baixado de uma plataforma,
repositório ou material didático específico, recomenda-se incluir aqui o link
original e os respectivos créditos.
