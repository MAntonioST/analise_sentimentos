📊 Análise de Sentimentos - B2W
Python scikit-learn NLTK License Status

Pós-Graduação em Inteligência Artificial para Desenvolvedores
FIAP — Disciplina: Processamento de Linguagem Natural (PLN)
Autor: Marco Antonio Teixeira

📖 Sobre o Projeto
Este projeto implementa um classificador de sentimentos para avaliações de produtos da B2W (Americanas, Submarino, Shoptime), utilizando técnicas de Processamento de Linguagem Natural (PLN) e Machine Learning.

O sistema analisa textos de reviews e classifica automaticamente o sentimento como:

✅ Positivo (1) — Cliente satisfeito
❌ Negativo (0) — Cliente insatisfeito
🎯 Objetivos
Implementar um pipeline completo de NLP seguindo boas práticas de engenharia de software
Aplicar princípios SOLID na arquitetura do código
Utilizar NLTK para tokenização e remoção de stopwords em português
Utilizar Bag of Words (CountVectorizer) para vetorização de texto
Treinar um modelo de Regressão Logística para classificação binária
Gerar visualizações para análise exploratória dos dados
Criar um projeto profissional e reutilizável como portfólio


Arquitetura do Projeto

graph TB
    A[main.py] --> B[PipelineOrchestrator]
    B --> C[DataLoader]
    B --> D[DataPreprocessor]
    B --> N[NLTKProcessor]
    B --> E[TextVectorizer]
    B --> F[SentimentClassifier]
    B --> G[Visualizer]
    B --> H[MetricsCalculator]
    C --> I[(b2w.csv)]
    D --> J[Textos Limpos]
    N --> J2[Tokens Filtrados]
    E --> K[Matriz de Features]
    F --> L[Previsoes]
    G --> M[Graficos]
    H --> N2[Relatorio]

Fluxo do Pipeline

sequenceDiagram
    participant M as main.py
    participant O as Orchestrator
    participant L as DataLoader
    participant P as Preprocessor
    participant NL as NLTKProcessor
    participant V as Vectorizer
    participant MD as Model
    participant E as Evaluator
    participant VIS as Visualizer

    M->>O: executar()
    O->>L: carregar()
    L-->>O: DataFrame bruto
    O->>P: processar(df)
    P-->>O: DataFrame limpo
    O->>NL: processar(df)
    NL-->>O: DataFrame tokenizado + sem stopwords
    O->>V: fit_transform(textos)
    V-->>O: Matriz esparsa (BoW)
    O->>MD: treinar(X_train, y_train)
    MD-->>O: Modelo treinado
    O->>MD: prever(X_test)
    MD-->>O: Predições
    O->>E: acuracia(), matriz_confusao()
    E-->>O: Métricas
    O->>VIS: nuvem_palavras(), matriz_confusao()
    VIS-->>O: Gráficos salvos
    O-->>M: Resultados

Estrutura de Diretórios

analise_sentimentos/
│
├── README.md
├── .gitignore
├── requirements.txt
├── Makefile
├── main.py
│
├── config/
│   └── config.yaml
│
├── data/
│   └── raw/
│       └── b2w.csv
│
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   └── preprocessor.py
│   │
│   ├── features/
│   │   ├── nltk_processor.py      ← NOVO: Tokenização + Stopwords
│   │   └── vectorizer.py
│   │
│   ├── models/
│   │   ├── base_model.py
│   │   └── sentiment_classifier.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── visualizer.py
│   │
│   └── pipeline/
│       └── orchestrator.py
│
├── notebooks/
│   └── 01_analise_exploratoria.ipynb
│
├── outputs/
│   ├── figures/
│   └── models/
│
└── tests/
    ├── test_preprocessor.py
    └── test_models.py


🚀 Como Executar
Pré-requisitos
Python 3.10+
Git
pip (gerenciador de pacotes Python)
Conexão com internet (download de recursos NLTK na primeira execução)
Instalação

1. Clonar o repositório
git clone https://github.com/MAntonioST/analise-sentimentos-b2w.git
cd analise-sentimentos-b2w

2. Criar ambiente virtual
python -m venv venv

3. Ativar o ambiente virtual
Linux / macOS:
source venv/bin/activate

Windows:
venv\Scripts\activate

4. Instalar dependências
pip install --upgrade pip
pip install -r requirements.txt

5. Adicionar o dataset
Coloque o arquivo b2w.csv na pasta data/raw/.

6. Executar
python main.py

💡 Na primeira execução, o NLTK fará o download automático dos recursos punkt, punkt_tab e stopwords.

📊 Resultados Esperados
Após a execução, o projeto gera:

📈 Métricas do Modelo

Acurácia: ~85%

Relatório de Classificação:
              precision    recall  f1-score   support

Negativo (0)       0.85      0.84      0.85       500
Positivo (1)       0.86      0.87      0.86       500

    accuracy                           0.85      1000
   macro avg       0.85      0.85      0.85      1000
weighted avg       0.85      0.85      0.85      1000

Visualizações Geradas

Gráfico,Descrição,Localização
Nuvem de Palavras (Geral),Palavras mais frequentes em todas as avaliações,outputs/figures/wordcloud_todas.png
Nuvem de Palavras (Positivas),Palavras características de avaliações positivas,outputs/figures/wordcloud_positivas.png
Nuvem de Palavras (Negativas),Palavras características de avaliações negativas,outputs/figures/wordcloud_negativas.png
Matriz de Confusão,Heatmap mostrando acertos e erros do modelo,outputs/figures/matriz_confusao.png
Distribuição de Classes,Gráfico de barras com balanceamento do dataset,outputs/figures/distribuicao_classes.png

Modelos Salvos
sentiment_model.joblib — Modelo de Regressão Logística treinado
vectorizer.joblib — Vetorizador CountVectorizer ajustado

🛠 Tecnologias Utilizadas
Linguagem e Frameworks
Python 3.10+ — Linguagem principal
Scikit-learn — Modelos de Machine Learning
Pandas — Manipulação de dados
NLTK — Processamento de linguagem natural (tokenização + stopwords)
Matplotlib / Seaborn — Visualização de dados
Joblib — Persistência dos modelos

Tabela de Dependências
| Tecnologia | Versão | Uso |
| --- | --- | --- |
| Python | 3.10+ | Linguagem principal |
| pandas | 2.2.0 | Manipulação e análise de dados |
| numpy | 1.26.3 | Computação numérica |
| scikit-learn | 1.4.0 | Machine Learning e métricas |
| nltk | 3.8.1 | Tokenização e remoção de stopwords |
| matplotlib | 3.8.2 | Visualizações estáticas |
| seaborn | 0.13.1 | Visualizações estatísticas |
| wordcloud | 1.9.3 | Nuvens de palavras |
| PyYAML | 6.0.1 | Configurações em YAML |
| joblib | 1.3.2 | Serialização de modelos |


Ferramentas de Desenvolvimento
Git — Controle de versão
VS Code — IDE
Jupyter Notebook — Análise exploratória
Make — Automação de comandos

🧪 Testes
Para executar os testes unitários:
python -m pytest tests/ -v


📚 Conceitos Aplicados
Processamento de Linguagem Natural (PLN)

| Conceito | Descrição |
| --- | --- |
| Tokenização | Divisão do texto em palavras (tokens) via nltk.word_tokenize |
| Stopwords | Remoção de palavras irrelevantes (artigos, preposições) em português |
| Filtro Alfabético | Manutenção apenas de tokens compostos por letras (isalpha()) |
| Vetorização | Conversão de texto em representação numérica |
| Bag of Words | Modelo que conta frequência de palavras |
| CountVectorizer | Implementação do sklearn para BoW |

Machine Learning

| Conceito | Descrição |
| --- | --- |
| Classificação Binária | Duas classes: positivo (1) e negativo (0) |
| Regressão Logística | Modelo linear para classificação probabilística |
| Train/Test Split | Divisão dos dados em treino (80%) e teste (20%) |
| Stratification | Manter proporção das classes em ambas as partições |
| Métricas | Acurácia, Precisão, Recall, F1-Score |

Engenharia de Software

| Conceito | Descrição |
| --- | --- |
| SOLID | Princípios de design orientado a objetos |
| Arquitetura Modular | Separação de responsabilidades por contexto |
| Pipeline de Dados | Fluxo organizado: carga → limpeza → tokenização → vetorização → modelo |
| Configuração YAML | Parâmetros centralizados em config/config.yaml |
| Git | Versionamento de código |
| Type Hints | Documentação de tipos para melhor legibilidade |

🤝 Contribuição
Contribuições são bem-vindas! Siga os passos abaixo:

Faça um fork do projeto
Crie uma branch para sua feature:
git checkout -b feature/AmazingFeature

Commit suas mudanças:
git commit -m 'Add some AmazingFeature'

Push para a branch:
git push origin feature/AmazingFeature

Abra um Pull Request
📝 Licença
Este projeto está licenciado sob a MIT License [blocked].

Copyright (c) 2026 Marco Antonio Teixeira

👤 Autor
Marco Antonio Teixeira
Ano: 2026

📬 Contato
GitHub: github.com/MAntonioST
Email: m.antonyteixeira@gmail.com
📚 Referências
scikit-learn Documentation
Pandas Documentation
NLTK Book
Bag of Words Model
Logistic Regression