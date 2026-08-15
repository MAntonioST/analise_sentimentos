"""
Módulo: vectorizer.py
Responsabilidade: Transformar texto em representação numérica (TF-IDF)
Princípio SOLID: SRP (Single Responsibility Principle)
- Esta classe tem uma única responsabilidade: VETORIZAR texto
"""

import yaml
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer


class TextVectorizer:
    """
    Classe responsável por converter texto em vetores numéricos
    usando TF-IDF (Term Frequency - Inverse Document Frequency).

    SRP: A única responsabilidade é VETORIZAR o texto.
    OCP: Pode ser estendida para suportar n-grams e outros métodos
         sem modificar o código existente (apenas via configuração).

    Attributes:
        vectorizer (TfidfVectorizer): Instância do TfidfVectorizer do sklearn.
        max_features (int): Número máximo de features (termos) a extrair.
        ngram_range (tuple): Faixa de n-grams considerada (ex: (1,1), (1,2)).
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Inicializa o vetorizador com as configurações do projeto.

        Args:
            config_path (str): Caminho para o arquivo de configuração YAML.
        """
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        cfg_vec = config["vectorization"]

        self.max_features = cfg_vec["max_features"]

        # ngram_min/ngram_max são opcionais na config.
        # Se não existirem, o padrão é (1, 1) — apenas unigramas.
        ngram_min = cfg_vec.get("ngram_min", 1)
        ngram_max = cfg_vec.get("ngram_max", 1)
        self.ngram_range = (ngram_min, ngram_max)

        # TfidfVectorizer: pondera cada termo pela frequência no documento (TF)
        # e pela raridade dele no conjunto de documentos (IDF).
        # Isso reduz o peso de palavras muito comuns e destaca termos
        # mais discriminativos para a classificação de sentimento.
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            lowercase=False,
        )

    def fit_transform(self, textos):
        """
        Ajusta o vetorizador aos textos e transforma em matriz esparsa TF-IDF.

        Método que executa duas operações:
        1. fit: Aprende o vocabulário e calcula o IDF de cada termo
        2. transform: Converte cada texto em um vetor TF-IDF

        Args:
            textos (array-like): Lista/Series de textos a serem vetorizados.

        Returns:
            scipy.sparse.spmatrix: Matriz esparsa TF-IDF.
                Formato: (n_documentos, max_features)
        """
        print(
            f"[Vectorizer] Criando matriz TF-IDF com {self.max_features} "
            f"features (ngram_range={self.ngram_range})..."
        )
        matriz_tfidf = self.vectorizer.fit_transform(textos)

        print(
            f"[Vectorizer] Matriz gerada: {matriz_tfidf.shape[0]} docs x "
            f"{matriz_tfidf.shape[1]} features"
        )
        return matriz_tfidf

    def get_feature_names(self) -> list:
        """
        Retorna os nomes das features (termos) extraídos pelo vetorizador.

        Returns:
            list: Lista com os nomes das features.
        """
        return self.vectorizer.get_feature_names_out().tolist()

    def transformar_em_dataframe(self, matriz_tfidf) -> "pd.DataFrame":
        """
        Converte a matriz esparsa TF-IDF em um DataFrame do pandas.

        Args:
            matriz_tfidf: Matriz esparsa retornada pelo fit_transform.

        Returns:
            pd.DataFrame: DataFrame esparso com colunas nomeadas.
        """
        import pandas as pd
        return pd.DataFrame.sparse.from_spmatrix(
            matriz_tfidf,
            columns=self.vectorizer.get_feature_names_out()
        )
