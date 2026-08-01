"""
Módulo: vectorizer.py
Responsabilidade: Transformar texto em representação numérica (Bag of Words)
Princípio SOLID: SRP (Single Responsibility Principle)
- Esta classe tem uma única responsabilidade: VETORIZAR texto
"""

import yaml
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import CountVectorizer


class TextVectorizer:
    """
    Classe responsável por converter texto em vetores numéricos
    usando CountVectorizer (Bag of Words).

    SRP: A única responsabilidade é VETORIZAR o texto.
    OCP: Pode ser estendida para suportar outros métodos (TF-IDF, Word2Vec)
         sem modificar o código existente.

    Attributes:
        vectorizer (CountVectorizer): Instância do CountVectorizer do sklearn.
        max_features (int): Número máximo de features (palavras) a extrair.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Inicializa o vetorizador com as configurações do projeto.

        Args:
            config_path (str): Caminho para o arquivo de configuração YAML.
        """
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        self.max_features = config["vectorization"]["max_features"]

        # Inicializa o CountVectorizer com o número máximo de features
        # CountVectorizer: cria uma matriz de contagem de palavras (Bag of Words)
        # Cada célula representa quantas vezes uma palavra aparece em um documento
        self.vectorizer = CountVectorizer(max_features=self.max_features)

    def fit_transform(self, textos):
        """
        Ajusta o vetorizador aos textos e transforma em matriz esparsa.

        Método que executa duas operações:
        1. fit: Aprende o vocabulário (as palavras mais frequentes)
        2. transform: Converte cada texto em um vetor de contagem

        Args:
            textos (array-like): Lista/Series de textos a serem vetorizados.

        Returns:
            scipy.sparse.spmatrix: Matriz esparsa Bag of Words.
                Formato: (n_documentos, max_features)
                Cada linha = um documento, cada coluna = uma palavra do vocabulário
        """
        print(f"[Vectorizer] Criando Bag of Words com {self.max_features} features...")
        bag_of_words = self.vectorizer.fit_transform(textos)

        print(f"[Vectorizer] Matriz gerada: {bag_of_words.shape[0]} docs x {bag_of_words.shape[1]} features")
        return bag_of_words

    def get_feature_names(self) -> list:
        """
        Retorna os nomes das features (palavras) extraídas pelo vetorizador.

        Útil para:
        - Entender quais palavras são mais relevantes
        - Criar visualizações (ex: nuvem de palavras baseada em coeficientes)

        Returns:
            list: Lista com os nomes das features ordenadas alfabeticamente.
        """
        return self.vectorizer.get_feature_names_out().tolist()

    def transformar_em_dataframe(self, bag_of_words) -> "pd.DataFrame":
        """
        Converte a matriz esparsa BoW em um DataFrame do pandas para visualização.

        Args:
            bag_of_words: Matriz esparsa retornada pelo fit_transform.

        Returns:
            pd.DataFrame: DataFrame esparso com colunas nomeadas.
        """
        import pandas as pd
        return pd.DataFrame.sparse.from_spmatrix(
            bag_of_words,
            columns=self.vectorizer.get_feature_names_out()
        )
