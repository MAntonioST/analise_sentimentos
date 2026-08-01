"""
Módulo: sentiment_classifier.py
Responsabilidade: Implementar o classificador de sentimentos com Regressão Logística
Princípio SOLID: LSP (Liskov Substitution Principle)
- Esta classe implementa o contrato definido por BaseModel
- Pode ser substituída por qualquer outro modelo sem quebrar o sistema
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from .base_model import BaseModel


class SentimentClassifier(BaseModel):
    """
    Classificador de sentimentos usando Regressão Logística.

    LSP: Implementa todos os métodos abstratos de BaseModel.
         Pode substituir BaseModel em qualquer lugar do código.

    A Regressão Logística é um modelo linear para classificação binária.
    Ela estima a probabilidade de um texto pertencer à classe positiva (polarity=1)
    ou negativa (polarity=0).

    Attributes:
        modelo (LogisticRegression): Instância do modelo sklearn.
        nome (str): Nome do modelo.
    """

    def __init__(self, nome: str = "RegressaoLogistica"):
        """
        Inicializa o classificador de sentimentos.

        Args:
            nome (str): Nome descritivo do modelo.
        """
        super().__init__(nome=nome)
        # Instancia o modelo de Regressão Logística do sklearn
        # max_iter=1000: Número máximo de iterações para convergência
        # random_state=71: Semente para reprodutibilidade (mesmo valor da aula)
        self.modelo = LogisticRegression(max_iter=1000, random_state=71)

    def treinar(self, X, y) -> "SentimentClassifier":
        """
        Treina o modelo de Regressão Logística com os dados fornecidos.

        O processo de treinamento:
        1. Ajusta os coeficientes (pesos) para cada feature
        2. Encontra a fronteira de decisão que melhor separa as classes
        3. Minimiza a função de perda logística

        Args:
            X: Matriz Bag of Words (esparsa) com shape (n_amostras, n_features).
            y: Vetor de polaridade (0=negativo, 1=positivo).

        Returns:
            SentimentClassifier: A própria instância (padrão fit do sklearn).
        """
        print(f"[SentimentClassifier] Treinando modelo '{self.nome}'...")
        print(f"[SentimentClassifier] Amostras de treino: {X.shape[0]}")
        print(f"[SentimentClassifier] Features: {X.shape[1]}")

        # fit(): Método do sklearn que realiza o treinamento
        # O modelo aprende a relação entre as palavras (features) e os sentimentos (labels)
        self.modelo.fit(X, y)

        print(f"[SentimentClassifier] Modelo treinado com sucesso!")
        return self

    def prever(self, X) -> np.ndarray:
        """
        Prediz o sentimento (0 ou 1) para novos textos vetorizados.

        Args:
            X: Matriz Bag of Words dos textos a classificar.

        Returns:
            np.ndarray: Array com as predições (0=negativo, 1=positivo).
        """
        return self.modelo.predict(X)

    def prever_proba(self, X) -> np.ndarray:
        """
        Retorna a probabilidade de cada classe.
        Útil para:
        - Entender a confiança do modelo em cada predição
        - Ajustar thresholds de decisão

        Args:
            X: Matriz de features.

        Returns:
            np.ndarray: Matriz com shape (n_amostras, 2).
                Coluna 0: probabilidade da classe negativa
                Coluna 1: probabilidade da classe positiva
        """
        return self.modelo.predict_proba(X)
