"""
Módulo: metrics.py
Responsabilidade: Calcular métricas de avaliação do modelo
Princípio SOLID: SRP (Single Responsibility Principle)
- Esta classe tem uma única responsabilidade: CALCULAR métricas
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


class MetricsCalculator:
    """
    Classe responsável por calcular métricas de desempenho do modelo.

    SRP: A única responsabilidade é CALCULAR métricas.
         Não faz plotagem, não treina, não carrega dados.

    Métricas calculadas:
    - Acurácia: proporção de predições corretas
    - Matriz de confusão: tabela com VPs, VNs, FPs, FNs
    - Relatório de classificação: precisão, recall, f1-score por classe
    """

    @staticmethod
    def acuracia(y_real: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calcula a acurácia do modelo.

        Acurácia = (VP + VN) / (VP + VN + FP + FN)
        onde:
            VP = Verdadeiros Positivos (modelo acertou o positivo)
            VN = Verdadeiros Negativos (modelo acertou o negativo)
            FP = Falsos Positivos (modelo errou ao prever positivo)
            FN = Falsos Negativos (modelo errou ao prever negativo)

        Args:
            y_real (np.ndarray): Classes verdadeiras.
            y_pred (np.ndarray): Classes preditas pelo modelo.

        Returns:
            float: Acurácia entre 0.0 e 1.0.
        """
        return accuracy_score(y_real, y_pred)

    @staticmethod
    def matriz_confusao(y_real: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calcula a matriz de confusão.

        Estrutura da matriz (para classificação binária):
            [[VN, FP],
             [FN, VP]]

        Args:
            y_real (np.ndarray): Classes verdadeiras.
            y_pred (np.ndarray): Classes preditas pelo modelo.

        Returns:
            np.ndarray: Matriz de confusão 2x2.
        """
        return confusion_matrix(y_real, y_pred)

    @staticmethod
    def relatorio_classificacao(y_real: np.ndarray, y_pred: np.ndarray) -> str:
        """
        Gera um relatório detalhado de classificação.

        O relatório inclui para cada classe (0 e 1):
        - Precision: VP / (VP + FP) — quantos positivos preditos são realmente positivos
        - Recall: VP / (VP + FN) — quantos positivos reais foram encontrados
        - F1-score: média harmônica de precision e recall
        - Support: quantidade de amostras de cada classe

        Args:
            y_real (np.ndarray): Classes verdadeiras.
            y_pred (np.ndarray): Classes preditas pelo modelo.

        Returns:
            str: Relatório em formato de texto.
        """
        return classification_report(
            y_real,
            y_pred,
            target_names=["Negativo (0)", "Positivo (1)"]
        )
