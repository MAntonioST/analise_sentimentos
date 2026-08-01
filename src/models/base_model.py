"""
Módulo: base_model.py
Responsabilidade: Definir a interface (contrato) para todos os modelos
Princípio SOLID: ISP (Interface Segregation) + LSP (Liskov Substitution)

ISP: A interface define apenas o que todo modelo precisa (treinar, prever, avaliar).
LSP: Qualquer subclasse pode substituir esta classe base sem quebrar o sistema.
DIP: Módulos de alto nível (Orchestrator) dependem desta abstração, não de modelos concretos.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    """
    Classe abstrata que define o CONTRATO para todos os modelos de ML do projeto.

    Toda classe que herdar de BaseModel DEVE implementar:
    - treinar(): Para ajustar o modelo aos dados de treino
    - prever(): Para fazer predições em novos dados
    - avaliar(): Para medir a acurácia do modelo

    LSP: Qualquer modelo concreto (LogisticRegression, SVM, etc.) pode ser usado
         no lugar desta abstração sem alterar o comportamento do sistema.

    Attributes:
        modelo: Instância do modelo de ML (definido pela subclasse).
        nome (str): Nome descritivo do modelo.
    """

    def __init__(self, nome: str = "ModeloBase"):
        """
        Inicializa o modelo base.

        Args:
            nome (str): Nome identificador do modelo.
        """
        self.modelo = None  # Será instanciado pela subclasse
        self.nome = nome

    @abstractmethod
    def treinar(self, X, y):
        """
        Treina o modelo com os dados fornecidos.

        Método ABSTRATO: DEVE ser implementado por TODAS as subclasses.

        Args:
            X: Matriz de features (esparsa ou densa).
            y: Vetor de classes (labels).

        Returns:
            self: Retorna a própria instância (padrão fit do sklearn).
        """
        pass

    @abstractmethod
    def prever(self, X):
        """
        Realiza predições usando o modelo treinado.

        Método ABSTRATO: DEVE ser implementado por TODAS as subclasses.

        Args:
            X: Matriz de features para predição.

        Returns:
            np.ndarray: Array com as classes preditas.
        """
        pass

    def avaliar(self, X, y) -> float:
        """
        Avalia o modelo calculando a acurácia.

        Método CONCRETO: Pode ser sobrescrito pelas subclasses se necessário,
        mas já fornece uma implementação padrão válida.

        Args:
            X: Matriz de features de teste.
            y: Vetor de classes reais.

        Returns:
            float: Acurácia do modelo (entre 0.0 e 1.0).
        """
        from sklearn.metrics import accuracy_score
        y_pred = self.prever(X)
        return accuracy_score(y, y_pred)
