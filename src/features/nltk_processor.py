"""
Módulo: nltk_processor.py
Responsabilidade: Tokenização e remoção de stopwords com NLTK
Princípio SOLID: SRP (Single Responsibility Principle)
- Esta classe tem UMA única razão para mudar: a lógica de tokenização/stopwords

OCP: Aberta para extensão — novos métodos de stemming, lematização etc.
     podem ser adicionados como novos métodos ou subclasses.
"""

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class NLTKProcessor:
    """
    Classe responsável pelo pré-processamento linguístico com NLTK.

    Pipeline:
    1. Converte para minúsculas
    2. Tokeniza o texto (word_tokenize)
    3. Remove stopwords em português
    4. Mantém apenas tokens alfabéticos

    SRP: Única responsabilidade = processamento NLTK.
    DIP: Pode ser substituída por SpacyProcessor sem quebrar o pipeline.

    Attributes:
        stops (set): Conjunto de stopwords em português.
    """

    def __init__(self):
        """
        Inicializa o processador NLTK.
        Baixa os recursos necessários na primeira execução.
        """
        self._baixar_recursos()
        self.stops = set(stopwords.words('portuguese'))

    def _baixar_recursos(self) -> None:
        """Baixa os recursos NLTK (executado apenas uma vez)."""
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)

    def processar(self, df: pd.DataFrame, coluna_texto: str) -> pd.DataFrame:
        """
        Aplica tokenização e remoção de stopwords à coluna de texto.

        Args:
            df (pd.DataFrame): DataFrame com a coluna de texto bruto.
            coluna_texto (str): Nome da coluna que contém os textos.

        Returns:
            pd.DataFrame: DataFrame com a coluna de texto processada.
                          O texto original é SUBSTITUÍDO pelo processado
                          para manter compatibilidade com o pipeline.
        """
        print("\n" + "=" * 50)
        print("  NLTK — TOKENIZAÇÃO + STOPWORDS")
        print("=" * 50)
        print(f"[NLTK] Stopwords carregadas: {len(self.stops)} palavras (pt-BR)")

        def _tokenizar_e_filtrar(texto: str) -> str:
            tokens = word_tokenize(texto.lower())
            tokens_filtrados = [
                token for token in tokens
                if token.isalpha() and token not in self.stops
            ]
            return " ".join(tokens_filtrados)

        df[coluna_texto] = df[coluna_texto].apply(_tokenizar_e_filtrar)

        # Exibe um exemplo do antes e depois
        print(f"[NLTK] Exemplo processado: '{df[coluna_texto].iloc[0][:100]}...'")
        print(f"[NLTK] Concluído com sucesso.\n")

        return df
