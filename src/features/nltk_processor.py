"""
Módulo: nltk_processor.py
Responsabilidade: processamento linguístico com NLTK.

Etapas disponíveis:
1. Tokenização;
2. Conversão para minúsculas;
3. Remoção de stopwords;
4. Manutenção de tokens alfabéticos;
5. Stemming em português.

Princípio SOLID:
- SRP: responsabilidade pelo processamento linguístico;
- OCP: novas etapas linguísticas podem ser adicionadas sem alterar
  o contrato principal do pipeline.
"""

from __future__ import annotations

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from src.features.stemming import PortugueseStemmer


class NLTKProcessor:
    """
    Processa textos usando tokenização, stopwords e stemming.

    A coluna original recebida é processada normalmente. O stemming,
    por sua vez, é salvo em uma nova coluna para preservar o resultado
    das etapas anteriores.
    """

    def __init__(self) -> None:
        """Inicializa recursos do NLTK e componentes linguísticos."""
        self._baixar_recursos()
        self.stops = set(stopwords.words("portuguese"))
        self._stemmer = PortugueseStemmer()

    def _baixar_recursos(self) -> None:
        """Baixa os recursos necessários do NLTK."""
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        nltk.download("stopwords", quiet=True)

    def processar(
        self,
        df: pd.DataFrame,
        coluna_texto: str,
    ) -> pd.DataFrame:
        """
        Tokeniza o texto e remove stopwords.

        A coluna informada é atualizada com o texto normalizado, mantendo
        o contrato já utilizado pelo pipeline atual.
        """
        print("\n" + "=" * 50)
        print("  NLTK — TOKENIZAÇÃO + STOPWORDS")
        print("=" * 50)
        print(
            f"[NLTK] Stopwords carregadas: "
            f"{len(self.stops)} palavras (pt-BR)"
        )

        def tokenizar_e_filtrar(texto: str) -> str:
            tokens = word_tokenize(texto.lower())

            tokens_filtrados = [
                token
                for token in tokens
                if token.isalpha() and token not in self.stops
            ]

            return " ".join(tokens_filtrados)

        df[coluna_texto] = df[coluna_texto].apply(tokenizar_e_filtrar)

        if not df.empty:
            exemplo = df[coluna_texto].iloc[0][:100]
            print(f"[NLTK] Exemplo processado: '{exemplo}...'")

        print("[NLTK] Concluído com sucesso.\n")

        return df

    def aplicar_stemming(
        self,
        df: pd.DataFrame,
        coluna_origem: str,
        coluna_destino: str,
    ) -> pd.DataFrame:
        """
        Aplica stemming sem destruir a coluna processada anteriormente.

        Args:
            df: DataFrame do pipeline.
            coluna_origem: coluna já tokenizada e sem stopwords.
            coluna_destino: nova coluna que receberá o texto stemmizado.
        """
        print("\n" + "=" * 50)
        print("  NLTK — STEMMING")
        print("=" * 50)

        df[coluna_destino] = self._stemmer.transformar(
            df[coluna_origem].fillna("").astype(str)
        )

        if not df.empty:
            exemplo = df[coluna_destino].iloc[0][:100]
            print(f"[NLTK] Exemplo stemmizado: '{exemplo}...'")

        print(
            f"[NLTK] Coluna criada: {coluna_destino}"
        )
        print("[NLTK] Stemming concluído com sucesso.\n")

        return df
