"""
Módulo: stemming.py
Responsabilidade: aplicar stemming em textos em português.

Princípio SOLID:
- SRP: este módulo possui somente a responsabilidade de reduzir
  palavras às suas raízes por meio do algoritmo RSLP.
"""

from __future__ import annotations

from collections.abc import Iterable

from nltk.stem import RSLPStemmer


class PortugueseStemmer:
    """Aplica o algoritmo RSLP de stemming para português."""

    def __init__(self) -> None:
        self._stemmer = RSLPStemmer()

    def transformar_texto(self, texto: str) -> str:
        """
        Aplica stemming aos tokens de um texto já tokenizado.

        O texto recebido deve estar em formato normalizado, com tokens
        separados por espaços e sem stopwords.
        """
        if not isinstance(texto, str) or not texto.strip():
            return ""

        tokens = texto.split()
        stems = [self._stemmer.stem(token) for token in tokens]

        return " ".join(stems)

    def transformar(self, textos: Iterable[str]) -> list[str]:
        """Aplica stemming a uma coleção de textos."""
        return [
            self.transformar_texto(texto)
            for texto in textos
        ]
