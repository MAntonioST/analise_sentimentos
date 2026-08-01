"""
Módulo de Dados

Responsável por carregar e pré-processar os dados brutos.
Segue o princípio SRP: cada classe tem uma única razão para mudar.
"""

from .loader import DataLoader
from .preprocessor import DataPreprocessor
