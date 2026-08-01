"""
Módulo de Modelos

Contém a abstração base e as implementações concretas dos modelos de ML.
Princípios aplicados:
- LSP (Liskov Substitution): Subtipos substituíveis pela classe base
- ISP (Interface Segregation): Interfaces específicas para cada propósito
- DIP (Dependency Inversion): Módulos de alto nível dependem de abstrações
"""

from .base_model import BaseModel
from .sentiment_classifier import SentimentClassifier
