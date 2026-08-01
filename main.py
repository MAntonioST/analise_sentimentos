"""
================================================================================
PROJETO: Análise de Sentimentos - B2W
CURSO: Pós-Graduação em IA para Devs - FIAP
DISCIPLINA: Processamento de Linguagem Natural (PLN)
AUTOR: Marco
================================================================================

ARQUITETURA:
    O projeto segue os princípios SOLID e arquitetura modular:
    
    S - Single Responsibility (SRP)
        ├── loader.py          → Responsável apenas por CARREGAR dados
        ├── preprocessor.py    → Responsável apenas por LIMPAR dados
        ├── vectorizer.py      → Responsável apenas por VETORIZAR texto
        ├── metrics.py         → Responsável apenas por CALCULAR métricas
        └── visualizer.py      → Responsável apenas por CRIAR gráficos
    
    O - Open/Closed (OCP)
        └── orchestrator.py    → Aberto para extensão, fechado para modificação
    
    L - Liskov Substitution (LSP)
        └── base_model.py      → Subtipos substituíveis pela classe base
    
    I - Interface Segregation (ISP)
        └── base_model.py      → Interface com apenas métodos necessários
    
    D - Dependency Inversion (DIP)
        └── orchestrator.py    → Depende de BaseModel, não de LogisticRegression

EXECUÇÃO:
    python main.py
================================================================================
"""

import sys
import os

# Adiciona o diretório src ao path para permitir imports absolutos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ===================== IMPORTAÇÕES =====================
from data.loader import DataLoader
from data.preprocessor import DataPreprocessor
from features.vectorizer import TextVectorizer
from models.sentiment_classifier import SentimentClassifier
from evaluation.visualizer import Visualizer
from evaluation.metrics import MetricsCalculator
from pipeline.orchestrator import PipelineOrchestrator


# ===================== CONFIGURAÇÃO =====================
CONFIG_PATH = "config/config.yaml"


def main():
    """
    Ponto de entrada do projeto.
    Instancia todos os componentes e executa o pipeline completo.
    """
    print("\n" + "=" * 60)
    print("  ANÁLISE DE SENTIMENTOS - B2W")
    print("  Pós-Graduação IA para Devs - FIAP")
    print("=" * 60)

    # ============================================
    # INSTANCIAÇÃO DOS COMPONENTES
    # ============================================
    # Cada componente é instanciado separadamente (SRP)
    # A dependência entre eles é gerenciada pelo orquestrador (DIP)

    loader = DataLoader(CONFIG_PATH)
    preprocessor = DataPreprocessor(CONFIG_PATH)
    vectorizer = TextVectorizer(CONFIG_PATH)
    modelo = SentimentClassifier()
    visualizer = Visualizer(CONFIG_PATH)
    metrics_calculator = MetricsCalculator()
    orchestrator = PipelineOrchestrator(CONFIG_PATH)

    # ============================================
    # EXECUÇÃO DO PIPELINE
    # ============================================
    # O orquestrador coordena todos os componentes
    # sem conhecer os detalhes de implementação (DIP)
    try:
        resultado = orchestrator.executar(
            loader=loader,
            preprocessor=preprocessor,
            vectorizer=vectorizer,
            modelo=modelo,
            visualizer=visualizer,
            metrics_calculator=metrics_calculator
        )

    except FileNotFoundError as e:
        print(f"\n ERRO: {e}")
        print("Certifique-se de que o b2w.csv está em data/raw/")
        sys.exit(1)

    except Exception as e:
        print(f"\n ERRO INESPERADO: {e}")
        print("Verifique o log acima para mais detalhes.")
        sys.exit(1)


# ===================== PONTO DE ENTRADA =====================
if __name__ == "__main__":
    main()
