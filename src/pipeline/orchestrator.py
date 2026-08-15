"""
Módulo: orchestrator.py
Responsabilidade: Orquestrar o pipeline completo de análise de sentimentos
Princípio SOLID: OCP (Open/Closed Principle) + DIP (Dependency Inversion)

OCP: O pipeline é aberto para extensão (novos passos podem ser adicionados)
     mas fechado para modificação (o fluxo base não muda).
DIP: Depende de abstrações (BaseModel), não de implementações concretas.
"""

import os
import yaml
import joblib
from sklearn.model_selection import train_test_split


class PipelineOrchestrator:
    """
    Classe responsável por ORQUESTRAR todo o fluxo do projeto.

    OCP: Aberto para extensão — novos modelos, vetorizadores ou visualizadores
         podem ser injetados sem modificar esta classe.
    DIP: Depende de abstrações (BaseModel), permitindo trocar o modelo
         sem alterar o código do orquestrador.

    Fluxo do pipeline:
    1. Carregar dados
    2. Pré-processar
    3. Tokenizar e remover stopwords
    4. Aplicar stemming
    5. Split treino/teste (ANTES da vetorização, evita vazamento)
    6. Vetorizar (fit apenas no treino, transform no teste)
    7. Treinar modelo
    8. Avaliar modelo
    9. Gerar visualizações
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Inicializa o orquestrador carregando as configurações.

        Args:
            config_path (str): Caminho para o arquivo de configuração YAML.
        """
        with open(config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

    def executar(
        self,
        loader,
        preprocessor,
        nltk_processor,
        vectorizer,
        modelo,
        visualizer,
        metrics_calculator
    ):
        """
        Executa o pipeline completo de análise de sentimentos.

        DIP: Todos os parâmetros são abstrações, permitindo trocar
             qualquer componente sem modificar este método.

        Args:
            loader: Instância de DataLoader.
            preprocessor: Instância de DataPreprocessor.
            nltk_processor: Instância de NLTKProcessor.
            vectorizer: Instância de TextVectorizer.
            modelo: Instância de BaseModel (ou qualquer subclasse).
            visualizer: Instância de Visualizer.
            metrics_calculator: Instância de MetricsCalculator.

        Returns:
            dict: Dicionário com os resultados do pipeline:
                - 'acuracia': Acurácia do modelo (float)
                - 'matriz_confusao': Matriz de confusão (np.ndarray)
                - 'relatorio': Relatório de classificação (str)
        """
        # ============================================
        # ETAPA 1: CARREGAR OS DADOS
        # ============================================
        print("\n" + "=" * 60)
        print("  PIPELINE DE ANÁLISE DE SENTIMENTOS - B2W")
        print("=" * 60)

        df = loader.carregar()

        # ============================================
        # ETAPA 2: PRÉ-PROCESSAR
        # ============================================
        df = preprocessor.processar(df)

        # ============================================
        # ETAPA 2.5: NLTK — TOKENIZAÇÃO + STOPWORDS
        # ============================================
        coluna_texto = self.config["dataset"]["text_column"]
        df = nltk_processor.processar(df, coluna_texto)

        # ============================================
        # ETAPA 2.6: NLTK — STEMMING
        # ============================================
        coluna_texto_stemmed = f"{coluna_texto}_stemmed"

        df = nltk_processor.aplicar_stemming(
            df=df,
            coluna_origem=coluna_texto,
            coluna_destino=coluna_texto_stemmed,
        )

        # ============================================
        # ETAPA 3: SPLIT TREINO/TESTE (ANTES DA VETORIZAÇÃO)
        # ============================================
        print("\n" + "=" * 50)
        print("  DIVISÃO TREINO/TESTE")
        print("=" * 50)

        textos = df[coluna_texto_stemmed]
        y = df[self.config["dataset"]["target_column"]]

        cfg_train = self.config["training"]

        # train_test_split: Divide os dados em treino e teste
        # stratify: Mantém a proporção das classes em ambas as partições
        # random_state: Garante que o split seja sempre o mesmo (reprodutibilidade)
        textos_train, textos_test, y_train, y_test = train_test_split(
            textos,
            y,
            test_size=cfg_train["test_size"],
            stratify=y if cfg_train["stratify"] else None,
            random_state=cfg_train["random_state"]
        )

        print(f"[Pipeline] Treino: {len(textos_train)} amostras")
        print(f"[Pipeline] Teste:  {len(textos_test)} amostras")
        print(f"[Pipeline] Distribuição treino:\n{y_train.value_counts()}")

        # ============================================
        # ETAPA 4: VETORIZAR (fit apenas no treino)
        # ============================================
        print("\n" + "=" * 50)
        print("  VETORIZAÇÃO DO TEXTO")
        print("=" * 50)

        # fit_transform aprende o vocabulário e o IDF SOMENTE com o treino.
        # transform aplica esse mesmo vocabulário/IDF ao teste,
        # sem deixar o teste "vazar" informação para o treinamento.
        X_train = vectorizer.fit_transform(textos_train)
        X_test = vectorizer.transform(textos_test)

        # ============================================
        # ETAPA 5: TREINAR MODELO
        # ============================================
        print("\n" + "=" * 50)
        print("  TREINAMENTO DO MODELO")
        print("=" * 50)

        modelo.treinar(X_train, y_train)

        # ============================================
        # ETAPA 6: AVALIAR MODELO
        # ============================================
        print("\n" + "=" * 50)
        print("  AVALIAÇÃO DO MODELO")
        print("=" * 50)

        # Predições no conjunto de teste
        y_pred = modelo.prever(X_test)

        # Calcula as métricas
        acuracia = metrics_calculator.acuracia(y_test, y_pred)
        cm = metrics_calculator.matriz_confusao(y_test, y_pred)
        relatorio = metrics_calculator.relatorio_classificacao(y_test, y_pred)

        print(f"\n Acurácia: {acuracia:.2%}")
        print(f"\n Relatório de Classificação:\n{relatorio}")

        # ============================================
        # ETAPA 7: VISUALIZAÇÕES
        # ============================================
        print("\n" + "=" * 50)
        print("  GERANDO VISUALIZAÇÕES")
        print("=" * 50)

        # Distribuição das classes
        coluna_target = self.config["dataset"]["target_column"]
        visualizer.distribuicao_classes(df, coluna_target)

        # Nuvem de palavras geral
        visualizer.nuvem_palavras(
            df,
            coluna_texto_stemmed,
            polaridade=None,
        )

        # Nuvem de palavras - apenas avaliações negativas (polarity == 0)
        visualizer.nuvem_palavras(
            df,
            coluna_texto_stemmed,
            polaridade=0,
        )

        # Nuvem de palavras - apenas avaliações positivas (polarity == 1)
        visualizer.nuvem_palavras(
            df,
            coluna_texto_stemmed,
            polaridade=1,
        )

        # Matriz de confusão
        visualizer.matriz_confusao(cm)

        # ============================================
        # ETAPA 8: SALVAR MODELO
        # ============================================
        self._salvar_modelo(modelo, vectorizer)

        # ============================================
        # RESUMO FINAL
        # ============================================
        print("\n" + "=" * 60)
        print("  PIPELINE CONCLUÍDO COM SUCESSO!")
        print("=" * 60)
        print(f"  Acurácia: {acuracia:.2%}")
        print(f"  Gráficos salvos em: {self.config['outputs']['figures_dir']}/")
        print(f"  Modelo salvo em: {self.config['outputs']['models_dir']}/")
        print("=" * 60)

        # Retorna os resultados para possível uso posterior
        return {
            "acuracia": acuracia,
            "matriz_confusao": cm,
            "relatorio": relatorio,
            "modelo": modelo,
            "vectorizer": vectorizer
        }

    def _salvar_modelo(self, modelo, vectorizer) -> None:
        """
        Salva o modelo treinado e o vetorizador em disco usando joblib.

        O joblib é mais eficiente que pickle para objetos do sklearn
        porque lida melhor com arrays numpy grandes.

        Args:
            modelo: Modelo treinado (BaseModel).
            vectorizer: Vetorizador ajustado (TextVectorizer).
        """
        models_dir = self.config["outputs"]["models_dir"]
        os.makedirs(models_dir, exist_ok=True)

        # Salva o modelo
        caminho_modelo = os.path.join(models_dir, "sentiment_model.joblib")
        joblib.dump(modelo, caminho_modelo)

        # Salva o vetorizador (necessário para transformar novos textos)
        caminho_vec = os.path.join(models_dir, "vectorizer.joblib")
        joblib.dump(vectorizer, caminho_vec)

        print(f"[Pipeline] Modelo salvo em: {caminho_modelo}")
        print(f"[Pipeline] Vetorizador salvo em: {caminho_vec}")
