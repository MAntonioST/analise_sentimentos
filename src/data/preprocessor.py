"""
Módulo: preprocessor.py
Responsabilidade: Limpar e pré-processar os dados
Princípio SOLID: SRP (Single Responsibility Principle)
- Esta classe tem uma única razão para mudar: a lógica de limpeza dos dados
"""

import pandas as pd
import yaml


class DataPreprocessor:
    """
    Classe responsável por toda a limpeza e preparação dos dados.

    SRP: A única responsabilidade é PRÉ-PROCESSAR os dados brutos.
    OCP: Novos passos de limpeza podem ser adicionados sem modificar esta classe,
         bastando estender com novos métodos.

    Attributes:
        config (dict): Dicionário com as configurações de pré-processamento.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Inicializa o preprocessador com as configurações do projeto.

        Args:
            config_path (str): Caminho para o arquivo de configuração YAML.
        """
        with open(config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

    def remover_colunas(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove colunas desnecessárias do dataset.
        Colunas removidas: original_index, review_text_processed,
                           review_text_tokenized, rating,
                           kfold_polarity, kfold_rating.

        Args:
            df (pd.DataFrame): DataFrame original com todas as colunas.

        Returns:
            pd.DataFrame: DataFrame apenas com as colunas relevantes
                          (review_text e polarity).
        """
        colunas_para_remover = self.config["dataset"]["colunas_remover"]

        # Remove apenas as colunas que existem no DataFrame
        colunas_existentes = [col for col in colunas_para_remover if col in df.columns]
        df_limpo = df.drop(colunas_existentes, axis=1)

        print(f"[Preprocessor] Colunas removidas: {colunas_existentes}")
        print(f"[Preprocessor] Colunas restantes: {list(df_limpo.columns)}")
        return df_limpo

    def remover_nulos(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove linhas com valores nulos (NaN) do DataFrame.

        Args:
            df (pd.DataFrame): DataFrame com possíveis valores nulos.

        Returns:
            pd.DataFrame: DataFrame sem linhas contendo valores nulos.
        """
        linhas_antes = df.shape[0]
        df_sem_nulos = df.dropna(axis=0)

        linhas_removidas = linhas_antes - df_sem_nulos.shape[0]
        print(f"[Preprocessor] Linhas removidas (nulas): {linhas_removidas}")
        print(f"[Preprocessor] Linhas restantes: {df_sem_nulos.shape[0]}")

        return df_sem_nulos

    def processar(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executa todo o pipeline de pré-processamento:
        1. Remove colunas desnecessárias
        2. Remove linhas com valores nulos

        Args:
            df (pd.DataFrame): DataFrame bruto carregado pelo DataLoader.

        Returns:
            pd.DataFrame: DataFrame limpo e pronto para modelagem.
        """
        print("\n" + "=" * 50)
        print("  PRÉ-PROCESSAMENTO DOS DADOS")
        print("=" * 50)

        # Passo 1: Remover colunas irrelevantes
        df = self.remover_colunas(df)

        # Passo 2: Remover linhas com valores nulos
        df = self.remover_nulos(df)

        # Exibe a distribuição das classes após limpeza
        coluna_target = self.config["dataset"]["target_column"]
        print(f"\n[Preprocessor] Distribuição das classes:")
        print(df[coluna_target].value_counts())

        return df
