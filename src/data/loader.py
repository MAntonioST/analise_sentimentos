"""
Módulo: loader.py
Responsabilidade: Carregar o dataset do disco
Princípio SOLID: SRP (Single Responsibility Principle)
- Esta classe tem uma única razão para mudar: a fonte de dados
"""

import os
import pandas as pd
import yaml


class DataLoader:
    """
    Classe responsável por carregar dados de diferentes fontes.

    SRP: A única responsabilidade é CARREGAR dados.
    DIP: Depende da abstração do pandas DataFrame, não de um formato específico.

    Attributes:
        config (dict): Dicionário com as configurações do dataset.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Inicializa o DataLoader carregando as configurações do arquivo YAML.

        Args:
            config_path (str): Caminho para o arquivo de configuração YAML.
        """
        with open(config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

    def carregar(self) -> pd.DataFrame:
        """
        Carrega o dataset do caminho especificado nas configurações.

        Returns:
            pd.DataFrame: DataFrame com os dados brutos do B2W.

        Raises:
            FileNotFoundError: Se o arquivo de dados não for encontrado.
        """
        caminho = self.config["dataset"]["path"]

        # Verifica se o arquivo existe antes de tentar carregar
        if not os.path.exists(caminho):
            raise FileNotFoundError(
                f"Arquivo não encontrado: {caminho}\n"
                f"Certifique-se de que o dataset b2w.csv está na pasta data/raw/"
            )

        print(f"[DataLoader] Carregando dataset de: {caminho}")
        df = pd.read_csv(caminho)
        print(f"[DataLoader] Dataset carregado: {df.shape[0]} linhas x {df.shape[1]} colunas")
        return df
