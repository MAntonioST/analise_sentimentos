"""
Módulo: visualizer.py
Responsabilidade: Criar visualizações (wordclouds, matriz de confusão, etc.)
Princípio SOLID: SRP (Single Responsibility Principle)
- Esta classe tem uma única responsabilidade: CRIAR VISUALIZAÇÕES
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import yaml


class Visualizer:
    """
    Classe responsável por gerar todas as visualizações do projeto.

    SRP: A única responsabilidade é GERAR GRÁFICOS.
         Não processa dados, não treina modelos, não calcula métricas.

    Attributes:
        config (dict): Configurações do projeto (wordcloud, outputs).
        output_dir (str): Diretório onde os gráficos serão salvos.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Inicializa o visualizador com as configurações do projeto.

        Args:
            config_path (str): Caminho para o arquivo de configuração YAML.
        """
        with open(config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

        self.output_dir = self.config["outputs"]["figures_dir"]
        # Cria a pasta de saída se não existir
        os.makedirs(self.output_dir, exist_ok=True)

    def nuvem_palavras(self, df: pd.DataFrame, coluna_texto: str, polaridade: int = None) -> None:
        """
        Gera e exibe uma nuvem de palavras (WordCloud).

        A nuvem de palavras mostra as palavras mais frequentes nos textos,
        onde o tamanho da fonte é proporcional à frequência da palavra.

        Args:
            df (pd.DataFrame): DataFrame contendo os textos.
            coluna_texto (str): Nome da coluna com os textos.
            polaridade (int, optional): Se None, gera nuvem geral.
                                         Se 0, apenas textos negativos.
                                         Se 1, apenas textos positivos.
        """
        # Filtra por polaridade se especificado
        if polaridade is not None:
            df_filtrado = df.query(f"polarity == {polaridade}")
            titulo = "Positivas" if polaridade == 1 else "Negativas"
        else:
            df_filtrado = df
            titulo = "Todas as Avaliações"

        # Junta todos os textos em uma única string
        # Isso é necessário porque o WordCloud espera um texto contínuo
        todas_avaliacoes = [str(texto) for texto in df_filtrado[coluna_texto]]
        todas_palavras = ' '.join(todas_avaliacoes)

        # Obtém as configurações da WordCloud do config.yaml
        wc_config = self.config["wordcloud"]

        # Cria a nuvem de palavras
        # collocations=False: Evita agrupar bigramas (pares de palavras frequentes)
        nuvem = WordCloud(
            width=wc_config["width"],
            height=wc_config["height"],
            max_font_size=wc_config["max_font_size"],
            collocations=wc_config["collocations"]
        ).generate(todas_palavras)

        # Configura e exibe o gráfico
        plt.figure(figsize=(10, 7))
        plt.imshow(nuvem, interpolation='bilinear')
        plt.axis("off")  # Remove os eixos para visualização mais limpa
        plt.title(f"Nuvem de Palavras - Avaliações {titulo}", fontsize=16, pad=20)

        # Salva a figura
        nome_arquivo = f"wordcloud_{titulo.lower().replace(' ', '_')}.png"
        caminho = os.path.join(self.output_dir, nome_arquivo)
        plt.savefig(caminho, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()

        print(f"[Visualizer] Nuvem de palavras salva em: {caminho}")

    def matriz_confusao(self, cm: np.ndarray) -> None:
        """
        Plota a matriz de confusão como um heatmap.

        A matriz de confusão mostra:
        - Diagonal principal: acertos do modelo (VP e VN)
        - Diagonal secundária: erros do modelo (FP e FN)

        Args:
            cm (np.ndarray): Matriz de confusão 2x2.
        """
        plt.figure(figsize=(6, 5))

        # Heatmap: gráfico de calor que facilita a visualização de valores
        # annot=True: Mostra os números dentro das células
        # fmt='d': Formata como inteiro decimal
        # cmap='Blues': Mapa de cores em tons de azul
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=["Negativo (0)", "Positivo (1)"],
            yticklabels=["Negativo (0)", "Positivo (1)"]
        )

        plt.title("Matriz de Confusão", fontsize=14, pad=15)
        plt.ylabel("Classe Real")
        plt.xlabel("Classe Predita")
        plt.tight_layout()

        # Salva a figura
        caminho = os.path.join(self.output_dir, "matriz_confusao.png")
        plt.savefig(caminho, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()

        print(f"[Visualizer] Matriz de confusão salva em: {caminho}")

    def distribuicao_classes(self, df: pd.DataFrame, coluna_target: str) -> None:
        """
        Plota a distribuição das classes (positivo vs negativo).

        Útil para:
        - Verificar balanceamento do dataset
        - Identificar possíveis vieses nos dados

        Args:
            df (pd.DataFrame): DataFrame com os dados.
            coluna_target (str): Nome da coluna de classes.
        """
        plt.figure(figsize=(8, 5))

        # Contagem de cada classe
        contagem = df[coluna_target].value_counts()

        # Gráfico de barras
        cores = ['#FF6B6B', '#4ECDC4']  # Vermelho para negativo, Verde para positivo
        plt.bar(['Negativo (0)', 'Positivo (1)'], contagem.values, color=cores, edgecolor='black')

        # Adiciona os valores no topo de cada barra
        for i, valor in enumerate(contagem.values):
            plt.text(i, valor + 10, str(valor), ha='center', fontsize=12, fontweight='bold')

        plt.title("Distribuição das Classes (Polaridade)", fontsize=14, pad=15)
        plt.ylabel("Quantidade de Avaliações")
        plt.tight_layout()

        # Salva a figura
        caminho = os.path.join(self.output_dir, "distribuicao_classes.png")
        plt.savefig(caminho, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()

        print(f"[Visualizer] Gráfico de distribuição salvo em: {caminho}")
