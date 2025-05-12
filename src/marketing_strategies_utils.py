import pandas as pd

def filtrar_por_cluster_e_aisles(df, cluster_col='cluster', aisle_col='aisle', aisle_filtros=None):
    """
    Filtra o DataFrame com base nos clusters e nas seções (aisles) de interesse para cada cluster.

    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame contendo os dados com uma coluna de cluster.
    cluster_col : str
        Nome da coluna que indica o cluster de cada linha.
    aisle_col : str
        Nome da coluna com o nome das seções (aisles).
    aisle_filtros : dict
        Dicionário onde a chave é o número do cluster e o valor é uma lista de seções de interesse.

    Retorna:
    --------
    tuple
        Tupla com os DataFrames na ordem dos clusters definidos no dicionário.
    """
    return tuple(
        df[
            (df[cluster_col] == cluster_id) &
            (df[aisle_col].isin(aisles))
        ].reset_index(drop=True)
        for cluster_id, aisles in sorted(aisle_filtros.items())
    )
    
def score_produtos_por_aisle_xlsx(df, caminho_arquivo = None):
    """
    Gera um arquivo Excel com cada aba correspondente a uma seção de produtos ('aisle'),
    contendo os produtos dessa seção com colunas de contagem, porcentagem,
    taxa de recompra e score (porcentagem * (taxa_recompra ** 2)*1000).

    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame com colunas 'aisle', 'product_name' e 'reordered' (0 ou 1).
    caminho_arquivo : str
        Caminho do arquivo Excel a ser gerado (padrão None).
    
    Retorna:
    --------
    pd.DataFrame
        DataFrame contendo as métricas calculadas para cada produto, incluindo
        contagem, proporção dentro da seção, taxa de recompra e score.
    """

    # Agrupamento e cálculo de métricas
    df_score = (
        df.groupby(['aisle', 'product_name'])
        .agg(
            count=('product_name', 'size'),
            taxa_recompra=('reordered', 'mean')
        )
        .reset_index()
    )

    # Porcentagem dentro de cada aisle
    df_score['proporcao'] = df_score.groupby('aisle')['count'].transform(lambda x: x / x.sum())

    # Score
    df_score['score'] = df_score['proporcao'] * (df_score['taxa_recompra'] ** 2)*1000 #Escalando de 0 a 1000

    # Criando arquivo Excel com múltiplas abas e salvando arquivo caso seja informado um caminho
    if caminho_arquivo:
        with pd.ExcelWriter(caminho_arquivo, engine='openpyxl') as writer:
            for aisle, grupo in df_score.groupby('aisle'):
                nome_aba = str(aisle)[:31].replace('/', '-')  # Excel permite no máximo 31 caracteres no nome da aba
                grupo_sorted = grupo.sort_values('score', ascending=False)
                grupo_sorted.to_excel(writer, sheet_name=nome_aba, index=False)

    return df_score
