import pandas as pd
import plotly.express as px
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


def gerar_dummies_transacoes(df_amostrado, coluna):
    """
    Transforma o DataFrame de itens por transações em uma matriz binária (uma linha por transação, colunas com dummies da variável de interesse).

    Parâmetros:
    -----------
    df_amostrado : pd.DataFrame
        DataFrame contendo pelo menos as colunas 'order_id' e a variável de interesse para dummies (ex: 'aisle').
    coluna : str
        Nome da variável que será usada para representar os itens da transação.

    Retorna:
    --------
    pd.DataFrame
        DataFrame no formato de matriz binária, com uma linha por transação e colunas indicando a presença de cada categoria da variável especificada.
    """
    transacoes = df_amostrado.groupby('order_id')[coluna].apply(list)
    te = TransactionEncoder()
    te_array = te.fit(transacoes).transform(transacoes)
    df_transacoes = pd.DataFrame(te_array, columns=te.columns_)
    return df_transacoes


def gerar_regras_apriori(df_transacoes, metric, min_support=0.02, min_threshold=0):
    """
    Gera regras de associação a partir de transações binárias utilizando o algoritmo Apriori.

    Parâmetros:
    -----------
    df_transacoes : pd.DataFrame
        DataFrame binário com as transações (output da função gerar_dummies_transacoes).
    min_support : float 
        Suporte mínimo para considerar itemsets frequentes (padrão 0.02).
    metric : str
        Métrica usada para geração das regras (ex: 'confidence', 'lift', 'support').
    min_threshold : float 
        Valor mínimo da métrica escolhida para filtrar as regras (padrão 0).

    Retorna:
    --------
    pd.DataFrame
        DataFrame contendo as regras geradas, ordenadas pela métrica escolhida.
    """
    itemsets = apriori(df_transacoes, min_support=min_support, use_colnames=True)
    regras = association_rules(itemsets, metric=metric, min_threshold=min_threshold)
    return regras.sort_values(by=metric, ascending=False)


def plot_dispersao_support_lift_interativo(
    regras,
    min_lift = 0,
    eixo_x='support',
    eixo_y='lift',
    titulo=None,
    xlabel=None,
    ylabel=None,
    tamanho_ponto=5,
    cor = 'deepskyblue',
    opacidade=0.7
):
    """
    Plota um scatter plot interativo das regras de associação:
    - Eixo X: suporte (ou outro especificado)
    - Eixo Y: lift (ou outro especificado)
    - Remove duplicatas ignorando a ordem entre antecedents e consequents
    - Mostra antecedents e consequents como sets separados com uma seta ⟷
    - Apenas regras com lift > min_lift

    Parâmetros:
    -----------
    regras : pd.DataFrame
        DataFrame contendo 'support', 'lift', 'confidence', 'antecedents' e 'consequents'.
    min_lift : float
        Valor mínimo de lift para filtrar as regras (padrão 0).
    eixo_x : str
        Nome da coluna no eixo X.
    eixo_y : str
        Nome da coluna no eixo Y.
    titulo : str ou None
        Título do gráfico (gerado automaticamente se None).
    xlabel : str ou None
        Rótulo do eixo X (usa nome da coluna se None).
    ylabel : str ou None
        Rótulo do eixo Y (usa nome da coluna se None).
    tamanho_ponto : int
        Tamanho dos pontos (padrão 5).
    cor : str
        Cor dos pontos (padrão 'deepskyblue').
    opacidade : float
        Opacidade dos pontos (padrão 0.7).
    """

    # Filtrando regras com lift > min_lift
    regras_filtradas = regras[regras['lift'] >= min_lift].copy()
    
    # Transformando antecedents e consequents em strings
    regras_filtradas['antecedents'] = regras_filtradas['antecedents'].apply(
        lambda x: ', '.join(x) if isinstance(x, (set, frozenset)) else x
    )
    regras_filtradas['consequents'] = regras_filtradas['consequents'].apply(
        lambda x: ', '.join(x) if isinstance(x, (set, frozenset)) else x
    )

    # Criando versões ordenadas dos antecedentes e consequentes
    regras_filtradas['antecedents_sorted'] = regras_filtradas['antecedents'].apply(lambda x: sorted(list(x)))
    regras_filtradas['consequents_sorted'] = regras_filtradas['consequents'].apply(lambda x: sorted(list(x)))

    # Criado chave para identificar duplicatas
    regras_filtradas['chave'] = regras_filtradas.apply(
        lambda row: (row['antecedents_sorted'], row['consequents_sorted']) 
        if row['antecedents_sorted'] <= row['consequents_sorted'] #Comparação lexicográfica
        else (row['consequents_sorted'], row['antecedents_sorted']),
        axis=1
    )

    # Removendo duplicatas baseadas na chave
    regras_filtradas = regras_filtradas.drop_duplicates(subset='chave')

    # Removendo as colunas auxiliares
    regras_filtradas = regras_filtradas.drop(columns=['antecedents_sorted', 'consequents_sorted', 'chave'])

    # Criando o texto formatado para o hover
    regras_filtradas['regra'] = regras_filtradas.apply(
        lambda row: f"{{{row['antecedents']}}} ⟷ {{{row['consequents']}}}",
        axis=1
    )
    
    # Hover 
    hover_data = {
        'regra': True,
        'support': ':.4f',
        'lift': ':.4f'
    }

    # Título automático se não fornecido
    if titulo is None:
        titulo = f'{eixo_x.capitalize()} vs {eixo_y.capitalize()}'

    # Labels automáticos se não fornecidos
    if xlabel is None:
        xlabel = eixo_x.capitalize()
    if ylabel is None:
        ylabel = eixo_y.capitalize()
        
    # Calculando médias
    media_x = regras_filtradas[eixo_x].mean()
    media_y = regras_filtradas[eixo_y].mean()

    # Criando gráfico
    fig = px.scatter(
        regras_filtradas,
        x=eixo_x,
        y=eixo_y,
        hover_data=hover_data,
        labels={eixo_x: xlabel, eixo_y: ylabel},
        title=titulo
    )

    fig.update_traces(
        marker=dict(size=tamanho_ponto, opacity=opacidade, color=cor),
        selector=dict(mode='markers')
    )

    # Linha vertical média suporte
    fig.add_shape(
        type="line",
        x0=media_x, x1=media_x,
        y0=regras_filtradas[eixo_y].min(), y1=regras_filtradas[eixo_y].max(),
        line=dict(color="gray", dash="dot"),
        name='Média Support'
    )

    # Linha horizontal média lift
    fig.add_shape(
        type="line",
        x0=regras_filtradas[eixo_x].min(), x1=regras_filtradas[eixo_x].max(),
        y0=media_y, y1=media_y,
        line=dict(color="gray", dash="dot"),
        name='Média Lift'
    )

    # Caixa com valor médio do Support
    fig.add_annotation(
        x=media_x, y=regras_filtradas[eixo_y].max(),
        text=f"Média Support = {media_x:.3f}",
        showarrow=False,
        font=dict(size=12, color="black"),
        bgcolor="white",
        bordercolor="gray",
        borderwidth=1,
        borderpad=4,
        xanchor="left",
        yanchor="top"
    )

    # Caixa com valor médio do Lift
    fig.add_annotation(
        x=regras_filtradas[eixo_x].max(), y=media_y,
        text=f"Média Lift = {media_y:.3f}",
        showarrow=False,
        font=dict(size=12, color="black"),
        bgcolor="white",
        bordercolor="gray",
        borderwidth=1,
        borderpad=4,
        xanchor="right",
        yanchor="bottom"
    )

    fig.update_layout(
        plot_bgcolor='white',
        title_x=0.5,
        hoverlabel=dict(bgcolor="white", font_size=12),
        xaxis=dict(
            gridcolor='lightgray',
            zeroline=False,
            title=xlabel,
            tick0=0,
            dtick=0.01
        ),
        yaxis=dict(
            gridcolor='lightgray',
            zeroline=False,
            title=ylabel,
            tickmode='linear',
            dtick=0.2
        ),
        font=dict(size=13),
        showlegend=False
    )

    fig.show()
    
def gerar_regras_por_cluster(df_com_cluster, coluna='aisle', departments_filtros=None, min_support=0.02, min_threshold=0):
    """
    Gera regras de associação (Apriori) separadas por cluster de usuários, com filtro por departamento.

    Parâmetros:
    -----------
    df_com_cluster : pd.DataFrame
        DataFrame contendo os dados transacionais e a coluna 'cluster' associada a cada usuário.
        Deve conter também as colunas 'user_id', 'department' e a coluna especificada em `coluna`.
    
    coluna : str 
        Nome da coluna a ser usada para gerar dummies (ex: 'aisle', 'product_name') (padrão = 'aisle').

    departments_filtros : dict, opcional
        Dicionário com filtros por cluster, onde a chave é o número do cluster (0 a 2) e o valor é
        uma lista de nomes de departamentos que devem ser mantidos para aquele cluster.
        Exemplo:
            {
                0: ['produce', 'snacks'],
                1: ['beverages', 'pantry'],
                ...
            }

    min_support : float, opcional
        Suporte mínimo para considerar itemsets frequentes (padrão 0.02).
        
    min_threshold : float, opcional 
        Valor mínimo da métrica escolhida para filtrar as regras (padrão 0).

    Retorna:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Tupla contendo 3 DataFrames, cada um com as regras de associação geradas para um dos clusters (0 a 2)
    """
    regras_por_cluster = {}
    
    # Criando dicionário aisle → departamento (pode haver múltiplos, então usamos set)
    aisle_para_departamento = (
        df_com_cluster[[coluna, 'department']]
        .drop_duplicates()
        .groupby(coluna)['department']
        .apply(lambda x: list(set(x)))
        .to_dict()
    )
    
    for i in range(3):  # clusters de 0 a 2
        df_cluster = df_com_cluster[df_com_cluster['cluster'] == i]
        if departments_filtros and i in departments_filtros:
            df_cluster = df_cluster[df_cluster['department'].isin(departments_filtros[i])]
        
        dummies_compras = gerar_dummies_transacoes(df_cluster, coluna)
        regras_cluster = gerar_regras_apriori(dummies_compras, metric='lift', min_support=min_support, min_threshold=min_threshold)

        # Mapeando departamentos para antecedents e consequents
        def mapear_departamentos(conjunto):
            return sorted(set(department for item in conjunto for department in aisle_para_departamento.get(item, [])))

        regras_cluster['antecedent_departments'] = regras_cluster['antecedents'].apply(mapear_departamentos)
        regras_cluster['consequent_departments'] = regras_cluster['consequents'].apply(mapear_departamentos)

        regras_por_cluster[f'regras_cluster_{i}'] = regras_cluster

    return regras_por_cluster['regras_cluster_0'], regras_por_cluster['regras_cluster_1'], \
           regras_por_cluster['regras_cluster_2']
    
def salvar_regras_apriori_xlsx(regras, caminho):
    """
    Salva um DataFrame de regras de associação no formato CSV.

    Parâmetros:
    -----------
    regras : pd.DataFrame
        DataFrame contendo as colunas 'antecedents' e 'consequents' como conjuntos (frozenset).
    caminho : str
        Caminho do arquivo CSV onde as regras serão salvas.
    """
    regras = regras.copy()
    
    # Convertendo conjuntos (frozensets) para strings
    regras['consequents'] = regras['consequents'].apply(lambda x: ', '.join(x))
    regras['antecedents'] = regras['antecedents'].apply(lambda x: ', '.join(x))
    
    # Convertendo listas de departamentos para strings
    regras['antecedent_departments'] = regras['antecedent_departments'].apply(lambda x: ', '.join(x))
    regras['consequent_departments'] = regras['consequent_departments'].apply(lambda x: ', '.join(x))

    regras.to_excel(caminho, index=False)
    

