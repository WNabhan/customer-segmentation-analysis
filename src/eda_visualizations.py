import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def plot_count_by_column(
    df,
    coluna,
    titulo='',
    xlabel='',
    ylabel='Número de Ocorrências',
    ordem=None,
    figsize=(10, 5),
    annotate=True,
    fontsize_titulo=12,
    fontsize_eixos=10,
    fontsize_xticks=9,
    fontsize_yticks=9,
    fontsize_annotate=9,
    rotation_xticks=0
):
    """
    Plota um gráfico de contagem para uma coluna categórica de um DataFrame.

    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame com os dados já processados (sem duplicação, por ex.).
    coluna : str
        Nome da coluna a ser usada no eixo X.
    titulo : str
        Título do gráfico.
    xlabel : str
        Rótulo do eixo X.
    ylabel : str
        Rótulo do eixo Y.
    ordem : list
        Ordem dos rótulos no eixo X (opcional).
    figsize : tuple
        Tamanho da figura.
    annotate : bool
        Se True, adiciona os valores em cima das barras.
    fontsize_titulo : int
        Tamanho da fonte do título.
    fontsize_eixos : int
        Tamanho da fonte dos rótulos dos eixos.
    fontsize_xticks : int
        Tamanho da fonte dos rótulos do eixo X.
    fontsize_yticks : int
        Tamanho da fonte dos rótulos do eixo Y.
    fontsize_annotate : int
        Tamanho da fonte dos valores anotados nas barras.
    rotation_xticks : int
        Rotação dos rótulos no eixo X.
    """
    plt.figure(figsize=figsize)
    ax = sns.countplot(data=df, x=coluna, order=ordem)
    
    # Colocando as barras sobre a grade
    for bar in ax.patches:
        bar.set_zorder(2)
        
    plt.title(titulo, fontsize=fontsize_titulo)
    plt.xlabel(xlabel or coluna, fontsize=fontsize_eixos)
    plt.ylabel(ylabel, fontsize=fontsize_eixos)
    plt.xticks(rotation=rotation_xticks, fontsize=fontsize_xticks)
    plt.yticks(fontsize=fontsize_yticks)

    # Removendo bordas superior e direita (limpando o visual)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Adicionando grade horizontal (deixando visualmente mais explicativo)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    if annotate:
        for p in ax.patches:
            altura = p.get_height()
            ax.annotate(f'{int(altura)}', (p.get_x() + p.get_width() / 2., altura),
                        ha='center', va='bottom', fontsize=fontsize_annotate)

    plt.tight_layout()
    plt.show()


def plot_media_por_grupo(
    serie,
    titulo='',
    xlabel='',
    ylabel='',
    figsize=(20, 10),
    color='dodgerblue',
    rotation_xticks=90,
    annotate=False,
    fontsize_titulo=20,
    fontsize_eixos=18,
    fontsize_xticks=14,
    fontsize_yticks=16,
    fontsize_annotate=14
):
    """
    Plota um gráfico de barras a partir de uma série indexada com médias por grupo.

    Parâmetros:
    -----------
    serie : pd.Series
        Série com o índice representando os grupos e os valores como médias.
    titulo : str
        Título do gráfico.
    xlabel : str
        Rótulo do eixo X.
    ylabel : str
        Rótulo do eixo Y.
    figsize : tuple
        Tamanho da figura.
    color : str
        Cor das barras.
    rotation_xticks : int
        Rotação dos rótulos no eixo X.
    annotate : bool
        Se True, mostra os valores das barras como rótulos.
    fontsize_titulo : int
        Tamanho da fonte do título.
    fontsize_eixos : int
        Tamanho da fonte dos rótulos dos eixos.
    fontsize_xticks : int
        Tamanho da fonte dos rótulos do eixo X.
    fontsize_yticks : int
        Tamanho da fonte dos rótulos do eixo Y.
    fontsize_annotate : int
        Tamanho da fonte das anotações sobre as barras.
    """
    plt.figure(figsize=figsize)
    ax = sns.barplot(x=serie.index, y=serie.values, color=color)
    plt.title(titulo, fontsize=fontsize_titulo)
    plt.xlabel(xlabel, fontsize=fontsize_eixos)
    plt.ylabel(ylabel, fontsize=fontsize_eixos)
    plt.xticks(rotation=rotation_xticks, fontsize=fontsize_xticks)
    plt.yticks(fontsize=fontsize_yticks)
    plt.grid(axis='y')

    if annotate:
        for i, v in enumerate(serie.values):
            ax.text(i, v + max(serie.values)*0.01, f'{v:.2f}',
                    ha='center', va='bottom', fontsize=fontsize_annotate)

    plt.tight_layout()
    plt.show()


def plot_heatmap_pedidos(
    df,
    coluna_filtro=None,
    valor_filtro=None,
    titulo='Distribuição de Pedidos Únicos por Hora e Dia da Semana',
    label_barra=None,
    figsize=(14, 6),
    cmap='rocket'
):
    """
    Plota um heatmap da quantidade de pedidos únicos por hora e dia da semana.
    
    Parâmetros:
    ------------
    df : pd.DataFrame
        DataFrame com os dados já processados (sem duplicação, por ex.).
    coluna_filtro : str ou None
        Nome da coluna a ser filtrada (como 'department' ou 'aisle'). Se None, não aplica filtro.
    valor_filtro : str ou None
        Valor que será usado no filtro da coluna especificada.
    titulo : str
        Título do gráfico.
    label_barra : str ou None
        Texto da legenda da barra de cores.
    figsize : tuple
        Tamanho da figura.
    cmap : str
        Paleta de cores do heatmap (padrão: 'rocket').
    """
    
    dias_ordenados = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']

    # Aplicando filtro, se especificado
    if coluna_filtro and valor_filtro:
        df = df[df[coluna_filtro] == valor_filtro].copy()

    # Garantindo que os dias da semana estão ordenados corretamente
    df['dia_da_semana'] = pd.Categorical(
        df['dia_da_semana'],
        categories=dias_ordenados,
        ordered=True
    )

    # Agrupando por dia da semana e hora do pedido
    heatmap_data = (
        df.groupby(['dia_da_semana', 'order_hour_of_day'])['order_id']
        .count()
        .unstack(fill_value=0)
    )

    # Ordenando as colunas de hora
    heatmap_data = heatmap_data[sorted(heatmap_data.columns)]

    # Plot
    plt.figure(figsize=figsize)
    sns.heatmap(
        heatmap_data,
        cmap=cmap,
        annot=True,
        fmt='d',
        linewidths=0.5,
        cbar_kws={'label': label_barra or 'Número de Pedidos Únicos'}
    )
    plt.title(titulo)
    plt.xlabel('Hora do Dia')
    plt.ylabel('Dia da Semana')
    plt.tight_layout()
    plt.show()

def plot_top_populares(
    df,
    coluna,
    top_n=None,
    titulo=None,
    nome_col_titulo=None,
    figsize=(20, 12),
    xlabel='Número de vezes comprado',
    ylabel=None,
    color='dodgerblue',
    fontsize_titulo=20,
    fontsize_eixos=17,
    fontsize_xticks=15,
    fontsize_yticks=15,
    fontsize_annotate=14,
):
    """
    Plota um gráfico horizontal com os N valores mais frequentes de uma coluna.

    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame com os dados.
    coluna : str
        Nome da coluna para contagem de frequência.
    top_n : int
        Número de itens mais frequentes a serem exibidos. Se None, será gerado com todos os valores.
    titulo : str ou None
        Título do gráfico. Se None, será gerado automaticamente.
    nome_col_titulo: str
        Nome da coluna selecionada que aparecerá no título do gráfico.
    figsize : tuple
        Tamanho da figura.
    xlabel : str
        Rótulo do eixo X.
    ylabel : str ou None
        Rótulo do eixo Y. Se None, usa o nome da coluna.
    color : str
        Paleta de cores do Seaborn (ex: 'rocket', 'viridis', 'Blues_d').
    fontsize_titulo : int
        Tamanho da fonte do título.
    fontsize_eixos : int
        Tamanho da fonte dos rótulos dos eixos.
    fontsize_yticks : int
        Tamanho da fonte dos rótulos do eixo Y.
    fontsize_xticks : int
        Tamanho da fonte dos rótulos do eixo X.
    fontsize_annotate : int
        Tamanho da fonte das anotações nas barras.
    """
    # Conta as ocorrências dos valores únicos na coluna especificada
    # Seleciona apenas os top N mais frequentes
    contagem = df[coluna].value_counts().head(top_n)

    # Renomeando 'coluna' para os nomes dos eixos
    if coluna == 'product_name':
        coluna = 'produtos'
    elif coluna == 'aisle':
        coluna = 'seções'
    elif coluna == 'department':
        coluna = 'departamentos'

    # Plot
    plt.figure(figsize=figsize)

    ax = sns.barplot(
        x=contagem.values,
        y=contagem.index,
        color=color,
        orient='h',
    )

    plt.title(titulo or f'Top {top_n} {nome_col_titulo} mais populares', fontsize=fontsize_titulo)
    plt.xlabel(xlabel, fontsize=fontsize_eixos)
    plt.ylabel(ylabel or coluna.capitalize(), fontsize=fontsize_eixos)
    plt.xticks(fontsize=fontsize_xticks)
    plt.yticks(fontsize=fontsize_yticks)
    
    # Removendo bordas superior e direita do gráfico
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Adicionando os valores absolutos ao lado de cada barra
    for i, v in enumerate(contagem.values):
        ax.text(v + 500 , i, str(v), va='center', fontsize=fontsize_annotate)

    plt.tight_layout()
    plt.show()

def plot_top_populares_2(
    df,
    coluna_referencia,
    coluna_detalhe,
    top_n,
    filtro_coluna_referencia = None,
    titulo=None,
    figsize=(20, 12),
    xlabel='Número de vezes comprado',
    color='mediumseagreen',
    fontsize_titulo=20,
    fontsize_eixos=17,
    fontsize_xticks=15,
    fontsize_yticks=15,
    fontsize_annotate=14,
):
    """
    Plota os valores mais frequentes da coluna desejada dentro de um valor específico da coluna de referência.

    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame com os dados.
    coluna_referencia : str
        Coluna de categoria (ex: 'department', 'aisle').
    filtro_coluna_referencia : str ou int
        Valor específico da coluna agrupadora que será filtrado.
    coluna_detalhe : str
        Coluna de itens detalhados (ex: 'aisle', 'product_name').
    top_n : int
        Número de itens mais frequentes a serem exibidos.
    titulo : str ou None
        Título do gráfico. Se None, será gerado automaticamente.
    figsize : tuple
        Tamanho da figura.
    xlabel : str
        Rótulo do eixo X.
    color : str
        Cor das barras.
    fontsize_titulo : int
        Tamanho da fonte do título.
    fontsize_eixos : int
        Tamanho da fonte dos rótulos dos eixos.
    fontsize_yticks : int
        Tamanho da fonte dos rótulos do eixo Y.
    fontsize_xticks : int
        Tamanho da fonte dos rótulos do eixo X.
    fontsize_annotate : int
        Tamanho da fonte das anotações nas barras.
    """
    
    # Verificando se foi passado um valor para filtrar na coluna de referência
    if filtro_coluna_referencia is not None:
        # Filtrando o DataFrame com base no valor da coluna de referência
        df_filtrado = df[df[coluna_referencia] == filtro_coluna_referencia]
    else:
        # Se não foi passado valor para filtrar, emite mensagem e encerra a função
        print('Insira o nome da categoria de filtragem da coluna de referência.')
        return

    # Contando as ocorrências dos valores na coluna de detalhe, considerando apenas os top N mais frequentes
    contagem = df_filtrado[coluna_detalhe].value_counts().head(top_n)

    # Criando a figura do gráfico
    plt.figure(figsize=figsize)

    # Plot
    ax = sns.barplot(
        x=contagem.values,
        y=contagem.index,
        color=color,
        orient='h'
    )

    # Renomeando 'coluna_detalhe' e 'coluna_referencia' para os nomes dos eixos e titulo
    if coluna_detalhe == 'product_name':
        coluna_detalhe = 'produtos'
    elif coluna_detalhe == 'aisle':
        coluna_detalhe = 'seções'
        
    if coluna_referencia == 'department':
        coluna_referencia = 'departamento'
    elif coluna_referencia == 'aisle':
        coluna_referencia = 'seção'
        
    if not titulo:
        # Definindo o título do gráfico de forma dinâmica, com base nos filtros e parâmetros
        titulo = f"Top {top_n} {coluna_detalhe} mais populares em {coluna_referencia} = '{filtro_coluna_referencia}'"
        
    plt.title(titulo, fontsize=fontsize_titulo)
    plt.xlabel(xlabel, fontsize=fontsize_eixos)
    plt.ylabel(coluna_detalhe.capitalize(), fontsize=fontsize_eixos)
    plt.xticks(fontsize=fontsize_xticks)
    plt.yticks(fontsize=fontsize_yticks)

    # Removendo as bordas superior e direita do gráfico
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Adicionando os valores absolutos ao lado de cada barra
    for i, v in enumerate(contagem.values):
        ax.text(v + max(contagem.values) * 0.01, i, str(v), va='center', fontsize=fontsize_annotate)

    plt.tight_layout()
    plt.show()
    
def plot_histograma_com_media(
    serie,
    titulo='Distribuição',
    figsize=(10, 6),
    xlabel='Valor',
    ylabel='Frequência',
    bins=30,
    filtro_max=None,
    color_linha='red',
    kde=False,
    fontsize_titulo=14,
    fontsize_eixos=12,
    fontsize_xticks=10,
    fontsize_yticks=10,
    fontsize_legenda=10
):
    """
    Plota um histograma de uma série numérica com uma linha indicativa da média.

    Parâmetros:
    -----------
    serie : pd.Series 
        Conjunto de dados numéricos a serem plotados.
    titulo : str
        Título do gráfico.
    figsize : tuple
        Tamanho da figura em polegadas (largura, altura). 
    xlabel : str
        Rótulo do eixo X. 
    ylabel : str
        Rótulo do eixo Y. 
    bins : int
        Número de divisões (barras) no histograma (padrão 30). 
    filtro_max : float
        Valor máximo para filtrar os dados antes de plotar (exclui valores maiores).
    color_linha : str
        Cor da linha da média. 
    kde : bool
        Se True, plota a estimativa de densidade kernel (KDE). 
    fontsize_titulo : int
        Tamanho da fonte do título. 
    fontsize_eixos : int
        Tamanho da fonte dos rótulos dos eixos. 
    fontsize_xticks : int
        Tamanho da fonte dos valores do eixo X.
    fontsize_yticks : int
        Tamanho da fonte dos valores do eixo Y.
    fontsize_legenda : int
        Tamanho da fonte da legenda. 
    """

    # Aplicando filtro de valor máximo, se fornecido
    if filtro_max is not None:
        serie = serie[serie < filtro_max]

    # Calculando a média da série filtrada, para uso posterior na linha indicativa
    media = serie.mean()

    # Criando uma nova figura com o tamanho especificado
    plt.figure(figsize=figsize)

    # Plota o histograma da série com a quantidade de bins desejada
    ax = sns.histplot(serie, bins=bins, kde=kde)

    # Colocando as barras sobre a grade
    for patch in ax.patches:
        patch.set_zorder(2)

    # Inserindo a linha da média
    ax.axvline(media, color=color_linha, linestyle='--', linewidth=2, label=f'Média = {media:.2f}', zorder=3)

    # Adicionando grade horizontal (deixando visualmente mais explicativo)
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

    ax.set_title(titulo, fontsize=fontsize_titulo)
    ax.set_xlabel(xlabel, fontsize=fontsize_eixos)
    ax.set_ylabel(ylabel, fontsize=fontsize_eixos)
    ax.tick_params(axis='x', labelsize=fontsize_xticks)
    ax.tick_params(axis='y', labelsize=fontsize_yticks)
    ax.legend(fontsize=fontsize_legenda)

    plt.tight_layout()
    plt.show()
    

def plot_dispersao_interativo(
    df,
    eixo_x,
    eixo_y,
    titulo=None,
    xlabel=None,
    ylabel=None,
    hover_data=None,
    tamanho_ponto=8,
    opacidade=0.6
):
    """
    Plota um gráfico de dispersão (scatter) interativo entre duas métricas, com visual clean.

    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame contendo as métricas.
    eixo_x : str
        Nome da coluna no eixo X.
    eixo_y : str
        Nome da coluna no eixo Y.
    titulo : str ou None
        Título do gráfico.
    xlabel : str ou None
        Rótulo do eixo X (usa o nome da coluna se None).
    ylabel : str ou None
        Rótulo do eixo Y (usa o nome da coluna se None).
    hover_data : list ou None
        Lista de colunas extras para aparecerem ao passar o mouse (respeita a ordem fornecida).
    tamanho_ponto : int
        Tamanho dos pontos (padrão 8).
    opacidade : float
        Opacidade dos pontos (padrão 0.6).
    """
    fig = px.scatter(
        df,
        x=eixo_x,
        y=eixo_y,
        hover_data=hover_data,
        labels={
            eixo_x: xlabel or eixo_x,
            eixo_y: ylabel or eixo_y
        },
        title=titulo
    )

    fig.update_traces(
        marker=dict(size=tamanho_ponto, opacity=opacidade, color='deepskyblue'),
        selector=dict(mode='markers')
    )
    
    fig.update_layout(
        plot_bgcolor='white',
        title_x=0.5,
        hoverlabel=dict(bgcolor="white", font_size=12),
        xaxis=dict(
            gridcolor='lightgray',
            zeroline=False,
            title=xlabel or eixo_x,
            tick0=0,         # Começa no 0
            dtick=50000      # Marcação a cada 50.000
        ),
        yaxis=dict(
            gridcolor='lightgray',
            zeroline=False,
            title=ylabel or eixo_y
        ),
        font=dict(size=13),
    )

    fig.show()

