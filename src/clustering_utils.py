import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def gerar_matriz_usuario(df):
    """
    Gera uma matriz de valores absolutos e proporcionais de número de compras de usuários por
    departamento de produtos.

    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame contendo pelo menos as colunas de usuário, departamento e produto,
        chamados respectivamente de 'user_id', 'department' e 'product_name'.

    Retorna:
    --------
    matriz_usuario : pd.DataFrame
        Matriz de contagem de produtos.
    matriz_usuario_prop : pd.DataFrame
        Matriz de proporções (cada linha soma 1).
    """
    # Contagem de produtos por usuário e categoria
    compras_usuario = df.groupby(['user_id', 'department'])['product_name'].count().reset_index()

    # Pivotar para matriz usuários x categorias
    matriz_usuario = compras_usuario.pivot_table(
        index='user_id',
        columns='department',
        values='product_name',
        fill_value=0
    )

    # Matriz proporcional (dividido pelo total por usuário)
    matriz_usuario_prop = matriz_usuario.div(matriz_usuario.sum(axis=1), axis=0) # Isto equivalente a uma normalização L1 da matriz_usuario, onde a norma L1 de cada linha é ajustada para 1.
    
    return matriz_usuario, matriz_usuario_prop

def calcular_silhouette_scores(df, cluster_range=range(2, 11), random_state=42):
    """
    Calcula os silhouette scores para GMM e K-means em uma faixa de clusters
    e retorna também um DataFrame comparativo.

    Parâmetros:
    ------------
    df : pd.DataFrame ou np.ndarray
        Matriz de dados para clustering.
    cluster_range : range
        Intervalo de número de clusters a testar (padrão range(2, 11)).
    random_state : int
        Semente para reprodutibilidade dos modelos (padrão 42).

    Retorna:
    --------
    tupla:
        (silhouette_scores_gmm, silhouette_scores_kmeans, df_scores)
    """
    silhouette_scores_gmm = []
    silhouette_scores_kmeans = []

    for n in cluster_range:
        # Gaussian Mixture
        gmm = GaussianMixture(n_components=n, random_state=random_state)
        gmm.fit(df)
        labels_gmm = gmm.predict(df)
        score_gmm = silhouette_score(df, labels_gmm)
        silhouette_scores_gmm.append(score_gmm)

        # K-means
        kmeans = KMeans(n_clusters=n, random_state=random_state)
        labels_kmeans = kmeans.fit_predict(df)
        score_kmeans = silhouette_score(df, labels_kmeans)
        silhouette_scores_kmeans.append(score_kmeans)

    # Criando DataFrame comparativo
    df_scores = pd.DataFrame({
        'n_clusters': list(cluster_range),
        'score_gmm': silhouette_scores_gmm,
        'score_kmeans': silhouette_scores_kmeans
    })

    return silhouette_scores_gmm, silhouette_scores_kmeans, df_scores

def calcular_db_index(df, cluster_range=range(2, 11), random_state=42):
    """
    Calcula os índices de Davies-Bouldin para GMM e K-means em uma faixa de clusters.

    Parâmetros:
    ------------
    df : pd.DataFrame ou np.ndarray
        Matriz de dados para clustering.
    cluster_range : range
        Intervalo de número de clusters a testar (padrão range(2, 11)).
    random_state : int
        Semente para reprodutibilidade dos modelos (padrão 42).

    Retorna:
    --------
    tupla:
        (db_index_gmm, db_index_kmeans, df_index)
    """
    db_index_gmm = []
    db_index_kmeans = []

    for n in cluster_range:
        # Gaussian Mixture
        gmm = GaussianMixture(n_components=n, random_state=random_state)
        gmm.fit(df)
        labels_gmm = gmm.predict(df)
        db_index_gmm.append(davies_bouldin_score(df, labels_gmm))

        # K-means
        kmeans = KMeans(n_clusters=n, random_state=random_state)
        labels_kmeans = kmeans.fit_predict(df)
        db_index_kmeans.append(davies_bouldin_score(df, labels_kmeans))
        
    # Criando DataFrame comparativo
    df_index = pd.DataFrame({
        'n_clusters': list(cluster_range),
        'index_gmm': db_index_gmm,
        'index_kmeans': db_index_kmeans
    })

    return db_index_gmm, db_index_kmeans, df_index

def plot_scores_index(cluster_range, silhouette_scores, db_index, modelo_nome='K-means'):
    """
    Plota os gráficos de Silhouette Score e Davies-Bouldin Index lado a lado.

    Parâmetros:
    ------------
    cluster_range : range, list ou tupla com (start, stop)
        Valores de número de clusters usados nos testes.
    silhouette_scores : list
        Lista de scores de silhouette correspondentes.
    db_index : list
        Lista de índices de Davies-Bouldin correspondentes.
    modelo_nome : str
        Nome do modelo para título dos gráficos ('K-means' ou 'GMM', por exemplo) (padrão K-means).
    """
    # Convertendo cluster_range se for uma tupla (start, stop)
    if isinstance(cluster_range, tuple) and len(cluster_range) == 2:
        cluster_range = list(range(cluster_range[0], cluster_range[1]))
    else:
        cluster_range = list(cluster_range)

    # Verificação de integridade
    if len(cluster_range) != len(silhouette_scores) or len(cluster_range) != len(db_index):
        raise ValueError(f"As listas devem ter o mesmo comprimento. "
                         f"cluster_range={len(cluster_range)}, "
                         f"silhouette_scores={len(silhouette_scores)}, "
                         f"db_index={len(db_index)}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot Silhouette
    axes[0].plot(cluster_range, silhouette_scores, marker='o')
    axes[0].set_title(f'Silhouette Score - {modelo_nome}')
    axes[0].set_xlabel('Número de Clusters')
    axes[0].set_ylabel('Silhouette Score')
    axes[0].grid(True)

    # Plot Davies-Bouldin
    axes[1].plot(cluster_range, db_index, marker='o')
    axes[1].set_title(f'Davies-Bouldin Index - {modelo_nome}')
    axes[1].set_xlabel('Número de Clusters')
    axes[1].set_ylabel('Davies-Bouldin Index')
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()
    
def plot_elbow_method(df, k_max=20, random_state=42):
    """
    Aplica o método do cotovelo (Elbow Method) para encontrar o número ideal de clusters usando KMeans.

    Parâmetros:
    -----------
    df : pd.DataFrame ou np.ndarray
        Dados de entrada para o KMeans (normalmente já pré-processados, como matriz de proporções).
    k_max : int
        Número máximo de clusters a testar (padrão 20).
    random_state : int
        Semente para reprodutibilidade do KMeans (padrão 42).
    """
    k_range = range(2, k_max + 1)
    inertia = []

    # Calcula a inércia (soma das distâncias quadradas aos centroides) para cada número de clusters
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        kmeans.fit(df)
        inertia.append(kmeans.inertia_)
    
    # Cálculo das diferenças de inércia
    dif_absoluta_entre_K = np.diff(inertia)  # diferença absoluta entre inércias consecutivas
    dif_relativa_entre_K = np.diff(inertia) / inertia[:-1]  # diferença relativa (percentual)

    # Plot da curva de Inércia
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertia, marker='o')
    plt.title('Método do Cotovelo (Elbow Method)')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Inércia')
    plt.xticks(k_range)
    plt.grid(True)
    plt.show()

    # Impressão das diferenças
    # ----------------------------------------------------------
    # Para ajudar na análise quantitativa do "cotovelo", este trecho exibe:
    # - A diferença absoluta de inércia entre K e K+1 (quanto a inércia caiu ao adicionar um cluster);
    # - A diferença relativa (percentual) dessa queda, em relação à inércia anterior.
    # Essas informações facilitam a identificação de onde o ganho marginal 
    # começa a diminuir significativamente (ponto de cotovelo).
    # ----------------------------------------------------------

    print('Diferença Absoluta e Relativa da Queda de Inércia entre número de clusters:')
    print('Ki → Kj : (diff_absoluta; diff_relativa)')
    for idx, k in enumerate(k_range[:-1]):  # até o penúltimo k
        print(f'{k} → {k+1}: ({dif_absoluta_entre_K[idx]:.4f}; {dif_relativa_entre_K[idx]:.4f})')

def kmeans(df, n_clusters, nome_coluna_cluster='cluster', random_state=42):
    """
    Aplica KMeans a um DataFrame e adiciona uma coluna de cluster.

    Parâmetros:
    -----------
    df : pd.DataFrame
        Dados de entrada para o KMeans (normalmente já pré-processados, como matriz de proporções).
    n_clusters : int
        Número de clusters a formar.
    nome_coluna_cluster : str
        Nome da coluna onde os rótulos de cluster serão armazenados (padrão 'cluster').
    random_state : int
        Semente para reprodutibilidade dos clusters (padrão 42).

    Retorna:
    --------
    data_clusterizado : pd.DataFrame
        Cópia do DataFrame original com a nova coluna de clusters adicionada.
    modelo_kmeans : sklearn.cluster.KMeans
        O modelo KMeans treinado.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(df)

    # Cria uma cópia para não alterar o original
    df_clusterizado = df.copy()
    df_clusterizado[nome_coluna_cluster] = labels

    return df_clusterizado, kmeans


def gerar_perfil_clusters(kmeans_model, df_com_cluster, coluna_cluster='cluster'):
    """
    Gera um DataFrame de perfil dos clusters a partir dos centróides do KMeans.

    Parâmetros:
    -----------
    kmeans_model : sklearn.cluster.KMeans
        Modelo KMeans já treinado.
    df_com_cluster : pd.DataFrame
        DataFrame original usado no treinamento, contendo a coluna de cluster (ou que será ignorada).
    coluna_cluster : str
        Nome da coluna de cluster a ser ignorada ao gerar o perfil (padrão 'cluster').

    Retorna:
    --------
    perfil_clusters : pd.DataFrame
        DataFrame transposto com departamentos (ou features) como linhas e clusters como colunas.
    """
    # Removendo a coluna de cluster se existir
    df_sem_cluster = df_com_cluster.drop(columns=coluna_cluster, errors='ignore')

    # Criando o DataFrame de centroides
    perfil_clusters = pd.DataFrame(
        kmeans_model.cluster_centers_,
        columns=df_sem_cluster.columns
    )

    # Removendo nome do índice de colunas
    perfil_clusters.columns.name = None

    # Transpondo para facilitar análise (features nas linhas)
    perfil_clusters = perfil_clusters.transpose()

    return perfil_clusters

def calc_dif_relativa(row):
    """
    Calcula a diferença relativa de cada valor em relação à média dos outros dois valores em uma linha.

    Para uma linha com três valores, a função retorna três diferenças relativas, uma para cada valor,
    calculada como:
        (valor - média_dos_outros_dois) / média_dos_outros_dois

    Parâmetros:
    -----------
    row : list, array-like ou pd.Series
        Linha contendo exatamente três valores numéricos.

    Retorna:
    --------
    pd.Series
        Série contendo três valores, cada um representando a diferença relativa do valor original
        em relação à média dos outros dois valores da mesma linha.
    """

    diffs = []

    for i in range(3):
        # Seleciona os dois valores que não são o valor atual (exclui o i-ésimo)
        outros = [row[j] for j in range(3) if j != i]

        # Calcula a média dos dois valores restantes
        media_outros = sum(outros) / 2

        # Calcula a diferença relativa do valor atual em relação à média dos outros dois
        # Se a média for zero, define a diferença como zero para evitar divisão por zero
        dif = (row[i] - media_outros) / media_outros if media_outros != 0 else 0

        # Armazena a diferença relativa na lista
        diffs.append(dif)

    return pd.Series(diffs, index=[0, 1, 2])    

def plot_proporcao_perfis_usuarios(matriz_usuario_prop_clusterizada, mapa_perfis, figsize=(12, 5)):
    """
    Plota a proporção de usuários em cada cluster com rótulos de perfil nomeados.

    Parâmetros:
    ------------
    matriz_usuario_prop_clusterizada : pd.DataFrame
        DataFrame contendo a coluna 'cluster' para os usuários.

    mapa_perfis : dict
        Dicionário mapeando os rótulos dos clusters para nomes de perfis compreensíveis.

    figsize : tuple
        Tamanho da figura do gráfico.
    """
    # Calculando a proporção de usuários por cluster
    proporcao_clusters = matriz_usuario_prop_clusterizada.groupby('cluster').size() / matriz_usuario_prop_clusterizada.shape[0]
    
    # Aplicando nomes dos perfis
    proporcao_clusters.index = proporcao_clusters.index.map(mapa_perfis)
    
    # Ordenando por proporção
    proporcao_clusters = proporcao_clusters.sort_values()

    # Plotando
    plt.figure(figsize=figsize)
    ax = proporcao_clusters.plot(kind='barh', color='skyblue')

    ax.set_xlabel('Proporção de Usuários', fontsize=13)
    ax.set_ylabel('Perfil de Cliente', fontsize=13)
    ax.set_title('Proporção de Usuários por Perfil de Cliente', fontsize=14)
    ax.tick_params(axis='both', labelsize=12) 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Adicionando os valores
    for i, (valor, nome) in enumerate(zip(proporcao_clusters, proporcao_clusters.index)):
        ax.text(valor + 0.002, i, f'{valor:.2%}', va='center', fontsize=11)
        
    # Removendo bordas superior e direita
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
def plot_clusters_pca_2d(df, cluster_col='cluster', sample_frac=0.07, cores=None,
                         caminho_salvar=None):
    """
    Plota uma visualização 2D dos clusters utilizando PCA, com amostragem dos dados e centróides projetados
    a partir do espaço original.

    Parâmetros:
    ------------
    df : pd.DataFrame
        DataFrame com as variáveis numéricas e uma coluna de cluster.
    cluster_col : str
        Nome da coluna que contém a atribuição de cluster (padrão 'cluster').
    sample_frac : float
        Fração dos pontos de cada cluster a serem amostrados para visualização (padrão 0,07).
    cores : list
        Lista de cores para cada cluster. Se None, será usada uma paleta padrão (padrão None).
    caminho_salvar : str ou None
        Caminho do arquivo para salvar a imagem (.png). Se None, a imagem não é salva (padrão None).

    Retorno:
    --------
    Nenhum. Apenas exibe (e opcionalmente salva) o gráfico.
    """

    X = df.drop(columns=cluster_col)
    y = df[cluster_col]

    # Padronização
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA para 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # DataFrame PCA
    df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    df_pca[cluster_col] = y.values

    # Amostragem por cluster
    df_amostrado = (
        df_pca.groupby(cluster_col, group_keys=False)
        .apply(lambda x: x.sample(frac=sample_frac, random_state=42))
    )

    # Cores padrão
    if cores is None:
        cores = ['limegreen', 'hotpink', 'deepskyblue']

    # Plotando
    plt.figure(figsize=(9, 7))
    for cluster_id in sorted(df_amostrado[cluster_col].unique()):
        subset = df_amostrado[df_amostrado[cluster_col] == cluster_id]
        nome_cluster = f'Cluster {cluster_id}'
        plt.scatter(
            subset['PC1'], subset['PC2'],
            s=25,
            color=cores[cluster_id % len(cores)],
            edgecolors='black',
            linewidth=0.4,
            label=nome_cluster,
            alpha=0.75
        )

    # Centróides no espaço original → padroniza → projeta com PCA
    centroides_originais = X.groupby(y).mean()
    centroides_padronizados = scaler.transform(centroides_originais)
    centroides_pca = pca.transform(centroides_padronizados)

    df_centroides = pd.DataFrame(centroides_pca, columns=['PC1', 'PC2'])
    df_centroides[cluster_col] = centroides_originais.index

    # Plotando centróides como X vazado em preto
    plt.scatter(
        df_centroides['PC1'], df_centroides['PC2'],
        s=15,
        facecolors='none',
        edgecolors='black',
        linewidths=2,
        marker='X',
        label='Centróides'
    )

    # Estética
    plt.title('Segmentação de Clientes - Clusters em 2D (PCA)', fontsize=14, weight='bold')
    plt.xlabel('Componente Principal 1', fontsize=12)
    plt.ylabel('Componente Principal 2', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(title='Clusters', fontsize=10, title_fontsize=11)
    plt.grid(False)

    ax = plt.gca()
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    if caminho_salvar:
        plt.savefig(caminho_salvar, dpi=300, bbox_inches='tight', facecolor='white')

    plt.show()
    
def salvar_clusters_csv(df_clusterizado, caminho_csv):
    """
    Salva um DataFrame contendo os clusters por usuário em um CSV.

    Parâmetros:
    ------------
    df_clusterizado : pd.DataFrame
        DataFrame contendo os clusters com 'user_id' no índice e uma coluna chamada 'cluster'.

    caminho_csv : str
        Caminho (relativo ou absoluto) para salvar o arquivo CSV.
    """
    df = df_clusterizado.reset_index()[['user_id', 'cluster']]
    df.to_csv(caminho_csv, index=False)