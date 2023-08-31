# Documentação do Projeto: Machine Learning para indicações de filmes com Python

## Introdução
Este notebook contém um projeto de Machine Learning aprendendo a introdução a algoritmos não supervisonados. O objetivo deste projeto é catalogar e recomendar filmes de acordo com seu gênero. Usando a linguagem Python e suas bibliotecas.

## Requisitos
Certifique-se de ter as seguintes bibliotecas instaladas em seu ambiente Python:
* pandas
* sklearn
* matplotlib
* seaborn
* scipy

Você pode instalá-las utilizando o seguinte comando:
```pip install pandas skelearn matplotlib seaborn scipy```

## Conjuto de Dados
O conjuto de dados utilizados neste projeto é o "movies.csv", que contém informações de filmes. O conjunto de dados possui as seguintes colunas:
* movieId: Id do filme
* title: nome do filme e seu ano
* genres: os gêneros que o filme faz parte

## Etapas do Projeto
O projeto é dividido nas seguintes etapas:

1. <b>Carregamento dos Dados</b>: Carregar o conjunto de dados a partir do arquivo CSV usando a biblioteca pandas e mudando o nome das colunas.
2. <b>Tratamento de Dados</b>: Divide os gêneros de filmes em colunas e atribui um valor binário e escala os gêneros em relevância.
3. <b>Carregamento do código</b>: Usa os dados para a criação de um modelo que gere gráficos para a melhor compreensão dos dados.
4. <b>Conclusões</b>: Mostrar os gráficos para o usuário de acordo com a sua escolha.

## Implementação

### 1. Carregamento dos Dados
Utilizando a biblioteca pandas para carregar o conjunto de dados a partir do arquivo CSV e mudar o nome das suas colunas:

```
filmes = pd.read_csv('movies.csv')
filmes.columns = ['filme_id', 'titulo', 'generos']
```

### 2. Tratamento de Dados:
Divide os gêneros de filmes em colunas e atribui um valor binário e escala os gêneros em relevância:
```
generos = filmes.generos.str.get_dummies()
dados_dos_filmes = pd.concat([filmes, generos], axis=1)
scaler = StandardScaler()
generos_escalados = scaler.fit_transform(generos)
```

### 3. Carregamento do código:
Usa os dados para a criação de um modelo que gere gráficos para a melhor compreensão dos dados:
```
def kmeans(numero_de_clusters, generos):
    modelo = KMeans(n_clusters=numero_de_clusters, n_init=10)
    modelo.fit(generos)
    return [numero_de_clusters, modelo.inertia_]
```

```
def grafico_inertia():
    print("\nIsso irá demorar um minuto!")
    resultado = [kmeans(numero_de_grupos, generos_escalados) for numero_de_grupos in range(1, 41)]
    resultados = pd.DataFrame(resultado, columns=['grupos', 'inertia'])
    resultados.inertia.plot(xticks=resultados.grupos)

    plt.show()
```

```
def grafico_grupos():
    print("\nIsso irá demorar um minuto!")
    grupos = pd.DataFrame(modelo.cluster_centers_, columns=generos.columns)
    grupos_transposed = grupos.transpose()
    grupos_transposed.plot.bar(subplots=True, figsize=(25, 50), sharex=False,
                               rot=0)
    plt.show()

```

```
def grafico_dimensional():
    print("\nIsso irá demorar um minuto!")
    tsne = TSNE()
    visualizacao = tsne.fit_transform(generos_escalados)
    sns.scatterplot(x=visualizacao[:, 0], y=visualizacao[:, 1],
                    hue=modelo.labels_,
                    palette=sns.color_palette('Set1', grupos))
    plt.show()
```

```
def dedograma():
    grupos = pd.DataFrame(modelo.cluster_centers_, columns=generos.columns)
    matriz_de_distancia = linkage(grupos)
    dendrogram(matriz_de_distancia)
    plt.show()
```

```
def filtragem_grupos():
    x = int(input('Qual grupo você deseja ver: '))
    linhas = int(input('Quantidade de linhas: '))
    grupo = x
    filtro = modelo.labels_ == grupo
    return dados_dos_filmes[filtro].head(linhas)
```
