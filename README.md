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
