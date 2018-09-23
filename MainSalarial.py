from Machine import *
import pandas as pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

'''
    Esse programa tem como objetivo descobrir o sexo da pessoa em  usando como dados o seu Rank, Anos no atual rank, Grau de formacao, Numero de anos trabalho no seu grau e seu salario

    OBS: Para conseguir classificar as pessoas foi a maquina foi executada apenas 1 vez, pois a classificacao foi de Homem ou Mulher

'''

# Arquivo CSV de entrada
file = "salaries.csv"

# Usa o pandas para ler o arquivo CSV e gerar a matriz de dados
dataFile = pandas.read_csv(file, delimiter=";", header=None)
data = pandas.DataFrame(dataFile)

# Nomeando as colunas
data.columns = ["sx", "rk", "yr", "dg", "yd", "sl"]

# Mapeando as classes para que as Strings sejam substituidas por valores numericos na coluna sx
mapping = {'   male ' : -1, ' female ' : 1}
data = data.replace({"sx": mapping})

# Mapeando as classes para que as Strings sejam substituidas por valores numericos na coluna rk
mapping = {'assistant ': 1, 'associate ': 2, '     full ' : 3}
data = data.replace({'rk':mapping})

# Mapeando as classes para que as Strings sejam substituidas por valores numericos na coluna dg
mapping = {'  masters ': 0, 'doctorate ': 1}
data = data.replace({'dg':mapping})

# Separa aleatoriamente 20% para teste e 80% para que a maquina possa aprender
dataTrain, dataTest = train_test_split(data, test_size=0.20)

# Divide os dados que a maquina vai precisar comparar Y com os dados de entrada X para o aprendizado
X = dataTrain[['rk', 'yr', 'dg', 'yd', 'sl']]
Xtrain = np.asarray(X)
Y = dataTrain['sx'].values
Ytrain = np.asarray(Y)

# Declara e faz com que a maquina aprenda com os dados de treino
ml = Machine()
ml.learning(Xtrain, Ytrain)

# Divide os dados que a maquina vai trabalhar que nao foram passados no treino
X = dataTest[['rk', 'yr', 'dg', 'yd', 'sl']]
Xtest = np.asarray(X)
Y = dataTest['sx'].values
Ytest = np.asarray(Y)

# Recebe o resultado encontrado dos dados passados para teste
result = ml.work(Xtest)

# Compara os resultados com os dados que deveriam sair e gera uma precisao
accuracy = accuracy_score(Ytest, result)

# Imprime na tela a precisao dos resultados
print "precisao= ", accuracy