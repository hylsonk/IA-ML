from Machine import *
import pandas as pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


'''
    Esse programa tem como objetivo classificar flores Iris-setosa, Iris-versicolor e Iris-virginica em  usando como dado PetalLength, PetalWidth, SepalLength e SepalWidth
    
    OBS: Para conseguir classificar as flores foram realizadas duas execucoes de maquinas para classificar se sao Iris-virginica ou nao e se sao Iris-versicolor ou nao
    
'''

def getData():
    # Arquivo CSV de entrada
    file = 'IrisDataset.csv'
    
    # Usa o pandas para ler o arquivo CSV e gerar a matriz de dados
    dataFile = pandas.read_csv(file, header=None)
    data = pandas.DataFrame(dataFile)

    # Nomeando as colunas
    data.columns = ["PetalLength", "PetalWidth",
                    "SepalLength", "SepalWidth", "Class"]
    return data


# Funcao que vai chamar a Machine para fazer o apendizado e execucao do que a Machine aprendeu exibindo a sua precisao
def MLearning(mapping):

    data = getData()

    data = data.replace({'Class': mapping})

    # Separa aleatoriamente 20% para teste e 80% para que a maquina possa aprender
    dataTrain, dataTest = train_test_split(data, test_size=0.20)

    # Divide os dados que a maquina vai precisar comparar Y com os dados de entrada X para o aprendizado
    X = dataTrain[['PetalLength', 'PetalWidth', 'SepalLength', 'SepalWidth']]
    Xtrain = np.asarray(X)
    Y = dataTrain['Class'].values
    Ytrain = np.asarray(Y)

    # Declara e faz com que a maquina aprenda com os dados de treino
    ml = Machine()
    ml.learning(Xtrain, Ytrain)
    return ml.getW()

def work(WVirginica, WVersicolor):
    data = getData()

    mapping = {'Iris-setosa': 0, 'Iris-versicolor': -1, 'Iris-virginica': 1}

    data = data.replace({'Class': mapping})

    dataTrain,dataTest = train_test_split(data, test_size=0.30)

    # Divide os dados que a maquina vai trabalhar que nao foram passados no treino
    X = dataTest[['PetalLength', 'PetalWidth', 'SepalLength', 'SepalWidth']]
    Xtest = np.asarray(X)
    Y = dataTest['Class'].values
    Ytest = np.asarray(Y)

    ml = Machine()
    # Recebe o resultado encontrado dos dados passados para teste
    result = ml.workIris(Xtest, WVirginica, WVersicolor)

    # Compara os resultados com os dados que deveriam sair e gera uma precisao
    accuracy = accuracy_score(Ytest, result)

    # Imprime na tela a precisao dos resultados
    print "precisao  = ", accuracy


# Mapeando as classes para que as Strings sejam substituidas por valores numericos
mappingVirginica = {'Iris-setosa': -1, 'Iris-versicolor': -1, 'Iris-virginica': 1}

# Chama a funcao de aprendizado para classifica se sao Iris-Virginica ou nao
WVirginica = MLearning(mappingVirginica)

# Mapeando as classes para que as Strings sejam substituidas por valores numericos
mappingVersicolor = {'Iris-setosa': -1, 'Iris-versicolor': 1, 'Iris-virginica': -1}

# Chama a funcao de aprendizado para classifica se sao Iris-Versicolor ou nao
WVersicolor = MLearning(mappingVersicolor)

work(WVirginica,WVersicolor)


