from Machine import *
import pandas as pandas
import numpy as np
from sklearn.model_selection import train_test_split #para embaralhar os dados aleatoriamente
from sklearn.metrics import accuracy_score #para medir nossa acuracia

file = 'IrisDataset.csv'

dataFile = pandas.read_csv(file, header=None)
data = pandas.DataFrame(dataFile)
# Nomeando as colunas
data.columns = ["PetalLength", "PetalWidth", "SepalLength", "SepalWidth", "Class"]

# Mapeando as classes para que as Strings sejam substituidas por valores numericos
mapping = {'Iris-setosa' : 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
data = data.replace({'Class':mapping})

X = data[['PetalLength', 'PetalWidth', 'SepalLength', 'SepalWidth']]
X = np.asarray(X)
Y = data['Class'].values
Y = np.asarray(Y)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.20, random_state=18)

# print Xtrain
ml = Machine()
ml.learning(Xtrain, Ytrain)
