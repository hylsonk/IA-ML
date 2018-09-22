import Machine as ml
import pandas as pandas
import sklearn as sk
import numpy as np

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
Y = data[['Class']]
Y = np.asarray(Y)

Xtrain, Xtest, Ytrain, Ytest = sk.model_selection.train_test_split(X, Y, test_size=0.20)

machine = ml.Machine(Xtrain,Ytrain)
machine.learning()