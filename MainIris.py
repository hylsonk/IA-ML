from Machine import *
import pandas as pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

file = 'IrisDataset.csv'

dataFile = pandas.read_csv(file, header=None)
data = pandas.DataFrame(dataFile)

print data

# Nomeando as colunas
data.columns = ["PetalLength", "PetalWidth", "SepalLength", "SepalWidth", "Class"]

# Mapeando as classes para que as Strings sejam substituidas por valores numericos
mapping = {'Iris-setosa' : -1, 'Iris-versicolor': -1, 'Iris-virginica': 1}
data = data.replace({'Class':mapping})


dataTrain, dataTest = train_test_split(data, test_size=0.20)


X = dataTrain[['PetalLength', 'PetalWidth', 'SepalLength', 'SepalWidth']]
Xtrain = np.asarray(X)
Y = dataTrain['Class'].values
Ytrain = np.asarray(Y)

# Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.20)

# print Xtrain
ml = Machine()
ml.learning(X, Y)

X = dataTest[['PetalLength', 'PetalWidth', 'SepalLength', 'SepalWidth']]
Xtest = np.asarray(X)
Y = dataTest['Class'].values
Ytest = np.asarray(Y)

result = ml.work(Xtest)
accuracy = accuracy_score(Ytest, result)

print result
print Ytest
print "precisao= ", accuracy
