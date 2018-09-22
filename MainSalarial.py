from Machine import *
import pandas as pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

file = "salaries.csv"

dataFile = pandas.read_csv(file, delimiter=";", header=None)
data = pandas.DataFrame(dataFile)

# Nomeando as colunas
data.columns = ["sx", "rk", "yr", "dg", "yd", "sl"]

# Mapeando as classes para que as Strings sejam substituidas por valores numericos
mapping = {'   male ' : -1, ' female ' : 1}
data = data.replace({"sx": mapping})


mapping = {'assistant ': 1, 'associate ': 2, '     full ' : 3}
data = data.replace({'rk':mapping})

mapping = {'  masters ': 0, 'doctorate ': 1}
data = data.replace({'dg':mapping})

dataTrain, dataTest = train_test_split(data, test_size=0.20)

X = dataTrain[['rk', 'yr', 'dg', 'yd', 'sl']]
Xtrain = np.asarray(X)
Y = dataTrain['sx'].values
Ytrain = np.asarray(Y)

ml = Machine()
ml.learning(Xtrain, Ytrain)

X = dataTest[['rk', 'yr', 'dg', 'yd', 'sl']]
Xtest = np.asarray(X)
Y = dataTest['sx'].values
Ytest = np.asarray(Y)

result = ml.work(Xtest)
accuracy = accuracy_score(Ytest, result)

print result
print Ytest
print "precisao= ", accuracy