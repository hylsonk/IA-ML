import Machine as ml
import pandas as pandas

file = 'IrisDataset.csv'

dataFile = pandas.read_csv(file, header=None)
data = pandas.DataFrame(dataFile)
# Nomeando as colunas
data.columns = ["PetalLength", "PetalWidth", "SepalLength", "SepalWidth", "Class"]

# Mapeando as classes para que as Strings sejam substituidas por valores numericos
mapping = {'Iris-setosa' : 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
data = data.replace({'Class':mapping})

machine = ml.Machine(data)
machine.learning()