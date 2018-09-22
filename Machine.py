import numpy as np

'''
    Classe Machine que vai realizar o aprendizado
    file = o arquivo CSV com o dataset
'''
class Machine:
    def __init__(self, file):
        self.file = file

    def learning(self):
        csv = np.genfromtxt(self.file, delimiter=",")
        n = len(csv)
        x0 = np.ones(shape=(n,1))
        x = np.concatenate((x0,csv[:,:4]),axis=1)
        y = csv[:,4]
        print y

