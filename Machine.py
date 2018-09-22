import numpy as np

'''
    Classe Machine que vai realizar o aprendizado
    file = o arquivo CSV com o dataset
'''
class Machine:

    def __init__(self):
        self.W = None

    def learning(self, Xtrain, Ytrain):
        numCol = Xtrain.shape[1]
        self.W = np.ones(shape=(numCol+1))
        numLin = len(Ytrain)
        x0 = np.ones(shape=(numLin, 1))
        x = np.concatenate((x0, Xtrain), axis=1)
        y = Ytrain
        print y.shape
        print "Learning"

        # go = True
        print range(numLin)
        # while go:
        #     go = False
        for c in range(0, 1000000):
            for i in range(numLin):
                # print i
                h = np.dot(self.W.transpose(), x[i, :])

                # print y[i]
                # print x[i, :]

                if (np.sign(h) != np.sign(y[i])):
                    # print 'Classificacao errada. ', np.sign(h), ' != ', np.sign(y[i])
                    # go = True
                    # print 'W antes de atualizar', self.W
                    self.W = self.W + np.dot(y[i], x[i, :])
                    # print np.dot(y[i], x[i, :])
                    # print 'W atualizado:', self.W
                else:
                    print 'Classificacao correta', np.sign(h), '=', np.sign(y[i])


    def work(self, Xtest):
        numCol = Xtest.shape[1]
        self.W = np.ones(shape=(numCol + 1))
        numLin = len(Xtest)
        x0 = np.ones(shape=(numLin, 1))
        x = np.concatenate((x0, Xtest), axis=1)
        result = []
        for i in range(numLin):
            h = np.dot(self.W.transpose(), x[i, :])
            result.append(np.sign(h))

        return result