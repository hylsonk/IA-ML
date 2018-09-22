import numpy as np

'''
    Classe Machine que vai realizar o aprendizado
    file = o arquivo CSV com o dataset
'''
class Machine:

    def learning(self, Xtrain, Ytrain):
        n = Xtrain.shape[1]
        W = np.ones(shape=(n))
        n = len(Xtrain)
        x0 = np.ones(shape=(n, 1))
        x = np.concatenate((x0, Xtrain), axis=1)
        y = Ytrain
        print y
        print "Learning"

        go = True

        while go:
            go = False
            for i in range(n):
                h = np.dot(W.transpose(), x[i, :])

                if (np.sign(h) != np.sign(y[i])):
                    print 'Classificacao errada. ', np.sign(h), ' != ', np.sign(y[i])
                    go = True
                    print 'W antes de atualizar', W
                    W = W + np.dot(y[i], x[i, :])


    def work(self, Xtest, Ytest):
        print "Working"