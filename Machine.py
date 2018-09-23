import numpy as np

'''
    Classe Machine que vai realizar o aprendizado
    
'''
class Machine:

    # Inicializa o W
    def __init__(self):
        self.W = None

    # Funcao de aprendizado
    def learning(self, Xtrain, Ytrain):

        # Pega o numero de colunas de Xtrain
        numCol = Xtrain.shape[1]

        # Inicializa W com um vetor de 1's
        self.W = np.ones(shape=(numCol+1))

        # Pega o numero de linha
        numLin = len(Ytrain)

        # Cria uma coluna com 1's e concatena com os valores de Xtrain
        x0 = np.ones(shape=(numLin, 1))
        x = np.concatenate((x0, Xtrain), axis=1)

        # atribui a y os valores de Ytrain
        y = Ytrain

        print "Learning"

        # Limita o aprendizado a 1000 hipoteses
        for c in range(0, 1000):

            # Testa as hipoteses ate encontrar algum erro e atualiza o W
            for i in range(numLin):
                h = np.dot(self.W.transpose(), x[i, :])

                # Caso a hipotese falhe atualiza W
                if np.sign(h) != np.sign(y[i]):
                    print 'W antes de atualizar', self.W
                    self.W = self.W + np.dot(y[i], x[i, :])
                    print 'W atualizado:', self.W
                # else:
                    # print 'Classificacao correta', np.sign(h), '=', np.sign(y[i])

    # Funcao de execucao - faz o Machine executar o que aprendeu
    def work(self, Xtest):
        # Pega o numero de linhas da amostra
        numLin = len(Xtest)

        # Cria uma coluna com 1's e concatena com os valores de Xtest
        x0 = np.ones(shape=(numLin, 1))
        x = np.concatenate((x0, Xtest), axis=1)

        # Inicializa o vetor de resultados
        result = []

        # Executa o que aprendeu
        for i in range(numLin):
            h = np.dot(self.W.transpose(), x[i, :])
            result.append(np.sign(h))

        # Retorna o resutado da aprendizagem
        return result