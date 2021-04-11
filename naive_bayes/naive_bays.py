import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os


class NaiveBayes:
    def __init__( self, data = None, epsilon = 1e-7 ):
        self.database = pd.read_csv(data, index_col = 0)
        print(self.database.head())
        self.epsilon = epsilon


    #prepares the data in the naive bayes form with ter-frequency per class
    #dir_name: type string
    def data_prepare( self, dir_name ):
        files = os.listdir(dir_name)
        classes = [ name.rstrip('.txt') for name in files ]
        n_classes = len(classes)
        self.database = pd.DataFrame(np.zeros((0, len(classes))), columns = classes )
        for filename in files:
            temp = open( dir_name+'/'+filename, 'r' )
            list_of_words = temp.read().split(' ')
            for word in list_of_words:
                if word not in self.database.index:
                    self.database.loc[word] = np.zeros(n_classes)
                    self.database[filename.rstrip('.txt')][word] = 1
                else:
                    self.database[filename.rstrip('.txt')][word] += 1
            nwords = len(list_of_words)
            self.database[filename.rstrip('.txt')] = self.database[filename.rstrip('.txt')]/float(nwords)
            temp.close()
        self.database.to_csv("nbdata.csv")


    #query: type: string. should be the name of a file with the text.
    def query( self, filename, silent = True ):
        fileptr = open(filename)
        text = fileptr.read()
        words = text.split(' ')
        fileptr.close()
        #a prob inicial é 1 / o numero de classes
        words = [ word for word in words if word in self.database.index ]
        #probs_per_class = np.log( np.array( [1/len(self.database.columns)]*len(self.database.columns)))
        #como meu dataset está balanceado, PARA ESSE PROBLEMA, a probabilidade por classe será igual para todos, então usarei uma constante ao invés de um vetor.
        p_cj = 1.0/float(len(self.database.columns))
        frequencies = self.database.loc[ words ].apply( lambda row: row/np.sum(row)+self.epsilon, axis = 1 )
        probs = frequencies.apply( lambda classe: np.log(np.sum(classe))+p_cj,axis = 0 )
        probs = probs.apply(np.exp)
        if( not silent ):
            print( probs )

        return self.database.columns[np.argmax( probs )]


    #TODO: IDF to be implemented
    def IDF():
        pass


    #TODO: to be implemented
    def test_final( self, dir_name ):
        try:
            tests = os.listdir( dir_name )
        except:
            print("o diretorio não existe!")
            return

        number_right_answers = 0
        number_wrong_answers = 0
        for classe in tests:
            filenames = os.listdir(dir_name+'/'+classe+'/test/')
            for name in filenames:
                ans = self.query(dir_name+'/'+classe+'/test/'+name)
                if ans == classe:
                    number_right_answers += 1
                else:
                    #print(ans,"  ",classe)
                    number_wrong_answers += 1
            print("classe ", classe, ":")
            print("\tacertos: ", number_right_answers)
            print("\terros: ", number_wrong_answers, "\n\n")

        return float(number_right_answers) / float(number_right_answers + number_wrong_answers)


if __name__ == '__main__':
    #tests
    nb = NaiveBayes("nbdata.csv")
    print("acuracia = ", nb.test_final("dados_processados"))


