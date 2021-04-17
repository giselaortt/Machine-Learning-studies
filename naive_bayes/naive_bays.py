import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import re

class NaiveBayes:
    def __init__( self, data = None, epsilon = 1e-7 ):
        self.epsilon = epsilon
        if( '.csv' in data ):
            self.database = pd.read_csv(data, index_col = 0)
            print(self.database.head())
        else:
            self.data_prepare(data)


    #prepares the data in the naive bayes form with term-frequency per class
    #dir_name: type string
    def data_prepare( self, dir_name ):
        try:
            files = os.listdir(dir_name)
        except:
            print('o repositório não existe! cheque o nome e tente novamente')
            return
        #classes = [ name.rstrip('.txt') for name in files ]
        classes = [ re.sub('\.txt$', '', name) for name in files ]
        n_classes = len(classes)
        self.database = pd.DataFrame(np.zeros((0, len(classes))), columns = classes )
        for filename in files:
            temp = open( dir_name+'/'+filename, 'r' )
            list_of_words = temp.read().split(' ')
            for word in list_of_words:
                if word not in self.database.index:
                    self.database.loc[word] = np.zeros(n_classes)
                    self.database[re.sub('\.txt$', '', filename)][word] = 1
                else:
                    self.database[re.sub('\.txt$', '', filename)][word] += 1
            nwords = len(list_of_words)
            self.database[ re.sub('\.txt$', '', filename) ] = self.database[re.sub('\.txt$', '', filename)]/float(nwords)
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


    #modifiquei essa função para refazer os testes apenas em um subconjunto das classes.
    # class_names: should be a list of the classes to be tested
    # verbose: if true, this method will print more information about the results
    def test_final( self, dir_name, class_names = None, verbose = False ):
        try:
            tests = os.listdir( dir_name )
        except:
            print("o diretorio não existe!")
            return
        number_right_answers = 0
        number_wrong_answers = 0
        for classe in self.database.classes:
            if( class_names is None or classe in class_names):
                results = pd.Series( data=np.zeros( self.database.shape[1] ), index=self.database.columns )
                class_right = 0
                class_wrong = 0
                filenames = os.listdir(dir_name+'/'+classe+'/test/')
                for name in filenames:
                    ans = self.query(dir_name+'/'+classe+'/test/'+name)
                    results[ans]+=1
                    if ans == classe:
                        number_right_answers += 1
                        class_right += 1
                    else:
                        #print(ans,"  ",classe)
                        number_wrong_answers += 1
                        class_wrong += 1
                print("classe ", classe, ":")
                print("\tacertos: ", class_right)
                print("\terros: ", class_wrong)
                print("\tacurácia da classe:", float(class_right)/float(class_right+class_wrong), "\n\n")
                print("predicted values:\n", results, '\n\n')

        return float(number_right_answers) / float(number_right_answers + number_wrong_answers)


if __name__ == '__main__':
    #tests
    #nb = NaiveBayes("nbdata.csv")
    #print("acuracia = ", nb.test_final("dados_processados"))
    #classes que tiveram acerto abaixo de 5% (resultado minimo esperado) 
    nb=NaiveBayes('dados_concatenados/train')
    #nb=NaiveBayes('nbdata.csv')
    #problematicos = ['sci.crypt', 'comp.windows.x', 'talk.politics.mideast']
    print( nb.test_final("dados_processados"))




