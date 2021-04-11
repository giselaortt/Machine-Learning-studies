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
    def query( self, filename ):
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
        print( probs )
        return self.database.columns[np.argmax( probs )]

#        for classe, i in zip(self.database.columns, range(len(self.database.columns)) ):
#            for word in words:
#                #usar log para não perder a precisão quando multiplicar numeros muito pequenos
#                if word in self.database.index :
#                    probs[i] = probs[i] + np.log(self.database[classe][word])
#                else:
#                    pass

        #aplicar exponencial
#        probs = np.exp(probs)
        #mostrar probabilidades
#        for classe, prob in zip(self.database.columns, probs):
#            print( classe, ": ", prob )

#        for instance in self.classes:
#            freq_i = len( self.database.loc[ self.database.iloc[:,self.ycolumn] == instance ] )
#            prob_class_i = freq_i/ self.database.shape[0]
#            for i, condition in zip(range(self.database.shape[1]),query):
#                if condition is not None:
#                  prob_class_i = prob_class_i*(len( self.database.loc[ (self.database.iloc[:,self.ycolumn]==instance)&(self.database.iloc[:,i] == condition) ])/freq_i )
#            probs.append( prob_class_i )
#        soma = sum( probs )
#        probs = [ (float)(i) / soma for i in probs]
#

        #TODO: print probabilities for all the classes insted of just showing the biggest one


#    #should return a dataframe with all the words present on the document and its normalized freequency
#    def term_frequency( text ):
#        list_of_words = text.split(' ')
#        words_unique = list(set(list_of_words))
#        
#        term_frequency = dict.fromkeys( words_unique, 0 )
#        total = 0
#        for word in list_of_words:
#            term_frequency[words] += 1
#            total += 1
#        for word, repetitions in term_frequency.items():
#            term_frequency[word] = repetitions / total
#
#        #TODO: passar para dataframe
#        #TODO: testar TF do sklearn
#
#        return term_frequency
#

    #TODO: IDF
    def IDF():
        pass


    #TODO:
    #dir_name: type string. contains the name of the directtory with all the files to text.
    #expected_output: type string. name of a file, witch should contain the list of all the files in the directory and its expected classes.
    #TODO: PROBLEMA: term frequency only returns a dict with the words from that class, i would like it to asign the value 0 to the words it doesnt have, that are in the vocabulary.
    def test_full_naive( dir_name, expected_output ):
        y =  pd.read_csv(expected_output)
        print(y.head())
        try:
            tests = os.listdir( dir_name )
        except:
            print("o diretorio não existe!")
            return
        number_right_answers = 0
        for test in tests:
            pass
            #TODO: Testar esse arquivo em especifico

            #TODO: comparar resultado com o resultado esperado
            #if correto -> number_right_answers++

        #TODO: retornar a acuracia
        #return float(number_right_answers) / len(tests)




if __name__ == '__main__':
    #tests
    data_directory = 'dados_concatenados/train/'
    nb = NaiveBayes("nbdata.csv")
    ans = nb.query("dados_processados/rec.sport.hockey/test/53739")
    print( "winner = ", ans )



