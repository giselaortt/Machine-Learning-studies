import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



class NaiveBayes:
    def __init__(self, database, ycolumn = None):
        self.database = pd.DataFrame(database)
        if( ycolumn is None ):
            ycolumn = database.shape[1]-1
            self.ycolumn = ycolumn
            self.classes = pd.unique( database.iloc[:,ycolumn] )

    def query( self, query ):
        #lista de probabbilidades para cada possível output
        probabilities = []
        for instance in self.classes:
            freq_i = len( self.database.loc[ self.database.iloc[:,self.ycolumn] == instance ] )
            prob_class_i = freq_i/ self.database.shape[0]
            for i, condition in zip(range(self.database.shape[1]),query):
                if condition is not None:
                  prob_class_i = prob_class_i*(len( self.database.loc[ (self.database.iloc[:,self.ycolumn]==instance)&(self.database.iloc[:,i] == condition) ])/freq_i )
            probabilities.append( prob_class_i )
        #normalizar probabilities
        soma = sum( probabilities )
        probabilities = [ (float)(i) / soma for i in probabilities]
        #retornar a classe correspondente a maior probabilidade

        return self.classes[np.argmax( probabilities )]
        #TODO: print probabilities for all the classes insted of just showing the biggest one


    #TODO:
    def bag_of_words():
        pass


    #should return a dataframe with all the words present on the document and its normalized freequency
    def term_frequency( text ):
        list_of_words = text.split(' ')
        words_unique = set(list_of_words)
        term_frequency = dict.fromkeys( words.unique, 0 )
        total = 0
        for word in list_of_words:
            term_frequency[words] += 1
            total += 1
        for word, repetitions in term_frequency.items():
            term_frequency[word] = repetitions / total
        #TODO: passar para dataframe
        return term_frequency


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
    pass
    
