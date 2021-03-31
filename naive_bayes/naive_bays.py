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
    #lista de probabbilidades para casa possível output
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


if __name__ == '__main__':
    #tests
    pass
    
