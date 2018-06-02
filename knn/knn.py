
import numpy as np


'''
dataset: Numpy object. Classes must be the last column.
'''
class knn:
	def __init__( self, votes, dataset ):
		self.votes = votes
		self.dataset = dataset

	def query( query ):
		#distances
		distances = np.array([distance(row, query) for row in self.dataset[:,:-1].astype(float)])

		#indices dos k elementos mais proximos
		indexes = np.argsort( distances )[:self.votes]

		#consultar classes
		classes = dataset[indexes,-1]
		(values,counts) = np.unique(classes, return_counts = True)

		return values, counts


	def predict( data ):
		answer = []
		for row in data:
			values, counts = self.query( row )
			answer.append( values[ np.argmax(counts) ] )
		return answer


def distance( vector, other_vector ):

	return np.sqrt( np.sum((vector - other_vector)**2))


