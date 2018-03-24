import numpy as np
import random


class BAM:

	def __init__(self, dimetions):
		self.matrix = np.zeros(dimetions)	


	def train(self, inputs, outputs ):
		for x,y in zip( inputs, outputs ):
			self.matrix = self.matrix + np.inner(x.reshape(x.shape[0],1), y.reshape(y.shape[0], 1))


	def recover_x(self, x):
	
		old_entropia = 0
		new_entropia = 0
		delta = 1

		while( delta != 0 ):
			y = self.acess_y( x )
			new_entropia = entropia( x, self.matrix, y )
			delta = abs( old_entropia - new_entropia )
			old_entropia = new_entropia
			x = self.acess_x( y )

		return x


	def recover_y(self, y):
	
		old_entropia = 0
		new_entropia = 0
		delta = 1

		while( delta != 0 ):
			x = self.acess_x( y )
			new_entropia = entropia( y, self.matrix.T, x )
			delta = abs( old_entropia - new_entropia )
			old_entropia = new_entropia
			y = self.acess_y( x )

		return y

 
	def acess_y(self, vec):
		return binary( np.inner(vec, self.matrix.T) )


	def acess_x(self, vec):
		return binary( np.inner(vec, self.matrix) )


def binary( vector ):
	vector[ ( vector == 0 ) ] = random.sample( [-1,1], 1 )
	return (vector/abs(vector)).astype(float)


def entropia( padrao, matriz, saida ):
	return np.inner( np.inner( padrao, matriz.T  ), saida )


