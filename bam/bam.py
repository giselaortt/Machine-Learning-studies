import numpy as np
import random


class BAM:

	def __init__(self, dimetions):
		self.matrix = np.zeros(dimetions)	


	def train(self, inputs, outputs ):
		for x,y in zip( inputs, outputs ):
			self.matrix = self.matrix + np.inner(x.reshape(x.shape[0],1), y.reshape(y.shape[0], 1))


	def recover_x(self, x):
	
		old_entropy = 0
		new_entropy = 0
		delta = 1

		while( delta != 0 ):
			y = self.acess_y( x )
			new_entropy = entropy( x, self.matrix, y )
			delta = abs( old_entropy - new_entropy )
			old_entropy = new_entropy
			x = self.acess_x( y )

		return x


	def recover_y(self, y):
	
		old_entropy = 0
		new_entropy = 0
		delta = 1

		while( delta != 0 ):
			x = self.acess_x( y )
			new_entropy = entropy( y, self.matrix.T, x )
			delta = abs( old_entropy - new_entropy )
			old_entropy = new_entropy
			y = self.acess_y( x )

		return y

 
	def acess_y(self, vec):
		return binary( np.inner(vec, self.matrix.T) )


	def acess_x(self, vec):
		return binary( np.inner(vec, self.matrix) )


def binary( vector ):
	vector[ ( vector == 0 ) ] = random.sample( [-1,1], 1 )
	return (vector/abs(vector)).astype(float)


def entropy( padrao, matriz, saida ):
	return np.inner( np.inner( padrao, matriz.T  ), saida )


