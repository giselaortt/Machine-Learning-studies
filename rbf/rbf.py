import sys
import pandas as pd
import numpy as np

from k_means import kmeans

class rbf:
	def __init__( size = 10, gamma = 0.1, treshold = 0.002, function = None ):
		#TODO care about the gamma thing
		#self.gammas = [gamma]*size
		self.gammas = gamma
		self.weights = np.random.random(centers)-0.5
		self.size = size
		self.centers = None
		self.treshold = treshold
		if function is None:
			self.function = self.standard
		else:
			self.function = function

	'''
	solving weights matrix
	'''
	#TODO how to code the pseudo-transpose?
	# w = ( phiT * phi )-1 * phiT * y
	# phi: functions matriz
	# y classes
	# TODO linear regression
	def fit( dataset, classes ):
		self.centers = kmeans(dataset, self.size, self.treshold)


	'''
	finding gamma parameters that minimizes error function
	'''
	def gradient_descendent( self, data, classes ):
		pass


	'''
	Goes back and forth through gradient-descendenting the gamma parameter and solving for the weights matrix.
	'''
	def expectation_maximization( self, data, classes ):
		pass


	def standard( self, point_one, point_two ):

		return np.exp( self.gamma*np.sum(( point_one - point_two )**2) )


	def error( self, classes, predictions ):
		pass


	def run( self, point ):
		if centers is None:
			print('you didnt fit your data!')
			return/

		#TODO simpler way?
		hidden = np.array([ self.function( point, center ) for center in self.centers ])
		return np.inner( hidden, self.weights )


	#TODO
	def predict( self, data ):
		pass
