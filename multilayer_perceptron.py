import numpy as np
import math
from random import random

def sigmoid(gamma):
	if gamma < 0:
		return 1 - 1/(1 + math.exp(gamma))
	else:
		return 1/(1 + math.exp(-gamma))

def derSigmoid(x):
	return ( sigmoid(x)*( 1 - sigmoid(x) ) )

class Fwd:
	def __init__( self, f_o, dfo_dneto, f_h, dfh_dneth ):
		self.f_o = f_o
		self.dfo_dneto = dfo_dneto
		self.dfh_dneth = dfh_dneth
		self.f_h = f_h

class Model:
	def __init__( self, inputLength, outputLength, hiddenLength, function = sigmoid, derivative = derSigmoid  ):
		self.hiddenLayer = np.random.random(( hiddenLength, inputLength + 1)) - 0.5
		self.outputLayer= np.random.random(( outputLength, hiddenLength + 1 )) - 0.5
		self.inputLength = inputLength
		self.outputLength = outputLength
		self.hiddenLength = hiddenLength
		self.function = function
		self.derivative = derivative

	def findNet( self, layer, vetor ):
		nlin = len( layer )
		ncol = len( layer.T )
		net = np.array( [ 0 for i in range( nlin ) ] )
		for i in range( nlin ):
			net[i] = layer[i][ncol-1]
			for j in range( ncol - 1 ):
				net[i] += layer[i][j]*vetor[j]
		return net;

	def forward( self, x ):
		netHidden = self.findNet( self.hiddenLayer, x )
		fHidden = np.array([ self.function( net ) for net in netHidden.flat ])
		dfhidden = np.array([ self.derivative( net ) for net in netHidden.flat ])
		netOutput = self.findNet( self.outputLayer, fHidden )
		fOutput = np.array([ self.function( net ) for net in netOutput.flat ])
		dfOutput = np.array([ self.derivative( net ) for net in netOutput.flat ])
		ans = Fwd( fOutput, dfOutput, fHidden, dfhidden )
		return ans

	def training(self, sampleInput, expectedOutput, precision = 0.016, rate = 0.05, maxiterations = 2**32 ):
		error = 2.5
		it = 0
		while error > precision:
			error = 0
			for x, y in zip( sampleInput, expectedOutput ):
				fwd = self.forward( x )
				delta_e  = ( fwd.f_o - y )
				for element in delta_e:
					error += element**2
				# atualizando a camada de saida
				for neuronio, flinha, erro in zip( self.outputLayer, fwd.dfo_dneto, delta_e ):
					neuronio -= rate * erro * flinha * np.append( fwd.f_h, 1 )
				#atualizando a camada escondida
				for neuronio, weigths, flinha in zip( self.hiddenLayer, self.outputLayer.T, fwd.dfh_dneth ):
					soma = np.sum( delta_e * fwd.dfo_dneto * weigths )
					delta_h = soma * flinha * np.append( x, 1 )
					neuronio -= rate * delta_h

			error = error/( len( sampleInput ) )
			print 'it = ', it, 'error = ',  error 
			it = it + 1
			if it > maxiterations :
				break

	def run( self, x ):
		ans = self.findNet( self.hiddenLayer, x )
		ans = np.array([ self.function( net ) for net in ans.flat ])
		ans = self.findNet( self.outputLayer, ans )
		ans = np.array([ self.function( net ) for net in ans.flat ])
		return ans

def rbf_kernel( x, y, gamma = 0.1 ):
	# aplies radial kernel on two vectors #
	 return np.exp( -gamma*np.sum( (x-y)**2 ) )

def matrix_kernel( x, y, kernel = rbf_kernel ):
	# receives two matrix and generates another matrix based on kernel #
	answer = np.zeros((x.shape[0], y.shape[0]))
	for i, xvector in enumerate(x):
		for j, yvector in enumerate(y):
			answer[i][j] = kernel( xvector , yvector , 1 )
	return answer
