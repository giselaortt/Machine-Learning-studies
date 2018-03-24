import numpy as np
import pandas as pd
import multilayer_perceptron as mlp

def breastTesting():
	arquive = open( "../datasets/beast/breast.dat", "r" )
	lines = arquive.readlines()
	array = []
	for line in lines:
		array.append( [ int(element) for element in line.split()] )
	array = np.array( array )
	for i in range( len( array.T ) ):
		array.T[i] = ( array.T[i] - np.amin( array.T[i] ) )/( np.amax( array.T[i] ) - np.amin( array.T[i] ) )

	tx = ((array[:650]).T[0:9] ).T
        ty = ((array[:650]).T[9:] ).T
        tesx = ((array[650:]).T[0:9] ).T
        tesy = ((array[650:]).T[9:]).T
	model = mlp.Model( hiddenLength = 13, outputLength = 1, inputLength = 9)
	model.training( tx, ty, maxiterations = 2000 , precision = 0.05, rate = 0.1)
	averageError = 0;

	for exemple, exp in zip(tesx, tesy):
		got = model.run( exemple )
		averageError += (got - exp )**2

	averageError = averageError/len( tesx.T )
	print 'average error is ', float( averageError )
	arquive.close()
	return

breastTesting()
