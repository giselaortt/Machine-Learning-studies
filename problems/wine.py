import numpy as np
import pandas as pd
import multilayer_perceptron as mlp

def wineTesting():
	arquive = open("../dataset/wine/wine.data")
	lines = arquive.readlines()
	array = []
	for line in lines:
		array.append( [ float(element) for element in line.split(',')] )
	array = np.array( array )
	np.random.shuffle( array ) 

	size = len( array )
	training_size = int( 0.85 * size )

	training_x = ((array[: training_size ]).T[1:] ).T
        training_y = ((array[: training_size ]).T[:1] ).T
        test_x = ((array[ training_size :]).T[1:] ).T
        test_y = ((array[ training_size :]).T[:1]).T

	#normalizando os casos de treinamento
	for i in range( len( training_x.T ) ):
		training_x.T[i] = ( training_x.T[i] - np.amin( training_x.T[i] ) )/( np.amax( training_x.T[i] ) - np.amin( training_x.T[i] ) )

	#normalizando os casos teste
	for i in range( len( test_x.T ) ):
		test_x.T[i] = ( test_x.T[i] - np.amin( test_x.T[i] ) )/( np.amax( test_x.T[i] ) - np.amin( test_x.T[i] ) )

	y = training_y
	training_y = np.array( [[0]*3]*len( y ) )
	for i in range( len(y) ):
		training_y[i][ int( y[i] ) -1 ] = 1

	y = test_y
	test_y = np.array( [[0]*3]*len( y ) )
	for i in range( len(y) ):
		test_y[i][int( y[i] ) -1 ] = 1
	
	model = mlp.Model( hiddenLength = 12, outputLength = 3, inputLength = 13)
	model.training( training_x, training_y, maxiterations = 2000, rate  = 0.05, precision = 0.05 )

	averageError = 0
	for exemple, exp in zip(test_x, test_y):
		averageError += (  np.argmax( model.run( exemple ) ) - np.argmax( exp ) )**2

	averageError = averageError/len( test_x.T )
	print 'average error is ', float( averageError )

	arquive.close()

wineTesting()
