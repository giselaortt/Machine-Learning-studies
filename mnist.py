import numpy as np..
import pandas as pd
import multilayer_perceptron as mlp

def mnistTesting( rate, precision, size, iterations, hidden ):
#	array = pd.read_csv("mnisttrain.csv", "r" )
#	print array['label']

	fileptr = open("../datasets/mnist/train.csv", "r" )
	array = np.array( [ [ word for word in line.split(',') ] for line in fileptr.readlines().pop(0) ] )
	print array.T[0]

	return
	training_x = (array.T[: size ]).T
        training_y = (y.T[: size ]).T
        test_x = (array.T[ size : ]).T
        test_y = (y.T[ size : ]).T

	y = training_y
	training_y = np.array( [[0]*10]*len( y ) )
	for i in range( len(y) ):
		training_y[i][ y[i] ] = 1

	y = test_y
	test_y = np.array( [[0]*10]*len( y ) )
	for i in range( len(y) ):
		test_y[i][ y[i] ] = 1

	model = mlp.Model( hiddenLength = hid, outputLength = 10, inputLength = 28*28 )
	model.training( training_x, training_y, maxiterations = iterations, rate  = rate, precision = precision )

	error = 0
	for exemple, exp in zip(test_x, test_y):
		if np.argmax( model.run( exemple ) ) != np.argmax( exp ):
			error = error+1 

	error = error/len( test_x.T )
	print 'average error is ', float( error )
	print 'used:\n', 'hidden length =', model.hiddenLength, 'rate =', rate, 'iterations', iterations, 'precision =', precision
	arquive.close()


def mnist( ):
	array = pd.read_csv("../datasets/mnist/train.csv", "r" ).as_matrix()
	test = pd.read_csv("../datasets/mnist/test.csv", "r" ).as_matrix()
	result = open( "mnistresult.csv", "w" )
	training_x = ( array.T[1:] ).T
        training_y = (array.T[:1]).T

	y = training_y
	training_y = np.array( [[0]*10]*len( y ) )
	for i in range( len(y) ):
		training_y[i][ y[i] ] = 1

	model = mlp.Model( hiddenLength = 10, outputLength = 10, inputLength = 28*28 )
	model.training( training_x, training_y, maxiterations = 2, rate  = 0.1, precision = 0.2 )

	print>>result, "ImageId,Label"
	i = 0
	for case in testcases:
		print>>result, i,",",argmax( model.run( case ) )
	arquive.close()

mnistTesting( rate = 0.5, size= 40000, precision = 0.1, iterations = 200, hidden = 20 )
