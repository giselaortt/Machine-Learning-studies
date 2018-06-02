# remove warnings
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import multilayer_perceptron as mlp

#reads data
def getdata():
	training_data = pd.read_csv("../datasets/titanic/train.csv")
	testing_data = pd.read_csv("../datasets/titanic/test.csv")
	expected = training_data.Survived
	training_data.drop('Survived', 1, inplace=True)
	training_data.drop('Cabin', 1, inplace=True)
	training_data.drop('Ticket', 1, inplace=True)
	training_data.drop('PassengerId', 1, inplace=True)
	testing_data.drop('Cabin', 1, inplace=True)
	testing_data.drop('Ticket', 1, inplace=True)
	testing_data.drop('PassengerId', 1, inplace=True)
	return training_data, expected, testing_data

#creates new atribute titles, based on names
def titles( matrix ):
	matrix['title'] = matrix['Name'].map( lambda name:name.split(',')[1].split('.')[0].strip() )
	Title_Dictionary = {
                        "Capt":       "Officer",
                        "Col":        "Officer",
                        "Major":      "Officer",
                        "Jonkheer":   "Royalty",
                        "Don":        "Royalty",
                        "Sir" :       "Royalty",
                        "Dr":         "Officer",
                        "Rev":        "Officer",
                        "the Countess":"Royalty",
                        "Dona":       "Royalty",
                        "Mme":        "Mrs",
                        "Mlle":       "Miss",
                        "Ms":         "Mrs",
                        "Mr" :        "Mr",
                        "Mrs" :       "Mrs",
                        "Miss" :      "Miss",
                        "Master" :    "Master",
                        "Lady" :      "Royalty"

	}
	matrix['title'] = matrix.title.map(Title_Dictionary)
	matrix.drop('Name', 1, inplace=True)

#generate medias about the ages in order to fill it up. as a first attempt i will try using only title colum for that.
def get_medias( dataset ):
	possible_titles = set( dataset['title'] )
	nrow = len( dataset )
	medias = {}
	ocorrencias = {}
	for title in possible_titles:
		medias[ title ] = 0
		ocorrencias[ title ] = 0
	for i in range( nrow ):
		if np.isnan( dataset.Age[i] ) == False:
			medias[ dataset.title[ i ] ] = medias[ dataset.title[ i ] ] + dataset.Age[i]
			ocorrencias[ dataset.title[ i ] ] = ocorrencias[ dataset.title[ i ] ] + 1
	for keyword in possible_titles:
		medias[ keyword ] = medias[ keyword]/ocorrencias[ keyword ]
	return medias

def filling( dataset ):
	medias = get_medias( dataset )
	nrow = len( dataset )
	for i in range( 0, nrow ):
		if np.isnan( dataset.Age[i] ):
			dataset.Age[i] = medias[ dataset.title[i] ]
	# filling the nan's
	# fare its filled by its mean and embarked its filled by the most common
	dataset.Fare.fillna( dataset.Fare.mean(), inplace = True )
	dataset.Embarked.fillna( 'S', inplace = True )
	return dataset

def family_process( dataset ):
	dataset['familysize'] = dataset['SibSp'] + dataset['Parch'] + 1
	dataset.drop('SibSp', 1, inplace=True)
	dataset.drop('Parch', 1, inplace=True)

def mapping( dataset ):
	embarked_dummies = pd.get_dummies(dataset['Embarked'],prefix='Embarked')
	dataset = pd.concat([dataset,embarked_dummies],axis=1)
	dataset.drop('Embarked',axis=1,inplace=True) 
	pclass = pd.get_dummies( dataset['Pclass'], prefix = 'Pclass')
	dataset = pd.concat([dataset, pclass],axis = 1 )
	dataset.drop('Pclass', axis=1, inplace=True )
	dataset.Sex = dataset.Sex.map({'male':1, 'female':0})
	return dataset

#used for naive bayes
#groups ages in tree classes, wich have been cosen by ploting ages correlation with surviving rate
#and grouped by similarity
def process_ages( dataset ):
	for i in range( len( dataset.Age ) ):
		if dataset.Age[i] <= 11:
			dataset.Age[i] = 0
		elif 11 < dataset.Age[i] and dataset.Age[i] <= 50:
			dataset.Age[i] = 1
		else:
			dataset.Age[i] = 2
	return

#proces dataset for multilayer perceptron using
def process( dataset ):
	titles( dataset )
	filling( dataset )
	dataset.drop('title', 1, inplace=True)
	family_process( dataset )
	dataset = mapping( dataset )
	dataset['Age'] = (dataset['Age'] - np.amin(dataset['Age']) )/(np.amax( dataset['Age'] ) - np.amin(dataset['Age']) )
	dataset['Fare'] = (dataset['Fare'] - np.amin(dataset['Fare']) )/(np.amax( dataset['Fare'] ) - np.amin(dataset['Fare']))
	#converting it numpy array
	dataset = dataset.as_matrix().T[1:].T
	return dataset

#process dataset for naivebayes use. gets rid of countinuos atributes and groups them
def process_naivebayes( dataset ):
	titles( dataset )
	filling( dataset )
	dataset.drop('title', 1, inplace=True)
	family_process( dataset )
	for i in range( len( dataset.familysize ) ):
		if dataset.familysize[i] > 5:
			dataset.familysize[i] = 5
	dataset.drop('Fare', 1, inplace = True )
	process_ages( dataset )
	return dataset

#tests multilayer perceptron algorithm on titanic dataset
def titanicTesting( size, iterations, rate, hidden, precision ):	
	train, expected, test = getdata()
	train = process( train )

	training_x = train[0:size]
	testing_x = train[size:]
	training_y = expected[0:size]
	testing_y = expected[size:]

	model = mlp.Model( inputLength = len(train.T), outputLength = 1, hiddenLength = hidden )
	model.training( expectedOutput = training_y, sampleInput = training_x, precision = precision, maxiterations = iterations, rate = rate )
	error = 0
	for element, expected in zip( testing_x, testing_y ):
		got = model.run( element )
		error = error + ( got - expected )**2
	error = error/len( testing_x )
	print 'Average error = ', error
	return

def titanic( size, iterations, rate, hidden, precision ):
	train, expected, test = getdata()
	train = process( train )
	test = process( test )
	result = open("titanicresult.txt", "w" )

	expected = expected.as_matrix()
	model = mlp.Model( hiddenLength = hidden, outputLength = 1, inputLength = len( train.T ) )
	model.training( precision = precision, rate = rate, sampleInput = train, maxiterations = iterations, expectedOutput = expected )

	error = 0
	print>>result,"PassengerId,Survived"
	i = 892
	for case in test:
		res = model.run( case )
		if res < 0.5:
			res = 0
		else:
			res = 1
		print>>result, str( i ) + "," + str( res )
		i = i+1

	result.close()
	return
