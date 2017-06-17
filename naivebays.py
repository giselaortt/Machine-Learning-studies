import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from titanic import *

def count( dataset, column, atribute ):
	return float( sum( [ _ == atribute for _ in dataset[ column ] ] ) )

# gets probability of surviving or not for all instances of an atribute
def get_condtional_probs( dataset, atribute ):
	#using set in order to remove repetitions
	colnames = set([ element for element in dataset[ atribute ]])
	data = pd.DataFrame( np.zeros(( 2, len(colnames) )), columns = list(colnames))
	survivals = count( dataset, 'Survived', 1 )
	ded = count( dataset, 'Survived', 0 )

	i = 0
	for instance in data:
		instance_and_survived = sum([ _ == instance and __ == 1 for _,__ in zip( dataset[ atribute ], dataset.Survived )])
		instance_and_ded = sum([ _ == instance and __ == 0 for _,__ in zip( dataset[ atribute ], dataset.Survived )]) 
		data[instance][1] = float(instance_and_survived)/survivals
		data[instance][0] = float(instance_and_ded)/ded
		i = i+1
	
	return data

# not generic naive bayes for now
def naive_bayes():
	train, surv, queries = getdata()
	train['Survived'] = surv
	train = process_naivebayes( train )
	queries = process_naivebayes( queries )

	p_survive =  float( sum( [ _ == 1 for _ in train.Survived ] ) )/float( ( len(train.Survived) ) )
	p_die = float( sum( [ _ == 0 for _ in train.Survived ] ) )/float( ( len(train.Survived) ) )
	result = open("naiveres.txt", "w" )

	dictionary = {}
	for atribute in train.columns:
		if atribute != 'Survived':
			dictionary[atribute] = get_condtional_probs( train, atribute )

	print>>result,"PassengerId,Survived"
	j = 892
	for i in range( len( queries.Sex ) ):
		p_yes = p_survive
		p_no = p_die
		for atribute in queries.columns:
			instance = queries[ atribute ][i]
			p_yes = p_yes*dictionary[ atribute ][ instance ][1]
			p_no = p_no*dictionary[ atribute ][ instance ][0]
		p_yes = p_yes
		p_no = p_no
		if p_yes > p_no:
			res = 1
		else:
			res = 0
                print>>result, str( j ) + "," + str( res )
                j = j+1

        result.close()

	return

naive_bayes()

