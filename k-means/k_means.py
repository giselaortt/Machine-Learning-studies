import pandas as pd
import numpy as np


def distance( x, y ):
	return np.sqrt( np.sum( (x-y)**2 ) )


def k_means( data:pandas.DataFrame, k = 10, treshold = 0.02 ):
	centers = data.sample(k, axis = 0)
	new_centers = centers.copy(deep = False)
	divergence = 1
	ids = np.zeros(data.shape[0])

	while( divergence > treshold ):
		divergence = 0
		for i, point in  enumerate(data) :
			distances = centers.apply( lambda x: np.sqrt(np.sum((point - x)**2)), axis = 0 )
			ids[i] = np.argmin( distances )

		for i in range(k):
			reach = data[ (ids == i) ]
			for j,column in enumerate(reach.columns):
				new_centers.iloc[i, j] = np.mean(reach[column])
			divergence += distance( new_centers[i].values, centers[i].values )
		centers = new_centers.copy()

	return centers


