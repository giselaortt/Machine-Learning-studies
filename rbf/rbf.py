import sys
import pandas as pd
import numpy as np

sys.path.append("../k-means")

from k_means import*

class rbf:
	def __init__( size, centers = 10, gamma = 0.1 ):
		self.gamma = gamma
		self.centers = centers
		self.hidden_layer = np.random.random(neurons)
		self.hidden_layer_size = size


	def fit( dataset, classes ):
		pass


