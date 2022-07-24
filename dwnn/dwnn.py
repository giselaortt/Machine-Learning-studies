import numy as np

class dwnn:
	def __init__(self, sigma, dataset):
		self.sigma = sigma
		self.dataset = dataset[:,:-1]
		self.classes = dataset[:,-1]


	# Euclidian distance
	def distance( self, a, b ):
		return np.sqrt(np.sum((a - b)**2))


	def weights( self, a, b ):
		dist = distance(a,b)
		return math.exp( -(dist**2)/2*(self.sigma**2) ) 


	def query( self, query ):
		weights = [ weights(query, row) for row in self.dataset ]
		return np.sum(weights*query)/np.sum(weights)


	def predict( self, data ):
		return np.array([ self.query(row) for row in data ])

