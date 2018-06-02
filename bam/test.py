import numpy as np
import random
from bam import *

inputs = np.array( [[-1,-1,-1,1,1,1],[-1,+1,+1,-1,-1,1]])
outputs = np.array( [[1,1,1,-1],[1,-1,-1,+1]] )

b = BAM((6,4))
b.train( inputs, outputs )

y1 = np.array([1,1,-1,-1])

print(inputs)

print( b.recover_y(y1) )
