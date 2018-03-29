import numpy as np

arr = np.array(['s','s','s','s','n','n','n','n','n'])
values, counts = np.unique(arr, return_counts = True)
print( values[ np.argmax(counts) ] )
