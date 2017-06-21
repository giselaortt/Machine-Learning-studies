arq = open("../datasets/mnist/train.csv", "r")
arq2 = open("../datasets/mnist/test.csv", "r")
parse = open("parsed_train.csv", "w" )
parse2 = open("parsed_test.csv", "w" )

text = arq.readlines()
text.pop( 0 )

print len( text ) -1


array = [ [int(word) for word in line.split(',') ]for line in text ]

for line in array:
	for number in line:
		parse.write( str( number ) )
		parse.write(' ')
	parse.write('\n')


text = arq2.readlines()
text.pop( 0 )

array = [ [int(word) for word in line.split(',') ]for line in text ]

for line in array:
	for number in line:
		parse2.write( str( number ) )
		parse2.write(' ')
	parse2.write('\n')

parse.close()
parse2.close()
arq.close()
arq2.close()
