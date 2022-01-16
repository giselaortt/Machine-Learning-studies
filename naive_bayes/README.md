
# Naive Bayes for text classification:

### Data:
Data extracted from here https://archive.ics.uci.edu/ml/datasets/Twenty+Newsgroups


### Reference:
Project idea and guidelines were taken from this video https://www.youtube.com/watch?v=JvvDN_zSXxU&list=PLKWX1jIoUZaWY_4zxjLXnIMU1Suyaa4VX&index=30
 
### Results:

	- General accuracy:  0.8072080579162733 (~80%)

	- accuracy per class:
		soc.religion.christian :
			acertos:  486
			erros:  168
			acurácia da classe: 0.7431192660550459 


		comp.graphics :
			acertos:  224
			erros:  76
			acurácia da classe: 0.7466666666666667 

		talk.politics.guns :
			acertos:  273
			erros:  27
			acurácia da classe: 0.91 


		rec.autos :
			acertos:  276
			erros:  24
			acurácia da classe: 0.92 


		rec.motorcycles :
			acertos:  288
			erros:  12
			acurácia da classe: 0.96 


		sci.crypt :
			acertos:  272
			erros:  28
			acurácia da classe: 0.9066666666666666 


		comp.sys.mac.hardware :
			acertos:  268
			erros:  32
			acurácia da classe: 0.8933333333333333 


		comp.windows.x :
			acertos:  263
			erros:  37
			acurácia da classe: 0.8766666666666667 


		sci.med :
			acertos:  278
			erros:  22
			acurácia da classe: 0.9266666666666666 


		comp.os.ms-windows.misc :
			acertos:  117
			erros:  183
			acurácia da classe: 0.39 


		rec.sport.hockey :
			acertos:  297
			erros:  3
			acurácia da classe: 0.99 


		talk.politics.misc :
			acertos:  179
			erros:  121
			acurácia da classe: 0.5966666666666667 


		sci.space :
			acertos:  283
			erros:  17
			acurácia da classe: 0.9433333333333334 


		talk.religion.misc :
			acertos:  114
			erros:  186
			acurácia da classe: 0.38 


		rec.sport.baseball :
			acertos:  291
			erros:  9
			acurácia da classe: 0.97 


		talk.politics.mideast :
			acertos:  290
			erros:  10
			acurácia da classe: 0.9666666666666667 


		alt.atheism :
			acertos:  230
			erros:  70
			acurácia da classe: 0.7666666666666667 


		misc.forsale :
			acertos:  218
			erros:  82
			acurácia da classe: 0.7266666666666667 


		comp.sys.ibm.pc.hardware :
			acertos:  247
			erros:  53
			acurácia da classe: 0.8233333333333334 


		sci.electronics :
			acertos:  235
			erros:  65
			acurácia da classe: 0.7833333333333333 

	- Matriz de confusão: 
				to be implemented


### To be implemented:
	- Refactor according to clean code guidelines
	- Change to IDF metrics and compare results
	- Confusion Matrix
	- Division for zero on the logaritmos need to be handled
=
