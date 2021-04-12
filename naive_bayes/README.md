
# Naive Bayes para classificação de textos

### Dados:
Dados foram tirados daqui https://archive.ics.uci.edu/ml/datasets/Twenty+Newsgroups


### Referências:
A ideia e as guidelines do projeto foram tirados daqui https://www.youtube.com/watch?v=JvvDN_zSXxU&list=PLKWX1jIoUZaWY_4zxjLXnIMU1Suyaa4VX&index=30
 
### Resultados:

	*Acurácia geral:  0.6825 ou 68.25%

	*Acurácia por classe:
		comp.graphics:
			acertos:  235
			erros:  65
			acurácia: 0.7833333333333333 

		  talk.politics.guns:
			acertos:  276
			erros:  24
			acurácia: 0.92 

		  rec.autos:
			acertos:  278
			erros:  22
			acurácia: 0.9266666666666666 

		  rec.motorcycles:
			acertos:  280
			erros:  20
			acurácia: 0.9333333333333333 

		  sci.crypt:
			acertos:  0
			erros:  300
			acurácia: 0.0 

		  comp.sys.mac.hardware:
			acertos:  260
			erros:  40
			acurácia: 0.8666666666666667 

		  comp.windows.x:
			acertos:  0
			erros:  300
			acurácia: 0.0 

		  sci.med:
			acertos:  279
			erros:  21
			acurácia: 0.93 

		  comp.os.ms-windows.misc:
			acertos:  100
			erros:  200
			acurácia: 0.3333333333333333 

		  rec.sport.hockey:
			acertos:  295
			erros:  5
			acurácia: 0.9833333333333333 

		  talk.politics.misc:
			acertos:  188
			erros:  112
			acurácia: 0.6266666666666667 

		  sci.space:
			acertos:  281
			erros:  19
			acurácia: 0.9366666666666666 

		  talk.religion.misc:
			acertos:  120
			erros:  180
			acurácia: 0.4 

		  rec.sport.baseball:
			acertos:  287
			erros:  13
			acurácia: 0.9566666666666667 

		  talk.politics.mideast:
			acertos:  0
			erros:  300
			acurácia: 0.0 

		  alt.atheism:
			acertos:  232
			erros:  68
			acurácia: 0.7733333333333333 

		  misc.forsale:
			acertos:  216
			erros:  84
			acurácia: 0.72 

		  comp.sys.ibm.pc.hardware:
			acertos:  254
			erros:  46
			acurácia: 0.8466666666666667 

		  sci.electronics:
			acertos:  232
			erros:  68
			acurácia: 0.7733333333333333 

	*Matriz de confusão: 
				aguardando a implementação


### O que ainda precisa ser feito:
	*Tratar o caso dos apóstrofos(no momento estão sendo apenas deletados, mas como são parte da lingua inglesa merecem um tratamento especial)
	*Comentarários
	*Tornar pytonico
	*Testar a complexidade e possivelmente reduzí-la
	*Implementar a métrica IDF e testar se ela reduz o erro
	*Implementar a matriz de confusão
	*Tratar a divisão por zero nos logarítimos
