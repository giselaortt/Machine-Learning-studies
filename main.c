#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#include "mlp.h"

int main(){
	int size = 42000;
	int i, j, k, a;

	FILE* arq = fopen("parsed_train.csv", "r" );
	FILE* que = fopen("parsed_test.csv", "r" );
	FILE* result = fopen("mnist_result5.csv", "w");

	long double** sample = alloc_matrix( size, 28*28 );
	long double** expected = alloc_matrix( size, 10 );
	long double** queries = alloc_matrix( 28000, 28*28 );

	zero_matrix( sample, size, 28*28 );
	zero_matrix( queries, 28000, 28*28 );
	zero_matrix( expected, size, 10 );
	printf("reading...\n");

	for( i=0; fscanf( arq,"%d", &a ) != EOF && i < size ; i++ ){
		expected[i][ a ] = 1;
		for( j=0; j<28*28; j++ ){
			long double b;
			fscanf( arq, "%Lf", &b );
			sample[i][j] = b/255.0 ;
		}
	}

	for( i=0; i < 28000 ; i++ ){
		for( j=0; j<28*28; j++ ){
			long double b;
			fscanf( que, "%Lf", &b );
			queries[i][j] = b /255.0 ;
		}
	}

	printf("training...\n");
	long double(*function)(long double);
	function = sigmoid;
	long double(*derivative)(long double);
	derivative = sigmoid_derivative; 
	model* m = build_model( 20, 10, 28*28, function, derivative);
	training( m, sample, expected, size, 0.001, 0.1, 1000 );

	printf("predicting...\n");
	fprintf( result, "ImageId,Label\n" );
	predict( m, queries, 28000, result );

	free_model( m );
	free_matrix( sample, size, 28*28 );
	free_matrix( expected, size, 10 );
	free_matrix( queries, 28000, 28*28 );

	fclose( result );
	fclose( que );
	fclose( arq );
	return 0;
}


