#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define ERRO 3

/*
typedef struct matrix{
	
}
*/
typedef struct model{
	int hiddenlen, outlen, inlen;
	long double** hiddenlayer;
	long double** outputlayer;
	long double(*function)( long double );
	long double(*der)( long double );
}model;

long double* allocvector( int len ){
	long double* ans = (long double*)malloc(sizeof(long double)*len );
	return ans;
}

long double** alocamatriz( int row, int col ){
	long double** ans = (long double**)malloc(sizeof(long double*)*row);
	int i;
	for( i=0; i<row; i++ )
		ans[i] = (long double*)malloc(sizeof(long double)*col);
	return ans;
}

void freematriz( long double** mat, int row, int col ){
	int i;
	for( i=0; i<col; i++ )
		free( mat[i] );
	free( mat );
}

void fill( long double** matriz, int row, int col ){
	srand(time(NULL));
	int i, j;
	for( i=0; i<row; i++ )
		for( j=0; j<col; j++ )
			matriz[i][j] = (float)( rand()%1000 )/1000.0 - 0.5;
}

model* buildmodel( int hiddenlen, int outlen, int inlen, long double(*function)(long double) , long double(*der)(long double)  ){
	model*  m = (model*)malloc(sizeof(model));
	m->hiddenlen = hiddenlen;
	m->inlen = inlen;
	m->outlen = outlen;
	m->function = function;
	m->der = der;
	m->hiddenlayer = alocamatriz( m->hiddenlen, m->inlen + 1 );
	m->outputlayer = alocamatriz( m->outlen, m->hiddenlen + 1 );
	fill( m->hiddenlayer, m->hiddenlen, m->inlen + 1 );
	fill( m->outputlayer, m->outlen, m->hiddenlen + 1 );
	return m;
}

void freemodel( model* m ){
	freematriz( m->hiddenlayer, m->hiddenlen, m->inlen + 1 );
	freematriz( m->outputlayer, m->outlen, m->hiddenlen + 1 );
	free( m );
}

long double sigmoid( long double gamma ){
        if( gamma < 0 )
                return 1.0 - 1.0/(1.0 + pow( M_E ,gamma));
        else
                return 1.0/(1.0 + pow( M_E, -gamma));
}

long double der_sigmoid( long double gamma ){
	return ( sigmoid(gamma)*( 1.0 - sigmoid(gamma) ) );
}

void findnet( long double** layer, int nrow, int ncol , long double* vetor, long double* net ){
	int i, j;
	for( i=0; i<nrow; i++ ){
		net[i] = 0;
		for( j=0; j<ncol - 1; j++ )
			net[i] += layer[i][j]*vetor[j];
		net[i] += layer[i][ ncol -1 ]; // bias!
	}
}

long double* run( model* m, long double* vetor ){
	int i;
	long double* net_hidden = allocvector( m->hiddenlen );
	findnet( m->hiddenlayer, m->hiddenlen, m->inlen + 1, vetor, net_hidden );
	for( i=0; i<m->hiddenlen; i++ )
		net_hidden[i] = m->function( net_hidden[i] );
	long double* net_output = allocvector(m->outlen );
	findnet( m->outputlayer, m->outlen, m->hiddenlen + 1, net_hidden /*now f_hidden!*/, net_output );
	for( i=0; i<m->outlen; i++ )
		net_output[i] = m->function( net_output[i] );
	free( net_hidden );
	return net_output;
}

long double* copy( long double* vetor, int len ){
	long double* ans = (long double*)malloc(sizeof(long double)*len);
	int i;
	for( i=0; i<len; i++ )
		ans[i] = vetor[i];
	return ans;
}

void apply( long double(*function)(long double), long double* vetor, long double* source, int len ){
	int i;
	for( i=0; i<len; i++ ){
		vetor[i] = function( source[i] );
	}
}

void printvector( long double* vec, int len ){
	int i=0;
	for( i=0; i<len; i++ ){
		printf("%Lf ", vec[i] );
	}
	printf("\n");
}

void printmatrix( long double** vec, int nrow, int ncol ){
	int i, j;
	for( i=0; i<nrow; i++ ){
		for( j=0; j<ncol; j++ ){
			printf("%Lf ", vec[i][j] );
		}
		printf("\n");
	}
	printf("\n");
}

void training( model* m, long double** sample, long double** expected, int len, long double precision, long double rate, int limit ){
	long double error, delta;
	int it = 0, j, i, k,l;
	long double *temp, *df_output, *delta_e ,*df_hidden, *net_hidden, *f_hidden, *net_output, *f_output,  *delta_o, *delta_h;

	delta_e = allocvector( m->outlen );
	delta_o = allocvector( m->outlen );
	delta_h = allocvector( m->hiddenlen );
	net_hidden = allocvector( m->hiddenlen );
	net_output = allocvector( m->outlen );
	f_hidden = allocvector( m->hiddenlen );
	df_hidden = allocvector( m->hiddenlen );
	f_output = allocvector( m->outlen );
	df_output = allocvector( m->outlen );

	do{
		error = 0;
		for( j=0; j < len; j++){

			findnet( m->hiddenlayer, m->hiddenlen, m->inlen + 1, sample[j], net_hidden );
			apply( m->function, f_hidden, net_hidden, m->hiddenlen );
			apply( m->der, df_hidden, net_hidden, m->hiddenlen );
			
			findnet( m->outputlayer, m->outlen, m->hiddenlen + 1, f_hidden , net_output );
			apply( m->function, f_output, net_output, m->outlen );
			apply( m->der, df_output, net_output, m->outlen );

			for( i=0; i<m->outlen; i++ ){
				delta_e[i] = ( -f_output[i] + expected[j][i] );
				error += delta_e[i]*delta_e[i];
			}

			for( i=0; i< m->outlen; i++ )
				delta_o[i] = delta_e[i] * df_output[i];

			for( i=0; i<m->hiddenlen; i++ ){
			    long double aux = 0;
			    for( k=0; k < m->outlen; k++ )
				aux += m->outputlayer[k][i] * delta_o[k] ;
			    delta_h[i] = aux * df_hidden[i];
			}
			// atualizing hidden layer
			for( i=0; i<m->hiddenlen; i++ ){
				for( k=0; k< m->inlen ; k++ )
					m->hiddenlayer[i][k] += delta_h[i] * rate * sample[j][k];
				m->hiddenlayer[i][ m->inlen ] += delta_h[i] * rate;
			}

			for( i=0; i<m->outlen; i++ ){
				for( k=0; k< m->hiddenlen; k++ )
					m->outputlayer[i][k] += delta_o[i] * rate * f_hidden[k];
				m->outputlayer[i][ m->hiddenlen ] += delta_o[i] * rate;
			}

		}
		error = error /= len;
		it++;
		printf("%d average error is %Lf\n", it ,error ); 
	}while( error > precision && it < limit );

	free( delta_e );
	free( delta_o );
	free( delta_h);
	free( net_hidden);
	free( net_output );
	free( f_hidden);
	free( df_hidden);
	free( f_output );
	free( df_output );
}

void zerarmatriz( long double** mat, int row, int col ){
	int i, j;
	for( i=0; i<row; i++ ){
		for( j=0; j<col; j++ ){
			mat[i][j] = 0;
		}
	}
}

int main(){ 
	FILE* arq = fopen("parsed_train.csv", "r" );
	FILE* que = fopen("parsed_test.csv", "r" );
	size = 42000;

	long double** sample = alocamatriz( size, 28*28 );
	long double** expected = alocamatriz( size, 10 );
	long double** queries = alocamatriz( 20000, 28*28 );
	zerarmatriz( sample, size, 28*28 );
	zerarmatriz( expected, size, 10 );
	int i, j, k, a;

	for( i=0; fscanf( arq,"%d", &a ) != EOF && i < size ; i++ ){
		expected[i][ a ] = 1;
		for( j=0; j<28*28; j++ ){
			long double b;
			fscanf( arq, "%Lf", &b );
			sample[i][j] = b /255.0 ;
		}
	}

	for( i=0; fscanf( arq,"%d", &a ) != EOF && i < size ; i++ ){
		expected[i][ a ] = 1;
		for( j=0; j<28*28; j++ ){
			long double b;
			fscanf( arq, "%Lf", &b );
			sample[i][j] = b /255.0 ;
		}
	}

	long double(*function)(long double);
	function = sigmoid;
	long double(*der)(long double);
	der = der_sigmoid; 

	model* m = buildmodel( 8, 10, 28*28, function, der);

	training( m, sample, expected, 15000, 0.01, 0.1, 600 );

	freematriz( sample, size, 28*28 );
	freematriz( expected, size, 10 );
	freematriz( queries, 20000, 28*28 );

	fclose( que );
	fclose( arq );
	return 0;
}
