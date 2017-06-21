#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#include "mlp.h"

long double* alloc_vector( int length ){
	long double* ans = (long double*)malloc(sizeof(long double)*length );
	return ans;
}

long double** alloc_matrix( int row, int col ){
	long double** ans = (long double**)malloc(sizeof(long double*)*row);
	int i;
	for( i=0; i<row; i++ )
		ans[i] = (long double*)malloc(sizeof(long double)*col);
	return ans;
}

void free_matrix( long double** mat, int row, int col ){
	int i;
	for( i=0; i<row; i++ )
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

model* build_model( int hidden_length, int output_length, int input_length,
		  long double(*function)(long double) , long double(*derivative)(long double)  ){

	model*  m = (model*)malloc(sizeof(model));
	m->hidden_length = hidden_length;
	m->input_length = input_length;
	m->output_length = output_length;
	m->function = function;
	m->derivative = derivative;
	m->hidden_layer = alloc_matrix( m->hidden_length, m->input_length + 1 );
	m->output_layer = alloc_matrix( m->output_length, m->hidden_length + 1 );
	fill( m->hidden_layer, m->hidden_length, m->input_length + 1 );
	fill( m->output_layer, m->output_length, m->hidden_length + 1 );

	return m;
}

void free_model( model* m ){
	free_matrix( m->hidden_layer, m->hidden_length, m->input_length + 1 );
	free_matrix( m->output_layer, m->output_length, m->hidden_length + 1 );
	free( m );
}

long double sigmoid( long double gamma ){
        if( gamma < 0 )
                return 1.0 - 1.0/(1.0 + pow( M_E ,gamma));
        else
                return 1.0/(1.0 + pow( M_E, -gamma));
}

long double sigmoid_derivative( long double gamma ){
	return ( sigmoid(gamma)*( 1.0 - sigmoid(gamma) ) );
}

void compute_net( long double** layer, int nrow, int ncol,
 		  long double* vetor, long double* net ){
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

	long double* net_hidden = alloc_vector( m->hidden_length );
	long double* net_output = alloc_vector(m->output_length );
	long double* f_hidden = alloc_vector( m->hidden_length );
	long double* f_output = alloc_vector(m->output_length );

	compute_net( m->hidden_layer, m->hidden_length, m->input_length + 1, vetor, net_hidden );
	apply( m->function, f_hidden, net_hidden, m->hidden_length );
	compute_net( m->output_layer, m->output_length, m->hidden_length + 1, f_hidden , net_output );
	apply( m->function, f_output, net_output, m->output_length );

	free( net_hidden );
	free( f_hidden );
	free( net_output );
	free( f_output );

	return f_output; // now f_output!

}

long double* copy( long double* vetor, int length ){
	long double* ans = (long double*)malloc(sizeof(long double)*length);
	int i;
	for( i=0; i<length; i++ )
		ans[i] = vetor[i];
	return ans;
}

void apply( long double(*function)(long double), long double* vetor, long double* source, int length ){
	int i;
	for( i=0; i<length; i++ ){
		vetor[i] = function( source[i] );
	}
}

void printvector( long double* vec, int length ){
	int i=0;
	for( i=0; i<length; i++ ){
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

void training( model* m, long double** sample, long double** expected, int length, long double precision, long double rate, int limit ){
	long double error, delta;
	int it = 0, j, i, k,l;
	long double *temp, *delta_o, *delta_h, *delta_e ,*df_hidden, *df_output, *net_hidden, *f_hidden, *net_output, *f_output;

	delta_e = alloc_vector( m->output_length );
	delta_o = alloc_vector( m->output_length );
	delta_h = alloc_vector( m->hidden_length );
	net_hidden = alloc_vector( m->hidden_length );
	net_output = alloc_vector( m->output_length );
	f_hidden = alloc_vector( m->hidden_length );
	df_hidden = alloc_vector( m->hidden_length );
	f_output = alloc_vector( m->output_length );
	df_output = alloc_vector( m->output_length );

	do{
		error = 0;
		for( j=0; j < length; j++){

			compute_net( m->hidden_layer, m->hidden_length, m->input_length + 1, sample[j], net_hidden );
			apply( m->function, f_hidden, net_hidden, m->hidden_length );
			apply( m->derivative, df_hidden, net_hidden, m->hidden_length );
			
			compute_net( m->output_layer, m->output_length, m->hidden_length + 1, f_hidden , net_output );
			apply( m->function, f_output, net_output, m->output_length );
			apply( m->derivative, df_output, net_output, m->output_length );

			for( i=0; i<m->output_length; i++ ){
				delta_e[i] = ( -f_output[i] + expected[j][i] );
				error += delta_e[i]*delta_e[i];
				delta_o[i] = delta_e[i] * df_output[i];
			}

			for( i=0; i<m->hidden_length; i++ ){
			    long double aux = 0;
			    for( k=0; k < m->output_length; k++ )
				aux += m->output_layer[k][i] * delta_o[k] ;
			    delta_h[i] = aux * df_hidden[i];
			}
			// atualizing hidden layer
			for( i=0; i<m->hidden_length; i++ ){
				for( k=0; k< m->input_length ; k++ )
					m->hidden_layer[i][k] += delta_h[i] * rate * sample[j][k];
				m->hidden_layer[i][ m->input_length ] += delta_h[i] * rate;
			}

			for( i=0; i<m->output_length; i++ ){
				for( k=0; k< m->hidden_length; k++ )
					m->output_layer[i][k] += delta_o[i] * rate * f_hidden[k];
				m->output_layer[i][ m->hidden_length ] += delta_o[i] * rate;
			}

		}
		error = error / length ;
		it++;
		printf("%d average error is %Lf percent\n", it ,error*100 ); 
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

void zero_matrix( long double** mat, int row, int col ){
	int i, j;
	for( i=0; i<row; i++ ){
		for( j=0; j<col; j++ ){
			mat[i][j] = 0;
		}
	}
}

long double** predict( model* m,  long double** queries, int length ){
	long double** answers = (long double**)malloc(sizeof(long double*)*length );
	int i;
	for( i=0; i<length; i++ ){
		long double* temp = run( m,  queries[i] );
		answers[i] = temp;
	}
	return answers;
}

int argmax( long double* vetor, int length ){
	int i, j = 0;
	for( i=1; i < length; i ++ )
		if( vetor[i] > vetor[j] )
			j = i;
	return j;
}
