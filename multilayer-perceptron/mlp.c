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

void print_vector( long double* vec, int length ){
	int i=0;
	for( i=0; i<length; i++ )
		printf("%Lf ", vec[i] );
	printf("\n");
}

void print_matrix( long double** matrix, int nrow, int ncol ){
	int i, j;
	for( i=0; i<nrow; i++ )
		for( j=0; j<ncol; j++ )
			printf("%Lf ", matrix[i][j] );
		printf("\n");
	printf("\n");
}

void zero_matrix( long double** mat, int row, int col ){
	int i, j;
	for( i=0; i<row; i++ )
		for( j=0; j<col; j++ )
			mat[i][j] = 0;
}

void fill( neuron** layer, int number_of_neuron, int number_of_weigths ){
	srand(time(NULL));
	int i, j;
	for( i=0; i<number_of_neuron; i++ ){
		for( j=0; j<number_of_weigths; j++ )
			layer[i]->weigths[j] = (float)( rand()%1000 )/1000.0 - 0.5;
		layer[i]->bias = (float)( rand()%1000 )/1000.0 - 0.5;
	}
}


int argmax( long double* vetor, int length ){
	int i, j = 0;
	for( i=1; i < length; i ++ )
		if( vetor[i] > vetor[j] )
			j = i;
	return j;
}

neuron* alloc_neuron( int number_of_weigths ){
	neuron* answer = (neuron*)malloc(sizeof(neuron)*number_of_weigths );
	answer->weigths = alloc_vector( number_of_weigths );
	return answer;
}

neuron** alloc_layer( int number_of_neurons, int number_of_weigths ){
	neuron** answer = (neuron**)malloc(sizeof(neuron)*number_of_neurons);
	int i;
	for( i=0; i<number_of_neurons; i++ )
		answer[i] = alloc_neuron( number_of_weigths );
	return answer;
}

model* build_model( int hidden_length, int output_length, int input_length,
		  long double(*function)(long double) , long double(*derivative)(long double)){

	model*  m = (model*)malloc(sizeof(model));
	m->hidden_length = hidden_length;
	m->input_length = input_length;
	m->output_length = output_length;
	m->function = function;
	m->derivative = derivative;
	m->hidden_layer = alloc_layer( m->hidden_length, m->input_length );
	m->output_layer = alloc_layer( m->output_length, m->hidden_length );
	fill( m->hidden_layer, m->hidden_length, m->input_length );
	fill( m->output_layer, m->output_length, m->hidden_length );

	return m;
}

void free_neuron( neuron* myneuron ){
	free( myneuron->weigths );
	free( myneuron );
}

void free_layer( neuron** layer, int number_of_neurons ){
	int i, j;
	for( i=0; i<number_of_neurons; i++)
		free_neuron( layer[i] );
	free( layer );
}

void free_model( model* m ){
	free_layer( m->hidden_layer, m->hidden_length );
	free_layer( m->output_layer, m->output_length );
	free( m );
}

long double sigmoid( long double x ){
        if( x < 0 )
                return 1.0 - 1.0/(1.0 + pow( M_E , x ));
        else
                return 1.0/(1.0 + pow( M_E, -x ));
}

long double sigmoid_derivative( long double x ){
	return ( sigmoid( x )*( 1.0 - sigmoid( x ) ) );
}

void apply( long double(*function)(long double),
		long double* output, long double* source, int length ){
	
	int i;
	for( i=0; i<length; i++ )
		output[i] = function( source[i] );
}

void compute_net( neuron** layer, int number_of_neurons, int number_of_weigths,
 		long double* vetor, long double* net ){

	int i, j;
	for( i=0; i<number_of_neurons; i++ ){
		net[i] = 0;
		for( j=0; j<number_of_weigths; j++ )
			net[i] += layer[i]->weigths[j]*vetor[j];
		net[i] += layer[i]->bias;
	}
}

long double* run( model* m, long double* input ){
	int i;

	long double* net_hidden = alloc_vector( m->hidden_length );
	long double* net_output = alloc_vector( m->output_length );
	long double* f_hidden = alloc_vector( m->hidden_length );
	long double* f_output = alloc_vector( m->output_length );

	compute_net( m->hidden_layer, m->hidden_length, m->input_length, input, net_hidden );
	apply( m->function, f_hidden, net_hidden, m->hidden_length );
	compute_net( m->output_layer, m->output_length, m->hidden_length, f_hidden , net_output );
	apply( m->function, f_output, net_output, m->output_length );

	free( net_hidden );
	free( f_hidden );
	free( net_output );
	free( f_output );

	return f_output;
}

void predict( model* m, long double** queries, int length, FILE* fp ){
	int i;

	long double* net_hidden = alloc_vector( m->hidden_length );
	long double* net_output = alloc_vector(m->output_length );
	long double* f_hidden = alloc_vector( m->hidden_length );
	long double* f_output = alloc_vector(m->output_length );

	for( i=0; i<length; i++ ){
		compute_net( m->hidden_layer, m->hidden_length, m->input_length, queries[i], net_hidden );
		apply( m->function, f_hidden, net_hidden, m->hidden_length );
		compute_net( m->output_layer, m->output_length, m->hidden_length, f_hidden , net_output );
		apply( m->function, f_output, net_output, m->output_length );
		fprintf(fp, "%d,%d\n", i+1, argmax( f_output, m->output_length ) );
	}

	free( net_hidden );
	free( f_hidden );
	free( net_output );
	free( f_output );
}

void training( model* m, long double** sample, long double** expected,
		int length, long double precision, long double rate, int limit ){
	long double error, delta;
	int it = 0, j, i, k,l;
	long double *temp, *delta_o, *delta_h, *delta_e ,*df_hidden;
	long double *df_output, *net_hidden, *f_hidden, *net_output, *f_output;

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

			compute_net( m->hidden_layer, m->hidden_length, m->input_length, sample[j], net_hidden );
			apply( m->function, f_hidden, net_hidden, m->hidden_length );
			apply( m->derivative, df_hidden, net_hidden, m->hidden_length );
			
			compute_net( m->output_layer, m->output_length, m->hidden_length, f_hidden , net_output );
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
					aux += m->output_layer[k]->weigths[i] * delta_o[k] ;
			    delta_h[i] = aux * df_hidden[i];
			}
			// atualizing hidden layer
			for( i=0; i<m->hidden_length; i++ ){
				for( k=0; k< m->input_length ; k++ )
					m->hidden_layer[i]->weigths[k] += delta_h[i] * rate * sample[j][k];
				m->hidden_layer[i]->bias += delta_h[i] * rate;
			}
			// atualizing output
			for( i=0; i<m->output_length; i++ ){
				for( k=0; k< m->hidden_length; k++ )
					m->output_layer[i]->weigths[k] += delta_o[i] * rate * f_hidden[k];
				m->output_layer[i]->bias += delta_o[i] * rate;
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
