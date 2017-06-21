#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

/*
typedef struct neuron{
	int size;
	double* weigth;
	double bias;
} neuron;

typedef struct model{
	int hidden_length, output_length, input_length;
	neuron* hidden_layer;
	neuron* output_layer;
	long double(*function)( long double );
	long double(*derivative)( long double );
} model;

*/

typedef struct model{
	int hidden_length, output_length, input_length;
	long double** hidden_layer;
	long double** output_layer;
	long double(*function)( long double );
	long double(*derivative)( long double );
}model;

long double* alloc_vector( int len );

long double** alloc_matrix( int row, int col );

void free_matrix( long double** mat, int row, int col );

void fill( long double** matriz, int row, int col );

model* build_model( int hidden_length, int output_length, int input_length,
		   long double(*function)(long double) , long double(*derivative)(long double)  );

void free_model( model* m );

long double sigmoid( long double gamma );

long double sigmoid_derivative( long double gamma );

void compute_net( long double** layer, int nrow, int ncol , long double* vetor, long double* net );

long double* run( model* m, long double* vetor );

long double* copy( long double* vetor, int len );

void apply( long double(*function)(long double), long double* vetor, long double* source, int len );

void print_vector( long double* vec, int len );

void print_matrix( long double** vec, int nrow, int ncol );

void training( model* m, long double** sample, long double** expected, int len, long double precision, long double rate, int limit );

void zero_matrix( long double** mat, int row, int col );

void predict( model* m, long double** queries, int length, FILE* fp );

int argmax( long double* vetor, int length );
