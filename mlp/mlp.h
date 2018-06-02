#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

// represents a neuron
typedef struct neuron{
	int size;
	long double* weigths;
	long double bias;
} neuron;

// represents a neural network with a single hidden layer
typedef struct model{
	int hidden_length, output_length, input_length;
	neuron** hidden_layer;
	neuron** output_layer;
	long double(*function)( long double );
	long double(*derivative)( long double );
} model;

long double* alloc_vector( int length );

long double** alloc_matrix( int row, int col );

void free_matrix( long double** mat, int row, int col );

void print_vector( long double* vec, int length );

void print_matrix( long double** matrix, int nrow, int ncol );

void zero_matrix( long double** mat, int row, int col );

// fills a layer with random numbers
void fill( neuron** layer, int number_of_neuron, int number_of_weigths );

// returns the position of the biggest element of an 1-D array
int argmax( long double* vetor, int length );

neuron* alloc_neuron( int number_of_weigths );

neuron** alloc_layer( int number_of_neurons, int number_of_weigths );

//allocs the momery of the neural network and fills its layers with ramdom numbers
model* build_model( int hidden_length, int output_length, int input_length,
		   long double(*function)(long double) , long double(*derivative)(long double)  );

void free_neuron( neuron* myneuron );

void free_layer( neuron** layer, int number_of_neurons );

void free_model( model* m );

long double sigmoid( long double x );

long double sigmoid_derivative( long double x );

// applies the given function to elements from source and saves results to output vector. expects length of the vector.
void apply( long double(*function)(long double), long double* vetor, long double* source, int len );

void compute_net( neuron** layer, int number_of_neurons, int number_of_weigths, 
		long double* vetor, long double* net );

long double* run( model* m, long double* input );

void predict( model* m, long double** queries, int length, FILE* fp );

// trains a given model until it reaches a limit of iterations or reaches
// a value less-equal to a given treshold
void training( model* m, long double** sample, long double** expected,
		int len, long double precision, long double rate, int limit );
