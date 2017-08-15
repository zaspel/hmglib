// Copyright (C) 2016 Peter Zaspel
//
// This file is part of hmglib.
//
// hmglib is free software: you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option) any
// later version.
//
// hmglib is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
// A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
// details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with hmglib.  If not, see <http://www.gnu.org/licenses/>.

#include <stdio.h>
#include <curand.h>
#include <thrust/inner_product.h>
#include <thrust/fill.h>
#include "hmglib.h"
#include <gsl/gsl_qrng.h>

void gen_halton_points(double** point_set, int dim, int point_count)
{
        // allocate temporary CPU point set
	double** point_set_h = new double*[dim];
        for (int d=0; d<dim; d++)
                point_set_h[d] = new double[point_count];

	// allocate random number generator
        gsl_qrng * q = gsl_qrng_alloc (gsl_qrng_halton, dim);

	// allocate temp vector
        double* v = new double[dim];

	// generate points
        for (int i=0; i<point_count; i++)
        {
                gsl_qrng_get (q, v);
                for (int d=0; d<dim; d++)
                        point_set_h[d][i] = v[d];
        }

	// copy points to GPU
	for (int d=0; d<dim; d++)
		cudaMemcpy(point_set[d], point_set_h[d], sizeof(double)*point_count, cudaMemcpyHostToDevice);

	// cleanup
        gsl_qrng_free(q);
	delete [] v;
	for (int d=0; d<dim; d++)
		delete [] point_set_h[d];
	delete [] point_set_h;
}


int main( int argc, char* argv[])
{
	if (argc!=9)
	{
		printf("./tree_test <Nx> <Ny> <k> <c_leaf> <exponent of epsilon> <eta> <kernel_type> <dim>\n");
		return 0;
	}

	// set dimension and morton code size
	int dim = atoi(argv[8]);
	
	int bits;
	if (dim==2)
		bits = 32;
	else if (dim==3)
		bits = 20;
	else
	{
		printf("Dimension %d is not supported. Exiting...\n", dim);
		exit(1);
	}

	// set number of points
	int point_count[2];
	point_count[0] = atoi(argv[1]);
	point_count[1] = atoi(argv[2]);

	// data structure for H matrix data
	struct h_matrix_data data;

	// initialize data structure
	init_h_matrix_data(&data, point_count, dim, bits);

	// set balance	
	data.eta=atof(argv[6]);

	// set maximum level
	data.max_level=50; // DEBUG

	// set maximum leaf size
	data.c_leaf=atoi(argv[4]);

	// set rank in ACA
	data.k = atoi(argv[3]);

	// set threshold for ACA (currently not use)
	data.epsilon = pow(10.0, atoi(argv[5]));

	// set kernel
	data.kernel_type = atoi(argv[7]);

	// generate Halton sequence points (on CPU due to missing CURAND support for Halton sequences)
	double** coords_1_h = new double*[dim];
	double** coords_2_h = new double*[dim];
	for (int d=0; d<dim; d++)
	{
		coords_1_h[d] = new double[point_count[0]];
		coords_2_h[d] = new double[point_count[1]];
	}
	
	// generate Halton points in in both point sets
	gen_halton_points( data.coords_d[0], dim, point_count[0]);
	gen_halton_points( data.coords_d[1], dim, point_count[1]);

	// run setup of H matrix
	setup_h_matrix(&data);

//	// precomputation of ACA
//	precompute_aca(&data);

	// allocate vectors for H matrix vs. full matrix test
	double* x;
	double* y;
	double* y_test;
	cudaMalloc((void**)&x, point_count[1]*sizeof(double));
	cudaMalloc((void**)&y, point_count[0]*sizeof(double));
	cudaMalloc((void**)&y_test, point_count[0]*sizeof(double));
	
	// get thrust device_ptr
	thrust::device_ptr<double> y_ptr(y);
	thrust::device_ptr<double> y_test_ptr(y_test);

	// construct random number generator
	curandGenerator_t vec_gen;
	curandCreateGenerator(&vec_gen, CURAND_RNG_PSEUDO_DEFAULT);

	int trials = 10;

	double* errors = new double[trials];

	for (int i=0; i<trials; i++)
	{
	
		thrust::fill(y_ptr, y_ptr+point_count[0], 0.0);
		thrust::fill(y_test_ptr, y_test_ptr+point_count[0], 0.0);
	
		// generate random vector x
		curandGenerateUniformDouble(vec_gen, x, point_count[1]);
	
		// apply H matrix to vector without batching
		apply_h_matrix_mvp_without_batching(x, y_test, &data);
	
		// apply H matrix to same vector with batching
		apply_h_matrix_mvp(x, y, &data);
	
	}

	delete [] errors;


	// cleanup of random number generator
	curandDestroyGenerator(vec_gen);

	
	// cleanup of vectors
	cudaFree(y_test);
	cudaFree(x);
	cudaFree(y);

	// cleanup of H matrix
	destroy_h_matrix_data(&data);
}
