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
#include "kernel_system_assembler.h"

int main( int argc, char* argv[])
{
	if (argc!=7)
	{
		printf("./hmglib_test <Nx> <Ny> <k> <c_leaf> <exponent of epsilon> <eta>\n");
		return 0;
	}

	// set dimension and morton code size
	int dim = 2;
	int bits = 32;  // dim == 2 => 32;  dim == 3 => 20

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

        // set batching sizes
        data.max_batched_dense_size = 8192;
        data.max_batched_aca_size = 65536;

	// generate random points for testing purpose
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	for (int d = 0; d < dim; d++ )
	{
		curandGenerateUniformDouble(gen, data.coords_d[0][d], point_count[0]);
		curandGenerateUniformDouble(gen, data.coords_d[1][d], point_count[1]);
	}
	curandDestroyGenerator(gen);


	// run setup of H matrix
	setup_h_matrix(&data);


        // setup kernel matrix assembler
	double regularization = 0.0;
	struct gaussian_kernel_system_assembler assem;
        struct gaussian_kernel_system_assembler** assem_d_p;
        cudaMalloc((void***)&assem_d_p, sizeof(struct gaussian_kernel_system_assembler*));
        create_gaussian_kernel_system_assembler_object(assem_d_p, regularization);
        struct gaussian_kernel_system_assembler* assem_d;
        cudaMemcpy(&assem_d, assem_d_p, sizeof(struct gaussian_kernel_system_assembler*), cudaMemcpyDeviceToHost);
	data.assem = assem_d;

	// precomputation of ACA
	precompute_aca(&data);

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

	thrust::fill(y_ptr, y_ptr+point_count[0], 0.0);
	thrust::fill(y_test_ptr, y_test_ptr+point_count[0], 0.0);


	// generate random vector x
	curandGenerator_t vec_gen;
	curandCreateGenerator(&vec_gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandGenerateUniformDouble(vec_gen, x, point_count[1]);
	curandDestroyGenerator(vec_gen);

	// apply full mvp for testing puposes
	apply_full_mvp(x, y_test, &data);

	// apply H matrix to same vector
	apply_h_matrix_mvp(x, y, &data);

	// compute and print relative error of H matrix approximation (wrt. mvp)	
	double y_test_norm = sqrt(thrust::inner_product(y_test_ptr, y_test_ptr+point_count[0], y_test_ptr, 0.0));
	thrust::transform(y_test_ptr, y_test_ptr+point_count[0], y_ptr, y_test_ptr, thrust::minus<double>());
	double abs_error = sqrt(thrust::inner_product(y_test_ptr, y_test_ptr+point_count[0], y_test_ptr, 0.0));
	double error = abs_error/y_test_norm;	
	printf("Relative error in H matrix matrix-vector product: %le\n", error);

	
	// cleanup of vectors
	cudaFree(y_test);
	cudaFree(x);
	cudaFree(y);

	// cleanup of assembler
	destroy_gaussian_kernel_system_assembler_object(assem_d_p);;
	cudaFree(assem_d_p);

	// cleanup of H matrix
	destroy_h_matrix_data(&data);
}
