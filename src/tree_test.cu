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
#include "hmglib.h"

int main( int argc, char* argv[])
{

//	int dim = 3;
//	int bits = 20;
	int dim = 2;
	int bits = 32;

	int point_count[2];
	point_count[0] = atoi(argv[1]);
	point_count[1] = atoi(argv[2]);

//	int point_count=16;
//	int point_count=20;
//	int point_count = atoi(argv[1]);
//	int point_count=4000000;
//	int point_count=29000000;

	if (argc!=7)
	{
		printf("./tree_test <Nx> <Ny> <k> <c_leaf> <exponent of epsilon> <eta>\n");
		return 0;
	}

	struct h_matrix_data data;


	init_h_matrix_data(&data, point_count, dim, bits);

//	data.eta=10.0;
//	data.eta=0.5;
	data.eta=atof(argv[6]);
	data.max_level=50; // DEBUG
//	data.c_leaf=1024;
	data.c_leaf=atoi(argv[4]);
//	data.c_leaf=4;

	data.dim = dim;
	data.point_count[0] = point_count[0];
	data.point_count[1] = point_count[1];

	data.k = atoi(argv[3]);
	data.epsilon = pow(10.0, atoi(argv[5]));




	// generate random points
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	for (int d = 0; d < dim; d++ )
	{
		curandGenerateUniformDouble(gen, data.coords_d[0][d], point_count[0]);
		curandGenerateUniformDouble(gen, data.coords_d[1][d], point_count[1]);
//		cudaMemcpy(data.coords_d[1][d], data.coords_d[0][d], point_count[0]*sizeof(double), cudaMemcpyDeviceToDevice);
	}
	curandDestroyGenerator(gen);

//	set_2d_test_set<<<1,16>>>(data->points_d);

//	set_3d_test_set<<<1,64>>>(data->points_d);



	setup_h_matrix(&data);

	double* x;
	double* y;
	double* y_test;
	cudaMalloc((void**)&x, point_count[1]*sizeof(double));
	cudaMalloc((void**)&y, point_count[0]*sizeof(double));
	cudaMalloc((void**)&y_test, point_count[0]*sizeof(double));

	// generate random vector x
	curandGenerator_t vec_gen;
	curandCreateGenerator(&vec_gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandGenerateUniformDouble(vec_gen, x, point_count[1]);
	curandDestroyGenerator(vec_gen);

	// apply dense mvp for testing puposes
	apply_dense_mvp(x, y_test, &data);

	
	apply_h_matrix_mvp(x, y, &data);


	thrust::device_ptr<double> y_ptr(y);
	thrust::device_ptr<double> y_test_ptr(y_test);

//	printf("x,y, y_test\n");
//	print_double(x, point_count);
//	print_double(y, point_count);
//	print_double(y_test, point_count);

	
	double y_test_norm = sqrt(thrust::inner_product(y_test_ptr, y_test_ptr+point_count[0], y_test_ptr, 0.0));
	thrust::transform(y_test_ptr, y_test_ptr+point_count[0], y_ptr, y_test_ptr, thrust::minus<double>());
	double abs_error = sqrt(thrust::inner_product(y_test_ptr, y_test_ptr+point_count[0], y_test_ptr, 0.0));
	
	double error = abs_error/y_test_norm;	

	printf("Error: %le\n", error);
//
	cudaFree(y_test);

	cudaFree(x);
	cudaFree(y);





/*	int tree_work_queue_size = 100*point_count; // DEBUG
	int mat_vec_work_queue_size = 100*point_count ; // DEBUG

	struct work_queue **tree_work_queue_h = new struct work_queue*[max_level];
	struct work_queue **tree_work_queue = new struct work_queue*[max_level];

	for (int l=0; l<max_level; l++)
	{
		allocate_work_queue(&(tree_work_queue_h[l]), &(tree_work_queue[l]), tree_work_queue_size);
		// calculate GPU thread configuration
		int invalidate_block_size = 512;
		int invalidate_grid_size = (tree_work_queue_size + (invalidate_block_size - 1)) / invalidate_block_size;
		invalidate_work_items<<<invalidate_grid_size,invalidate_block_size>>>(tree_work_queue_h[l]->data, tree_work_queue_size);
	}
	
	struct work_queue tmp;
	cudaMemcpy(&tmp, tree_work_queue[0], sizeof(struct work_queue), cudaMemcpyDeviceToHost);
	printf("work size: %d\n", tmp.queue_size);


	struct work_queue **tree_work_queue_dev;
	cudaMalloc((void**)&tree_work_queue_dev, max_level*sizeof(struct work_queue*));
	
	cudaMemcpy(tree_work_queue_dev, tree_work_queue, max_level*sizeof(struct work_queue*), cudaMemcpyHostToDevice);
	
	struct work_queue *mat_vec_work_queue_h;
	struct work_queue *mat_vec_work_queue;

	allocate_work_queue(&mat_vec_work_queue_h, &mat_vec_work_queue, mat_vec_work_queue_size);
		
	init_tree_work_queue_root<<<1,1>>>(tree_work_queue_dev, 0, point_count-1, 0, point_count-1);
	cudaThreadSynchronize();
	checkCUDAError("init_tree_work_queue_root");

	traverse_single_single_queue<<<1, 4>>>(mat_vec_work_queue, tree_work_queue_dev, morton_d, morton_d, points_d, points_d, eta, max_level, c_leaf);
//	traverse<<<grid_size, block_size>>>(mat_vec_work_queue, tree_work_queue_dev, morton_d, morton_d, points_d, points_d, eta, max_level, c_leaf);
	cudaThreadSynchronize();
	checkCUDAError("traverse");

	cudaMemcpy(mat_vec_work_queue_h, mat_vec_work_queue, sizeof(struct work_queue), cudaMemcpyDeviceToHost);
	checkCUDAError("memcpy");

	struct work_item* items = new struct work_item[mat_vec_work_queue_size];

	printf("memcpy: %p %p %lu\n",items, mat_vec_work_queue_h->data, mat_vec_work_queue_size*sizeof(struct work_queue));

	int real_mat_vec_work_queue_size = mat_vec_work_queue_h->tail - mat_vec_work_queue_h->head;

	printf("data %p\n",mat_vec_work_queue_h->data);
	printf("head %p\n",mat_vec_work_queue_h->head);
	printf("tail %p\n",mat_vec_work_queue_h->tail);
	printf("size %ld\n",mat_vec_work_queue_h->tail-mat_vec_work_queue_h->head);

	cudaMemcpy(items, mat_vec_work_queue_h->data, real_mat_vec_work_queue_size*sizeof(struct work_item), cudaMemcpyDeviceToHost);
	checkCUDAError("memcpy");

	for (int it=0; it<real_mat_vec_work_queue_size; it++)
	{
		printf("item %d: %d %d %d %d, %d\n",it, items[it].set1_l, items[it].set1_u, items[it].set2_l, items[it].set2_u, items[it].work_type);
	}
	delete [] items;

	delete_work_queue(mat_vec_work_queue_h, mat_vec_work_queue);

	cudaFree(tree_work_queue_dev);

	for (int l=0; l<max_level; l++)
		delete_work_queue(tree_work_queue_h[l], tree_work_queue[l]);
	
*/

	
	
	
//	// print ordered points
//	print_points(points_d);

//	// write points to file
//	char file_name[2000];
//	sprintf(file_name,"points.dat");
//	write_points(points_d,file_name);
	

	destroy_h_matrix_data(&data);


}
