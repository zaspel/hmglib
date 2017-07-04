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
#include "morton.h"
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <curand.h>
#include <thrust/sort.h>
#include "tree.h"
#include "linear_algebra.h"
#include <thrust/inner_product.h>
#include "helper.h"

struct h_matrix_data
{
	// point coordinates in dim dimensions
	double** coords_d[2];
	
	// extremal values per dimension
	double* max_per_dim_d[2];
	double* min_per_dim_d[2];

	// device pointer for dimension-wise access
	double** coords_device[2];

	// morton codes
	uint64_t* code_d[2];

	// point set
	struct point_set* points_d[2];

	// morton code
	struct morton_code* morton_d[2];

	// order of points following Z curve
	uint64_t* order[2];

	int dim;

	double eta;

	int max_level;

	int c_leaf;

	int mat_vec_data_count;

        int max_elements_in_array; 
        int max_elements_in_mat_vec_data_array;

        struct work_item** mat_vec_data;
        int mat_vec_data_array_size;

	int k;

	double epsilon;

	mat_vec_data_info mat_vec_info;

	int point_count[2];
};

void init_h_matrix_data(struct h_matrix_data* data, int point_count[2], int dim, int bits)
{
	for (int i=0; i<2; i++)
	{
		// allocating memory for point_count coordinates in dim dimensions
		data->coords_d[i] = new double*[dim];
		for (int d = 0; d < dim; d++)
		{
			cudaMalloc((void**)&(data->coords_d[i][d]), point_count[i]*sizeof(double));
			checkCUDAError("cudaMalloc");
			}
	
		// allocating memory for extremal values per dimension
		cudaMalloc((void**)&(data->max_per_dim_d[i]), dim*sizeof(double));
		cudaMalloc((void**)&(data->min_per_dim_d[i]), dim*sizeof(double));
	
		// generating device pointer that holds the dimension-wise access
		cudaMalloc((void**)&(data->coords_device[i]), dim*sizeof(double*));
		cudaMemcpy(data->coords_device[i], data->coords_d[i], dim*sizeof(double*), cudaMemcpyHostToDevice);
	
		// allocationg memory for morton codes
		cudaMalloc((void**)&(data->code_d[i]), point_count[i]*sizeof(uint64_t));
		checkCUDAError("cudaMalloc");
	
		// setting up data strcture for point set
		cudaMalloc((void**)&(data->points_d[i]), sizeof(struct point_set));
		init_point_set<<<1,1>>>(data->points_d[i], data->coords_device[i], dim, data->max_per_dim_d[i], data->min_per_dim_d[i], point_count[i]);
	
		// setting up data structure for morton code
		cudaMalloc((void**)&(data->morton_d[i]), sizeof(struct morton_code));
		init_morton_code<<<1,1>>>(data->morton_d[i], data->code_d[i], dim, bits, point_count[i]);
	}
}

void setup_h_matrix(struct h_matrix_data* data)
{
	// compute extremal values for the point set
	compute_minmax(data->points_d[0]);
	compute_minmax(data->points_d[1]);
	
	// calculate GPU thread configuration	
	int block_size = 512;
	int grid_size = (max(data->point_count[0],data->point_count[1]) + (block_size - 1)) / block_size;

	printf("gs %d\n",grid_size);

	// generate morton codes
	TIME_start;
	get_morton_code<<<grid_size, block_size>>>(data->points_d[0], data->morton_d[0]);
	get_morton_code<<<grid_size, block_size>>>(data->points_d[1], data->morton_d[1]);
	TIME_stop("get_morton_3d");
	checkCUDAError("get_morton_code");

//	print_points_with_morton_codes(data->points_d, data->morton_d);


	// find ordering of points following Z curve
	cudaMalloc((void**)&(data->order[0]), data->point_count[0]*sizeof(uint64_t));
	cudaMalloc((void**)&(data->order[1]), data->point_count[1]*sizeof(uint64_t));
	get_morton_ordering(data->points_d[0], data->morton_d[0], data->order[0]);
	get_morton_ordering(data->points_d[1], data->morton_d[1], data->order[1]);

//	print_points_with_morton_codes(data->points_d, data->morton_d);

	// reorder points following the morton code order
//	TIME_start;
	reorder_point_set(data->points_d[0], data->order[0]);
	reorder_point_set(data->points_d[1], data->order[1]);
//	TIME_stop("reorder_point_set");

	struct work_item root_h;
	root_h.set1_l = 0;
	root_h.set1_u = data->point_count[0] - 1;
	root_h.set2_l = 0;
	root_h.set2_u = data->point_count[1] - 1;

	data->mat_vec_data_count = 0;  // will be filled with the size number of mat_vec_data entries

/*
	int max_elements_in_array = -1; // TODO
	int max_elements_in_mat_vec_data_array = point_count*7; // TODO

	struct work_item* mat_vec_data;
	cudaMalloc((void**)&mat_vec_data, max_elements_in_mat_vec_data_array*sizeof(struct work_item));

	TIME_start;
	traverse_with_dynamic_arrays(root_h, mat_vec_data, &mat_vec_data_count, morton_d, morton_d, points_d, points_d, eta, max_level, c_leaf, max_elements_in_array);
	TIME_stop("traverse_with_arrays");

	printf("mat_vec_data_count: %d\n", mat_vec_data_count);
	print_work_items(mat_vec_data, mat_vec_data_count);

	cudaFree(mat_vec_data);
*/


	data->max_elements_in_array = -1; // TODO
	data->max_elements_in_mat_vec_data_array = -1; // TODO

	data->mat_vec_data = new struct work_item*[1];
	data->mat_vec_data_array_size = 1048576;
	cudaMalloc((void**)data->mat_vec_data, data->mat_vec_data_array_size*sizeof(struct work_item));

	TIME_start;
	traverse_with_dynamic_arrays_dynamic_output(root_h, data->mat_vec_data, &(data->mat_vec_data_count), &(data->mat_vec_data_array_size), data->morton_d[0], data->morton_d[1], data->points_d[0], data->points_d[1], data->eta, data->max_level, data->c_leaf, data->max_elements_in_array);
	TIME_stop("traverse_with_arrays");

//	printf("mat_vec_data_count: %d\n", mat_vec_data_count);
//	print_work_items(*mat_vec_data, mat_vec_data_count);
//	char file_name[1000];
//	sprintf(file_name, "mat_vec_data.txt");
//	write_work_items(file_name, *mat_vec_data, mat_vec_data_count);

	organize_mat_vec_data(*(data->mat_vec_data), data->mat_vec_data_count, &(data->mat_vec_info));



}

void apply_h_matrix_mvp(double* x, double* y, struct h_matrix_data* data)
{
	reorder_vector(x, data->point_count[1], data->order[1]);	

	printf("dort\n");
	TIME_start;
	sequential_h_matrix_mvp(x, y, *(data->mat_vec_data), &(data->mat_vec_info), data->points_d[0], data->points_d[1], data->eta, data->epsilon, data->k);
	TIME_stop("sequential_h_matrix");

	reorder_back_vector(x, data->point_count[1], data->order[1]);
	reorder_back_vector(y, data->point_count[0], data->order[0]);
}


void destroy_h_matrix_data(struct h_matrix_data* data)
{

	cudaFree(*(data->mat_vec_data));

	for (int i=0; i<2; i++)
	{
		cudaFree(data->order[i]);
	
		// freeing memory for morton codes
		cudaFree(data->code_d[i]);
	
		// freeing coordinates memory
		for (int d = 0; d < data->dim; d++)
		{
			cudaFree(data->coords_d[i][d]);
		}
		delete [] data->coords_d[i];
	}
}



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

	TIME_start;
	struct work_item test_mat_vec_data;
	test_mat_vec_data.set1_l = 0;
	test_mat_vec_data.set1_u = point_count[0] - 1;
	test_mat_vec_data.set2_l = 0;
	test_mat_vec_data.set2_u = point_count[1] - 1;
	double* test_matrix;
	cudaMalloc((void**)&test_matrix, point_count[0]*point_count[1]*sizeof(double));
	checkCUDAError("cudaMalloc");
	int block_size1 = 512;
	fill_matrix<<<(point_count[0]*point_count[1] + (block_size1 - 1)) / block_size1, block_size1>>>(test_matrix, test_mat_vec_data, data.points_d[0], data.points_d[1], point_count[0], point_count[1]);
	cudaThreadSynchronize();
	checkCUDAError("fill_matrix");
    cublasStatus_t stat;
    cublasHandle_t handle;
    stat = cublasCreate(&handle);

	reorder_vector(x,point_count[1],data.order[1]);

	// matrix-vector-product
	double one;
	double zero;
	one = 1.0;
	zero = 0.0;
	stat = cublasDgemv(handle, CUBLAS_OP_N, point_count[0], point_count[1], &one, test_matrix, point_count[0], x, 1, &zero, y_test, 1);
	if (stat!=CUBLAS_STATUS_SUCCESS)
	{
		printf("dgemv did not succeed...\n");
		exit(1);
	}
    cublasDestroy(handle);
    TIME_stop("dense_mvp");

	reorder_back_vector(x,point_count[1],data.order[1]);
	reorder_back_vector(y_test,point_count[0],data.order[0]);


    cudaFree(test_matrix);

	
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
