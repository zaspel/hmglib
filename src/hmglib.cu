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

#include "hmglib.h"

cudaEvent_t hmglib_start, hmglib_stop;
float hmglib_milliseconds;
#define TIME_start {cudaEventCreate(&hmglib_start); cudaEventCreate(&hmglib_stop); cudaEventRecord(hmglib_start);}
#define TIME_stop(a) {cudaEventRecord(hmglib_stop); cudaEventSynchronize(hmglib_stop); cudaEventElapsedTime(&hmglib_milliseconds, hmglib_start, hmglib_stop); printf("%s: Elapsed time: %lf ms\n", a, hmglib_milliseconds); }

__global__ void init_point_set(struct point_set* points, double** coords_device, unsigned int* point_ids_d, int dim, double* max_per_dim, double* min_per_dim, int size)
{
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx>=1) return;

        points->dim = dim;
        points->size = size;
        points->coords = coords_device;
        points->max_per_dim = max_per_dim;
        points->min_per_dim = min_per_dim;
	points->point_ids = point_ids_d;

        return;
}

__global__ void init_morton_code(struct morton_code* morton, uint64_t* code, int dim, int bits, int size)
{
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx>=1) return;

        morton->code = code;
        morton->dim = dim;
        morton->bits = bits;
        morton->size = size;
	morton->max_values = 0;  // just as initialization
	morton->min_values = 0;  // just as initialization

	return;
}

void init_h_matrix_data(struct h_matrix_data* data, int point_count[2], int dim, int bits)
{
	data->dim = dim;
	data->bits = bits;

	for (int i=0; i<2; i++)
	{
		// copying point_count
		data->point_count[i] = point_count[i];

		// allocating memory for point_count coordinates in dim dimensions
		data->coords_d[i] = new double*[dim];
		for (int d = 0; d < dim; d++)
		{
			cudaMalloc((void**)&(data->coords_d[i][d]), point_count[i]*sizeof(double));
			checkCUDAError("cudaMalloc");
			}
		
		// allocating memory for point ids
		cudaMalloc((void**)&(data->point_ids_d[i]), point_count[i]*sizeof(unsigned int));

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
		init_point_set<<<1,1>>>(data->points_d[i], data->coords_device[i], data->point_ids_d[i], dim, data->max_per_dim_d[i], data->min_per_dim_d[i], point_count[i]);
		cudaThreadSynchronize();
	
		// setting up data structure for morton code
		cudaMalloc((void**)&(data->morton_d[i]), sizeof(struct morton_code));
		init_morton_code<<<1,1>>>(data->morton_d[i], data->code_d[i], dim, bits, point_count[i]);
		cudaThreadSynchronize();
		checkCUDAError("init_morton_code");
	}

	// initialize fields for precomputation
	data->U = 0;
	data->V = 0;	
	data->dA = 0;

	// initialize magma
        magma_init();
        magma_int_t dev = 0;
        data->magma_queue=NULL;
        magma_queue_create( dev, &(data->magma_queue));
}

void setup_h_matrix(struct h_matrix_data* data)
{
	TIME_start;

	// compute extremal values for the point set
	compute_minmax(data->points_d[0]);
	compute_minmax(data->points_d[1]);
	
	// calculate GPU thread configuration	
	int block_size = 512;
	int grid_size = (max(data->point_count[0],data->point_count[1]) + (block_size - 1)) / block_size;

	// generate morton codes
	get_morton_code(data->points_d[0], data->morton_d[0], grid_size, block_size);
	get_morton_code(data->points_d[1], data->morton_d[1], grid_size, block_size);
	checkCUDAError("get_morton_code");

//	print_points_with_morton_codes(data->points_d[0], data->morton_d[0]);


	// find ordering of points following Z curve
	cudaMalloc((void**)&(data->order[0]), data->point_count[0]*sizeof(uint64_t));
	cudaMalloc((void**)&(data->order[1]), data->point_count[1]*sizeof(uint64_t));
	checkCUDAError("cuda_malloc");

	get_morton_ordering(data->points_d[0], data->morton_d[0], data->order[0]);
	get_morton_ordering(data->points_d[1], data->morton_d[1], data->order[1]);

//	print_points_with_morton_codes(data->points_d, data->morton_d);

	// reorder points following the morton code order
//	TIME_start;
	reorder_point_set(data->points_d[0], data->order[0]);
	reorder_point_set(data->points_d[1], data->order[1]);
//	TIME_stop("reorder_point_set");

	TIME_stop("data structure setup");


	TIME_start;
	
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

	traverse_with_dynamic_arrays_dynamic_output(root_h, data->mat_vec_data, &(data->mat_vec_data_count), &(data->mat_vec_data_array_size), data->morton_d[0], data->morton_d[1], data->points_d[0], data->points_d[1], data->eta, data->max_level, data->c_leaf, data->max_elements_in_array);

//	printf("mat_vec_data_count: %d\n", mat_vec_data_count);
//	print_work_items(*mat_vec_data, mat_vec_data_count);
//	char file_name[1000];
//	sprintf(file_name, "mat_vec_data.txt");
//	write_work_items(file_name, *mat_vec_data, mat_vec_data_count);

	organize_mat_vec_data(*(data->mat_vec_data), data->mat_vec_data_count, &(data->mat_vec_info));

	precompute_work_sizes(&(data->dense_work_size), &(data->aca_work_size), &(data->dense_batch_count), &(data->aca_batch_count), *(data->mat_vec_data), &(data->mat_vec_info), data->max_batched_dense_size, data->max_batched_aca_size);


	TIME_stop("tree construction and traversal");

}


void precompute_aca(struct h_matrix_data* data)
{
        TIME_start; 

	if (data->aca_work_size[0]>0)
		precompute_aca_for_h_matrix_mvp(*(data->mat_vec_data), &(data->mat_vec_info), data->points_d[0], data->points_d[1], data->eta, data->epsilon, data->k, &(data->U), &(data->V), data->assem);

        TIME_stop("precompute_aca");
}


void precompute_dense(struct h_matrix_data* data)
{
	TIME_start;

	if ((data->dense_batch_count > 2) || ((data->dense_batch_count == 2) && (data->dense_work_size[1]>0)))
	{
		printf("Dense block does not fit into memory provided by 'max_batched_dense_size'. Therefore, precomputing the dense blocks is impossible. Exiting ...\n");
		exit(1);
	}

//	size_t free_mem;
//	size_t total_mem;
//	cudaMemGetInfo(&free_mem, &total_mem);
//	printf("Memory free: %d / %d MB\n", (int)(free_mem/1024/1024), (int)(total_mem/1024/1024));
//	printf("Allocating %d MB of memory for precomputing.\n",(int)(sizeof(double)*data->max_batched_dense_size/1024/1024));
	
	cudaMalloc((void**)&(data->dA), sizeof(double)*data->max_batched_dense_size);

	cublasStatus_t stat;
	cublasHandle_t handle;
	stat = cublasCreate(&handle);

	precompute_batched_dense_magma(*(data->mat_vec_data), data->dense_work_size[0], data->points_d[0], data->points_d[1], stat, handle,data->assem, &(data->magma_queue), data->dA);

	TIME_stop("precompute_dense");	

//	cudaMemGetInfo(&free_mem, &total_mem);
//	printf("Memory free: %d / %d MB\n", (int)(free_mem/1024/1024), (int)(total_mem/1024/1024));

}


void apply_h_matrix_mvp(double* x, double* y, struct h_matrix_data* data)
{
	TIME_start;

	reorder_vector(x, data->point_count[1], data->order[1]);	

	bool use_precomputed_aca = data->U==0 ? false : true;
	bool use_precomputed_dense = data->dA==0 ? false : true;
	
	h_matrix_mvp(x, y, *(data->mat_vec_data), &(data->mat_vec_info), data->points_d[0], data->points_d[1], data->eta, data->epsilon, data->k, data->dA, data->U, data->V, data->assem, data->max_batched_dense_size, data->dense_batching_ratio, data->max_batched_aca_size, data->magma_queue, data->dense_work_size, data->aca_work_size, data->dense_batch_count, data->aca_batch_count, use_precomputed_aca, use_precomputed_dense);

	reorder_back_vector(x, data->point_count[1], data->order[1]);
	reorder_back_vector(y, data->point_count[0], data->order[0]);

        TIME_stop("apply_h_matrix_mvp");
}

void apply_h_matrix_mvp_without_batching(double* x, double* y, struct h_matrix_data* data)
{
        TIME_start;

        reorder_vector(x, data->point_count[1], data->order[1]);

        // ignoring precomputed data
	sequential_h_matrix_mvp_without_batching(x, y, *(data->mat_vec_data), &(data->mat_vec_info), data->points_d[0], data->points_d[1], data->eta, data->epsilon, data->k, data->assem);

        reorder_back_vector(x, data->point_count[1], data->order[1]);
        reorder_back_vector(y, data->point_count[0], data->order[0]);

        TIME_stop("apply_h_matrix_mvp_without_batching");
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
	
		cudaFree(data->point_ids_d[i]);

		cudaFree(data->morton_d[i]);
	
		cudaFree(data->points_d[i]);

		cudaFree(data->coords_device[i]);
		
		cudaFree(data->max_per_dim_d[i]);
		cudaFree(data->min_per_dim_d[i]);
	}

	// in case ACA / dense precomputation was done, delete precomputed data
	if (data->U!=0)
		cudaFree(data->U);
	if (data->V!=0)
		cudaFree(data->V);
	if (data->dA!=0)
		cudaFree(data->dA);

	delete [] data->aca_work_size;
	delete [] data->dense_work_size;

	// shutdown magma
	magma_finalize();
}


void apply_full_mvp(double* x, double* y, struct h_matrix_data* data)
{
        // generate some necessary data structure
	TIME_start;
        struct work_item test_mat_vec_data;
        test_mat_vec_data.set1_l = 0;
        test_mat_vec_data.set1_u = data->point_count[0] - 1;
        test_mat_vec_data.set2_l = 0;
        test_mat_vec_data.set2_u = data->point_count[1] - 1;

	// allocate full matrix
        double* full_matrix;
        cudaMalloc((void**)&full_matrix, data->point_count[0]*data->point_count[1]*sizeof(double));
        checkCUDAError("cudaMalloc");
        
	// fill full matrix
	int block_size1 = MATRIX_ENTRY_BLOCK_SIZE;
        
	fill_matrix_fun(full_matrix, test_mat_vec_data, data->points_d[0], data->points_d[1], data->point_count[0], data->point_count[1], (data->point_count[0]*data->point_count[1] + (block_size1 - 1)) / block_size1, block_size1, data->assem);
        cudaThreadSynchronize();
        checkCUDAError("fill_matrix");
	
	// init CUBLAS
	cublasStatus_t stat;
	cublasHandle_t handle;
	stat = cublasCreate(&handle);

	// adapt order of variables
        reorder_vector(x,data->point_count[1],data->order[1]);

        // matrix-vector-product
        double one;
        double zero;
        one = 1.0;
        zero = 0.0;
        stat = cublasDgemv(handle, CUBLAS_OP_N, data->point_count[0], data->point_count[1], &one, full_matrix, data->point_count[0], x, 1, &zero, y, 1);
        if (stat!=CUBLAS_STATUS_SUCCESS)
        {
                printf("dgemv did not succeed...\n");
                exit(1);
        }
	cublasDestroy(handle);
	TIME_stop("dense_mvp");

	// recover original variable order
        reorder_back_vector(x,data->point_count[1],data->order[1]);
        reorder_back_vector(y,data->point_count[0],data->order[0]);

	// cleanup of full matrix
	cudaFree(full_matrix);
}

__global__ void gen_gaussian_kernel_rhs(double* rhs, double** points, int dim, int row_count)
{
        int idx = blockIdx.x*blockDim.x+threadIdx.x;

        if (idx<row_count)
        {
                // get row index in rhs

                double result = 1.0;

                for (int d=0; d<dim; d++)
                        result = result * sqrt(M_PI) * (erf(1.0- points[d][idx]) - erf(0.0 - points[d][idx])) / (2.0);

                rhs[idx] = result;
        }
}

void set_gaussian_kernel_rhs(double* b, struct h_matrix_data* data)
{
        int block_size = 1024;
        int grid_size = (data->point_count[0] + block_size -1)/block_size;

        double** points = data->coords_d[0];
        int dim = data->dim;

        double** points_device;
        cudaMalloc((void**)&points_device, sizeof(double*)*dim);
        cudaMemcpy(points_device, points, sizeof(double*)*dim, cudaMemcpyHostToDevice);
        checkCUDAError("memcpy");

        gen_gaussian_kernel_rhs<<<grid_size, block_size>>>(b, points_device, dim, data->point_count[0]);
        cudaThreadSynchronize();
        checkCUDAError("gen_gaussian_kernel_rhs");

        reorder_back_vector(b, data->point_count[0], data->order[0]);
		
}


