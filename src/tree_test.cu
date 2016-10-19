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

cudaEvent_t start, stop; 
float milliseconds;

#define TIME_start {cudaEventCreate(&start); cudaEventCreate(&stop); cudaEventRecord(start);}
#define TIME_stop(a) {cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&milliseconds, start, stop); printf("%s: Elapsed time: %lf ms\n", a, milliseconds); }

#ifndef CHECK_CUDA_ERROR
#define CHECK_CUDA_ERROR
void checkCUDAError(const char* msg) {
cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}
#endif

void print_binary(uint64_t val)
{
	char c;
	for (int i=0; i<64; i++)
	{
		if ((val & 0x8000000000000000u)>0)
			c='1';
		else
			c='0';
		val = val << 1;
		printf("%c",c);
		
	}
	printf("\n");
}



__global__ void fill_with_indices(uint64_t* indices, int count)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (idx>=count) return;

	indices[idx] = (uint64_t)idx;

	return;
 
}

__global__ void reorder_by_index(double* output, double* input, uint64_t* indices, int count)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx>=count) return;

	uint64_t ind = indices[idx];

	output[idx] = input[ind];

	return;
}

__global__ void init_point_set(struct point_set* points, double** coords_device, int dim, double* max_per_dim, double* min_per_dim, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx>=1) return;

	points->dim = dim;
	points->size = size;
	points->coords = coords_device;
	points->max_per_dim = max_per_dim;
	points->min_per_dim = min_per_dim;

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
}

void get_min_and_max(double* min, double* max, double* values, int size)
{
	thrust::device_ptr<double> values_ptr =  thrust::device_pointer_cast(values);
	thrust::pair<thrust::device_ptr<double>,thrust::device_ptr<double> > minmax = thrust::minmax_element(values_ptr, values_ptr + size);
	*min = *minmax.first;
	*max = *minmax.second;
}

void compute_minmax(struct point_set* points_d)
{
	struct point_set points_h;
	cudaMemcpy(&points_h, points_d, sizeof(struct point_set), cudaMemcpyDeviceToHost);
	int dim = points_h.dim;

	double** coords_d = new double*[dim];
	cudaMemcpy(coords_d, points_h.coords, dim*sizeof(double*), cudaMemcpyDeviceToHost);

	// compute extremal values for the point set
	double* min_per_dim_h = new double[dim];
	double* max_per_dim_h = new double[dim];
	for (int d=0; d<dim; d++)
		get_min_and_max(&(min_per_dim_h[d]), &(max_per_dim_h[d]), coords_d[d], points_h.size);
	cudaMemcpy(points_h.max_per_dim, max_per_dim_h, points_h.dim*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(points_h.min_per_dim, min_per_dim_h, points_h.dim*sizeof(double), cudaMemcpyHostToDevice);

	delete [] min_per_dim_h;
	delete [] max_per_dim_h;
	delete [] coords_d;
}

void get_morton_ordering(struct point_set* points_d, struct morton_code* morton_d, uint64_t* order)
{
	struct point_set points_h;
	cudaMemcpy(&points_h, points_d, sizeof(struct point_set), cudaMemcpyDeviceToHost);
	struct morton_code morton_h;
	cudaMemcpy(&morton_h, morton_d, sizeof(struct morton_code), cudaMemcpyDeviceToHost);
	
	int point_count = points_h.size;

	thrust::device_ptr<uint64_t> morton_codes_ptr = thrust::device_pointer_cast(morton_h.code);
	thrust::device_ptr<uint64_t> order_ptr = thrust::device_pointer_cast(order);

	// calculate GPU thread configuration	
	int block_size = 512;
	int grid_size = (point_count + (block_size - 1)) / block_size;

	
	// generate index array initially set to 1:point_count	
	fill_with_indices<<<grid_size, block_size>>>(order, point_count);

	// find ordering of points following Z curve
	TIME_start;
	thrust::sort_by_key(morton_codes_ptr, morton_codes_ptr + point_count, order_ptr);
	TIME_stop("sort_by_key");
}

void print_points(struct point_set* points_d)
{	
	struct point_set points_h;
	cudaMemcpy(&points_h, points_d, sizeof(struct point_set), cudaMemcpyDeviceToHost);
	int dim = points_h.dim;
	int point_count = points_h.size;
	
	double** coords_d_host = new double*[dim];
	cudaMemcpy(coords_d_host, points_h.coords, sizeof(double*)*dim, cudaMemcpyDeviceToHost);
	
	double** coords_h = new double*[dim];
	for (int d=0; d<dim; d++)
	{
		coords_h[d] = new double[point_count];
		cudaMemcpy(coords_h[d], coords_d_host[d], sizeof(double)*point_count, cudaMemcpyDeviceToHost);
	}


	for (int p=0; p<point_count; p++)
	{
		for (int d=0; d<dim; d++)
		{
			printf("%lf ", coords_h[d][p]);
		}
		printf("\n");
	}
	
	for (int d=0; d<dim; d++)
		delete [] coords_h[d];
	delete [] coords_h;
	delete [] coords_d_host;

}

void write_points(struct point_set* points_d, char* file_name)
{	
	struct point_set points_h;
	cudaMemcpy(&points_h, points_d, sizeof(struct point_set), cudaMemcpyDeviceToHost);
	int dim = points_h.dim;
	int point_count = points_h.size;
	
	double** coords_d_host = new double*[dim];
	cudaMemcpy(coords_d_host, points_h.coords, sizeof(double*)*dim, cudaMemcpyDeviceToHost);
	
	double** coords_h = new double*[dim];
	for (int d=0; d<dim; d++)
	{
		coords_h[d] = new double[point_count];
		cudaMemcpy(coords_h[d], coords_d_host[d], sizeof(double)*point_count, cudaMemcpyDeviceToHost);
	}

	FILE* f = fopen(file_name,"w");	

	for (int p=0; p<point_count; p++)
	{
		for (int d=0; d<dim; d++)
		{
			fprintf(f,"%lf ", coords_h[d][p]);
		}
		fprintf(f,"\n");
	}
	
	fclose(f);

	for (int d=0; d<dim; d++)
		delete [] coords_h[d];
	delete [] coords_h;
	delete [] coords_d_host;

}



__global__ void set_2d_test_set(struct point_set* points)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx>=16) return;

	int ix = idx / (4);
	int iy = idx % 4;

	points->coords[0][idx] = ix;
	points->coords[1][idx] = iy;

	return;
}

__global__ void set_3d_test_set(struct point_set* points)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx>=64) return;

	int ix = idx / (4*4);
	int tmp;
	tmp = idx % (4*4);
	int iy = tmp / 4;
	int iz = tmp % 4;

	points->coords[0][idx] = ix;
	points->coords[1][idx] = iy;
	points->coords[2][idx] = iz;
}

void reorder_point_set(struct point_set* points_d, uint64_t* order)
{
	struct point_set points_h;
	cudaMemcpy(&points_h, points_d, sizeof(struct point_set), cudaMemcpyDeviceToHost);
	int dim = points_h.dim;
	int point_count = points_h.size;
	
	double** coords_d_host = new double*[dim];
	cudaMemcpy(coords_d_host, points_h.coords, sizeof(double*)*dim, cudaMemcpyDeviceToHost);

	double* coords_tmp;
	cudaMalloc((void**)&coords_tmp, point_count*sizeof(double));

	// calculate GPU thread configuration	
	int block_size = 512;
	int grid_size = (point_count + (block_size - 1)) / block_size;


	TIME_start
	for (int d=0; d<dim; d++)
	{
		reorder_by_index<<<grid_size, block_size>>>(coords_tmp, coords_d_host[d], order, point_count);
		cudaMemcpy(coords_d_host[d], coords_tmp, point_count*sizeof(double), cudaMemcpyDeviceToDevice);	
	}
	TIME_stop("reorder points");
	
	cudaFree(coords_tmp);
}


void print_points_with_morton_codes(struct point_set* points_d, struct morton_code* code_d)
{
	struct point_set points_h;
	cudaMemcpy(&points_h, points_d, sizeof(struct point_set), cudaMemcpyDeviceToHost);
	int dim = points_h.dim;
	int point_count = points_h.size;
	
	double** coords_d_host = new double*[dim];
	cudaMemcpy(coords_d_host, points_h.coords, sizeof(double*)*dim, cudaMemcpyDeviceToHost);
	
	double** coords_h = new double*[dim];
	for (int d=0; d<dim; d++)
	{
		coords_h[d] = new double[point_count];
		cudaMemcpy(coords_h[d], coords_d_host[d], sizeof(double)*point_count, cudaMemcpyDeviceToHost);
	}

	struct morton_code code_h;
	cudaMemcpy(&code_h, code_d, sizeof(struct morton_code), cudaMemcpyDeviceToHost);

	uint64_t* codes_h = new uint64_t[point_count];
	cudaMemcpy(codes_h, code_h.code, point_count*sizeof(uint64_t), cudaMemcpyDeviceToHost);
	
	for (int p=0; p<point_count; p++)
	{
		for (int d=0; d<dim; d++)
		{
			printf("%lf ", coords_h[d][p]);
		}
		printf("\n");
		print_binary(codes_h[p]);
	}
	
	for (int d=0; d<dim; d++)
		delete [] coords_h[d];
	delete [] coords_h;
	delete [] coords_d_host;

	delete [] codes_h;

	
}


void allocate_work_queue(struct work_queue **new_work_queue_h, struct work_queue **new_work_queue, int work_queue_size)
{
	new_work_queue_h[0] = new struct work_queue;
	(new_work_queue_h[0])->init(work_queue_size);
	
	cudaMalloc((void**)&(new_work_queue[0]), sizeof(struct work_queue));
	checkCUDAError("malloc");
	cudaMemcpy(*new_work_queue, *new_work_queue_h, sizeof(struct work_queue), cudaMemcpyHostToDevice);
	checkCUDAError("memcpy");
	
}

void delete_work_queue(struct work_queue *new_work_queue_h, struct work_queue *new_work_queue)
{
	new_work_queue_h->destroy(); // deleting internal queue storage on GPU
	cudaFree(new_work_queue); // deleting mother object on GPU
	delete new_work_queue_h; // deleting mother object on CPU
}

__global__ void init_tree_work_queue_root(struct work_queue **tree_work_queue, int set1_l, int set1_u, int set2_l, int set2_u)
{
	struct work_item item;
	item.set1_l = set1_l;
	item.set1_u = set1_u;
	item.set2_l = set2_l;
	item.set2_u = set2_u;

	(tree_work_queue[0])->warp_single_put(item, 0);

}

__global__ void invalidate_work_items(struct work_item* items, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx>=size)
		return;

	items[idx].set1_l = -1;
	items[idx].set1_u = -1;
	items[idx].set2_l = -1;
	items[idx].set2_u = -1;
}



int main( int argc, char* argv[])
{

//	int dim = 3;
//	int bits = 20;
	int dim = 2;
	int bits = 32;

//	int point_count=16;
//	int point_count=20;
	int point_count = atoi(argv[1]);
//	int point_count=4000000;
//	int point_count=29000000;

	if (argc!=6)
	{
		printf("./tree_test <N> <k> <c_leaf> <exponent of epsilon> <eta>\n");
		return 0;
	}

	// allocating memory for point_count coordinates in dim dimensions
	double** coords_d;
	coords_d = new double*[dim];
	for (int d = 0; d < dim; d++)
	{
		cudaMalloc((void**)&(coords_d[d]), point_count*sizeof(double));
		checkCUDAError("cudaMalloc");
	}

	// allocating memory for extremal values per dimension
	double* max_per_dim_d;
	cudaMalloc((void**)&max_per_dim_d, dim*sizeof(double));
	double* min_per_dim_d;
	cudaMalloc((void**)&min_per_dim_d, dim*sizeof(double));

	// generating device pointer that holds the dimension-wise access
	double** coords_device;
	cudaMalloc((void**)&(coords_device), dim*sizeof(double*));
	cudaMemcpy(coords_device, coords_d, dim*sizeof(double*), cudaMemcpyHostToDevice);
	
	// allocationg memory for morton codes
	uint64_t* code_d;
	cudaMalloc((void**)&code_d, point_count*sizeof(uint64_t));
	checkCUDAError("cudaMalloc");

	// setting up data strcture for point set
	struct point_set* points_d;
	cudaMalloc((void**)&points_d, sizeof(struct point_set));
	init_point_set<<<1,1>>>(points_d, coords_device, dim, max_per_dim_d, min_per_dim_d, point_count);

	// setting up data structure for morton code
	struct morton_code* morton_d;
	cudaMalloc((void**)&morton_d, sizeof(struct morton_code));
	init_morton_code<<<1,1>>>(morton_d, code_d, dim, bits, point_count);


	// generate random points
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	for (int d = 0; d < dim; d++ )
	{
		curandGenerateUniformDouble(gen, coords_d[d], point_count);
	}
	curandDestroyGenerator(gen);



//	set_2d_test_set<<<1,16>>>(points_d);

//	set_3d_test_set<<<1,64>>>(points_d);



	// compute extremal values for the point set
	compute_minmax(points_d);
	
	// calculate GPU thread configuration	
	int block_size = 512;
	int grid_size = (point_count + (block_size - 1)) / block_size;



	// generate morton codes
	TIME_start;
	get_morton_code<<<grid_size, block_size>>>(points_d, morton_d);
	TIME_stop("get_morton_3d");
	checkCUDAError("get_morton_code");

//	print_points_with_morton_codes(points_d, morton_d);


	// find ordering of points following Z curve
	uint64_t* order;
	cudaMalloc((void**)&order, point_count*sizeof(uint64_t));
	get_morton_ordering(points_d, morton_d, order);

//	print_points_with_morton_codes(points_d, morton_d);

	// reorder points following the morton code order
//	TIME_start;
	reorder_point_set(points_d, order);
//	TIME_stop("reorder_point_set");

//	double eta=10.0;
//	double eta=0.5;
	double eta=atof(argv[5]);
	int max_level=50; // DEBUG
//	int c_leaf=1024;
	int c_leaf=atoi(argv[3]);
//	int c_leaf=4;

	struct work_item root_h;
	root_h.set1_l = 0;
	root_h.set1_u = point_count - 1;
	root_h.set2_l = 0;
	root_h.set2_u = point_count - 1;

	int mat_vec_data_count = 0;  // will be filled with the size number of mat_vec_data entries

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


	int max_elements_in_array = -1; // TODO
	int max_elements_in_mat_vec_data_array = -1; // TODO

	struct work_item** mat_vec_data = new struct work_item*[1];
	int mat_vec_data_array_size = 1048576;
	cudaMalloc((void**)mat_vec_data, mat_vec_data_array_size*sizeof(struct work_item));

	TIME_start;
	traverse_with_dynamic_arrays_dynamic_output(root_h, mat_vec_data, &mat_vec_data_count, &mat_vec_data_array_size, morton_d, morton_d, points_d, points_d, eta, max_level, c_leaf, max_elements_in_array);
	TIME_stop("traverse_with_arrays");

//	printf("mat_vec_data_count: %d\n", mat_vec_data_count);
//	print_work_items(*mat_vec_data, mat_vec_data_count);
//	char file_name[1000];
//	sprintf(file_name, "mat_vec_data.txt");
//	write_work_items(file_name, *mat_vec_data, mat_vec_data_count);

	double* x;
	double* y;
	double* y_test;
	cudaMalloc((void**)&x, point_count*sizeof(double));
	cudaMalloc((void**)&y, point_count*sizeof(double));
	cudaMalloc((void**)&y_test, point_count*sizeof(double));

	// generate random vector x
	curandGenerator_t vec_gen;
	curandCreateGenerator(&vec_gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandGenerateUniformDouble(vec_gen, x, point_count);
	curandDestroyGenerator(vec_gen);

//	TIME_start;
//	struct work_item test_mat_vec_data;
//	test_mat_vec_data.set1_l = 0;
//	test_mat_vec_data.set1_u = point_count - 1;
//	test_mat_vec_data.set2_l = 0;
//	test_mat_vec_data.set2_u = point_count - 1;
//	double* test_matrix;
//	cudaMalloc((void**)&test_matrix, point_count*point_count*sizeof(double));
//	checkCUDAError("cudaMalloc");
//	int block_size1 = 512;
//	fill_matrix<<<(point_count*point_count + (block_size1 - 1)) / block_size1, block_size1>>>(test_matrix, test_mat_vec_data, points_d, points_d, point_count, point_count);
//	cudaThreadSynchronize();
//	checkCUDAError("fill_matrix");
//    cublasStatus_t stat;
//    cublasHandle_t handle;
//    stat = cublasCreate(&handle);
//	// matrix-vector-product
//	double one;
//	double zero;
//	one = 1.0;
//	zero = 0.0;
//	stat = cublasDgemv(handle, CUBLAS_OP_N, point_count, point_count, &one, test_matrix, point_count, x, 1, &zero, y_test, 1);
//	if (stat!=CUBLAS_STATUS_SUCCESS)
//	{
//		printf("dgemv did not succeed...\n");
//		exit(1);
//	}
//    cublasDestroy(handle);
//    TIME_stop("dense_mvp");
//
//    cudaFree(test_matrix);

	int k = atoi(argv[2]);

	double epsilon;
	epsilon = pow(10.0, atoi(argv[4]));

	printf("dort\n");
	TIME_start;
	sequential_h_matrix_mvp(x, y, *mat_vec_data, mat_vec_data_count, mat_vec_data_array_size, points_d, points_d, point_count, eta, epsilon, k);
	TIME_stop("sequential_h_matrix");

	thrust::device_ptr<double> y_ptr(y);
	thrust::device_ptr<double> y_test_ptr(y_test);

//	printf("x,y, y_test\n");
//	print_double(x, point_count);
//	print_double(y, point_count);
//	print_double(y_test, point_count);

	thrust::transform(y_test_ptr, y_test_ptr+point_count, y_ptr, y_test_ptr, thrust::minus<double>());
	double error = sqrt(thrust::inner_product(y_test_ptr, y_test_ptr+point_count, y_test_ptr, 0.0));

	printf("Error: %le\n", error);
//
	cudaFree(y_test);

	cudaFree(x);
	cudaFree(y);

	cudaFree(*mat_vec_data);




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
	

	cudaFree(order);

	// freeing memory for morton codes
	cudaFree(code_d);

	// freeing coordinates memory
	for (int d = 0; d < dim; d++)
	{
		cudaFree(coords_d[d]);
	}
	delete [] coords_d;

}
