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

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>
#include "tree.h"

#include "helper.h"


cudaEvent_t helper_start, helper_stop;
float helper_milliseconds;
#define TIME_start // {cudaEventCreate(&helper_start); cudaEventCreate(&helper_stop); cudaEventRecord(helper_start);}
#define TIME_stop(a) // {cudaEventRecord(helper_stop); cudaEventSynchronize(helper_stop); cudaEventElapsedTime(&helper_milliseconds, helper_start, helper_stop); printf("%s: Elapsed time: %lf ms\n", a, helper_milliseconds); }

void checkCUDAError(const char* msg) {
cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

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

__global__ void reorder_back_by_index(double* output, double* input, uint64_t* indices, int count)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx>=count) return;

	uint64_t ind = indices[idx];

	output[ind] = input[idx];

	return;
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

void write_vector(double* x, int n, char* file_name)
{	
	double* x_h = new double[n];
	cudaMemcpy(x_h, x, sizeof(double*)*n, cudaMemcpyDeviceToHost);
	
	FILE* f = fopen(file_name,"w");	

	for (int p=0; p<n; p++)
	{
			fprintf(f,"%le\n", x_h[p]);
	}
	
	fclose(f);

	delete [] x_h;
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

void reorder_vector(double* vector, int vector_length, uint64_t* order)
{
	double* vector_tmp;
	cudaMalloc((void**)&vector_tmp, vector_length*sizeof(double));

	// calculate GPU thread configuration	
	int block_size = 512;
	int grid_size = (vector_length + (block_size - 1)) / block_size;


	TIME_start
		reorder_by_index<<<grid_size, block_size>>>(vector_tmp, vector, order, vector_length);
		cudaMemcpy(vector, vector_tmp, vector_length*sizeof(double), cudaMemcpyDeviceToDevice);	
	TIME_stop("reorder vector");
	
	cudaFree(vector_tmp);
}

void reorder_back_vector(double* vector, int vector_length, uint64_t* order)
{
	double* vector_tmp;
	cudaMalloc((void**)&vector_tmp, vector_length*sizeof(double));

	// calculate GPU thread configuration	
	int block_size = 512;
	int grid_size = (vector_length + (block_size - 1)) / block_size;

	TIME_start
		reorder_back_by_index<<<grid_size, block_size>>>(vector_tmp, vector, order, vector_length);
		cudaMemcpy(vector, vector_tmp, vector_length*sizeof(double), cudaMemcpyDeviceToDevice);	
	TIME_stop("reorder vector");
	
	cudaFree(vector_tmp);
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


//void allocate_work_queue(struct work_queue **new_work_queue_h, struct work_queue **new_work_queue, int work_queue_size)
//{
//	new_work_queue_h[0] = new struct work_queue;
//	(new_work_queue_h[0])->init(work_queue_size);
//	
//	cudaMalloc((void**)&(new_work_queue[0]), sizeof(struct work_queue));
//	checkCUDAError("malloc");
//	cudaMemcpy(*new_work_queue, *new_work_queue_h, sizeof(struct work_queue), cudaMemcpyHostToDevice);
//	checkCUDAError("memcpy");
//	
//}

//void delete_work_queue(struct work_queue *new_work_queue_h, struct work_queue *new_work_queue)
//{
//	new_work_queue_h->destroy(); // deleting internal queue storage on GPU
//	cudaFree(new_work_queue); // deleting mother object on GPU
//	delete new_work_queue_h; // deleting mother object on CPU
//}

//__global__ void init_tree_work_queue_root(struct work_queue **tree_work_queue, int set1_l, int set1_u, int set2_l, int set2_u)
//{
//	struct work_item item;
//	item.set1_l = set1_l;
//	item.set1_u = set1_u;
//	item.set2_l = set2_l;
//	item.set2_u = set2_u;
//
//	(tree_work_queue[0])->warp_single_put(item, 0);
//
//}

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


