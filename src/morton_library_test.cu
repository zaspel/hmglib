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

cudaEvent_t start, stop; 
float milliseconds;

#define TIME_start {cudaEventCreate(&start); cudaEventCreate(&stop); cudaEventRecord(start);}
#define TIME_stop(a) {cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&milliseconds, start, stop); printf("%s: Elapsed time: %lf ms\n", a, milliseconds); }

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

int main( int argc, char* argv[])
{

	int dim = 3;
	int bits = 20;
//	int dim = 2;
//	int bits = 32;

//	int point_count=32;
	int point_count=29000000;


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

	// reorder points following the morton code order
	reorder_point_set(points_d, order);
	
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
