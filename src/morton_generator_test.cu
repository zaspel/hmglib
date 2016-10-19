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



int main( int argc, char* argv[])
{

	int dim = 3;
//	int dim = 2;

	int point_count=29000000;
//	int point_count=16;


	// allocating memory for point_count coordinates in dim dimensions
	double** coords;
	coords = new double*[dim];
	for (int d = 0; d < dim; d++)
	{
		cudaMalloc((void**)&(coords[d]), point_count*sizeof(double));
		checkCUDAError("cudaMalloc");
	}

	// allocationg memory for morton codes
	uint64_t* morton_codes;
	cudaMalloc((void**)&morton_codes, point_count*sizeof(uint64_t));
	checkCUDAError("cudaMalloc");

    	// wrap raw pointers with a device pointers 
	thrust::device_ptr<uint64_t> morton_codes_ptr = thrust::device_pointer_cast(morton_codes);
	thrust::device_ptr<double> coords_x_ptr = thrust::device_pointer_cast(coords[0]);
	thrust::device_ptr<double> coords_y_ptr = thrust::device_pointer_cast(coords[1]);
	thrust::device_ptr<double> coords_z_ptr = thrust::device_pointer_cast(coords[2]);

	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	for (int d = 0; d < dim; d++ )
	{
		curandGenerateUniformDouble(gen, coords[d], point_count);
	}


/*	for (int i=0; i<64; i++)
	{
		int ix = i / (4*4);
		int tmp;
		tmp = i % (4*4);
		int iy = tmp / 4;
		int iz = tmp % 4;

		coords_x_ptr[i] = ix;
		coords_y_ptr[i] = iy;
		coords_z_ptr[i] = iz;
	}
*/

/*	for (int i=0; i<16; i++)
	{
		int ix = i / (4);
		int iy = i % 4;

		coords_x_ptr[i] = ix;
		coords_y_ptr[i] = iy;
	}
*/

	TIME_start
	// find minimum and maximum value per space dimension
	thrust::pair<thrust::device_ptr<double>,thrust::device_ptr<double> > minmax_x = thrust::minmax_element(coords_x_ptr, coords_x_ptr + point_count);
	thrust::pair<thrust::device_ptr<double>,thrust::device_ptr<double> > minmax_y = thrust::minmax_element(coords_y_ptr, coords_y_ptr + point_count);
	thrust::pair<thrust::device_ptr<double>,thrust::device_ptr<double> > minmax_z = thrust::minmax_element(coords_z_ptr, coords_z_ptr + point_count);
	TIME_stop("minmax_element");

	// extract extremal values
	double min_x = *minmax_x.first;
	double min_y = *minmax_y.first;
	double min_z = *minmax_z.first;
	double max_x = *minmax_x.second;
	double max_y = *minmax_y.second;
	double max_z = *minmax_z.second;

	printf("min: %lf %lf %lf\n", min_x, min_y, min_z);
	printf("max: %lf %lf %lf\n", max_x, max_y, max_z);

//	printf("min: %lf %lf\n", min_x, min_y);
//	printf("max: %lf %lf\n", max_x, max_y);



	
	int block_size = 512;
	int grid_size = (point_count + (block_size - 1)) / block_size;

	TIME_start;
	get_morton_3d<<<grid_size, block_size>>>(morton_codes, coords[0], coords[1], coords[2], min_x, min_y, min_z, max_x-min_x, max_y-min_y, max_z-min_z, point_count);
//	get_morton_2d<<<grid_size, block_size>>>(morton_codes, coords[0], coords[1], min_x, min_y, max_x-min_x, max_y-min_y, point_count);
	TIME_stop("get_morton_3d");
	checkCUDAError("get_morton_3d");

/*	for (int i = 0; i<point_count; i++)
	{
		double x,y,z;
//		double x,y;
		uint64_t m;
		x=coords_x_ptr[i];
		y=coords_y_ptr[i];
		z=coords_z_ptr[i];

		m=morton_codes_ptr[i];

		printf("%d: %lf %lf %lf\n", i, x,y,z);
//		printf("%lf %lf\n", x,y);
		print_binary(m);
	}
*/	

	uint64_t* indices;
	cudaMalloc((void**)&indices, point_count*sizeof(uint64_t));
	thrust::device_ptr<uint64_t> indices_ptr = thrust::device_pointer_cast(indices);

	fill_with_indices<<<grid_size, block_size>>>(indices, point_count);

	// find ordering of points following Z curve
	TIME_start;
//	thrust::sort(morton_codes_ptr, morton_codes_ptr+point_count);
	thrust::sort_by_key(morton_codes_ptr, morton_codes_ptr + point_count, indices_ptr);
	TIME_stop("sort_by_key");

/*	for (int p=0; p<point_count; p++)
	{
		uint64_t ind = indices_ptr[p];
		printf("%lud\n",ind);
		uint64_t m = morton_codes_ptr[p];
		print_binary(m);
		double x,y,z;
		x=coords_x_ptr[ind];
		y=coords_y_ptr[ind];
		z=coords_z_ptr[ind];

		printf("%lf %lf %lf;\n",x,y,z);
	}

*/

	double* coords_tmp;
	cudaMalloc((void**)&coords_tmp, point_count*sizeof(double));

	TIME_start
	for (int d=0; d<dim; d++)
	{
		reorder_by_index<<<grid_size, block_size>>>(coords_tmp, coords[d], indices, point_count);
		cudaMemcpy(coords[d], coords_tmp, point_count*sizeof(double), cudaMemcpyDeviceToDevice);	
	}
	TIME_stop("reorder points");
	
	cudaFree(coords_tmp);

/*	for (int i=0;i<point_count;i++)
	{
		uint64_t ind,indt;
		indt = indices_tmp_ptr[i];
		ind = indices_ptr[i];
		double x = coords_x_ptr[i];
		double y = coords_y_ptr[i];
		double z = coords_z_ptr[i];
		printf("%lu %lu:\t %lf %lf %lf\n",ind,indt, x,y,z); 
	}
*/

	double* coords_x_h = new double[point_count];
	double* coords_y_h = new double[point_count];
	double* coords_z_h = new double[point_count];

		
	cudaMemcpy(coords_x_h, coords[0], point_count*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(coords_y_h, coords[1], point_count*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(coords_z_h, coords[2], point_count*sizeof(double), cudaMemcpyDeviceToHost);

/*	FILE* f = fopen("a.dat","w");
	

	for (int p=0; p<point_count; p++
)
	{
		fprintf(f,"%lf %lf %lf\n",coords_x_h[p],coords_y_h[p],coords_z_h[p]);
//		uint64_t m = morton_codes_ptr[p];
//		print_binary(m);
//		printf("%lf %lf;\n",coords_x_h[p],coords_y_h[p]);
	}
	fclose(f);
*/
	delete [] coords_x_h;
	delete [] coords_y_h;
	delete [] coords_z_h;	


	curandDestroyGenerator(gen);

	cudaFree(indices);

	// freeing memory for morton codes
	cudaFree(morton_codes);

	// freeing coordinates memory
	for (int d = 0; d < dim; d++)
	{
		cudaFree(coords[d]);
	}
	delete [] coords;

}
