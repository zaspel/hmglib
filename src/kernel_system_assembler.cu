#include <stdio.h>
#include <curand.h>

#include "system_assembler.h"
//#include "generic_system_adapter.h"
#include "kernel_system_assembler.h"

__global__ void create_gaussian_kernel_system_assembler_object_kernel(struct gaussian_kernel_system_assembler** assem, double regularization)
{
	(*assem) = new gaussian_kernel_system_assembler();
	(*assem)->regularization = regularization;
}

void create_gaussian_kernel_system_assembler_object(struct gaussian_kernel_system_assembler** assem, double regularization)
{
	create_gaussian_kernel_system_assembler_object_kernel<<<1,1>>>(assem, regularization);
	cudaThreadSynchronize();
	checkCUDAError("create_gaussian_kernel_system_assembler_object");
}


__global__ void destroy_gaussian_kernel_system_assembler_object_kernel(struct gaussian_kernel_system_assembler** assem)
{
	delete *assem;
}

void destroy_gaussian_kernel_system_assembler_object(struct gaussian_kernel_system_assembler** assem)
{
	destroy_gaussian_kernel_system_assembler_object_kernel<<<1,1>>>(assem);
	cudaThreadSynchronize();
 	checkCUDAError("destroy_gaussian_kernel_system_assembler_object");
}



__global__ void create_matern_kernel_system_assembler_object_kernel(struct matern_kernel_system_assembler** assem, double regularization)
{
	(*assem) = new matern_kernel_system_assembler();
	(*assem)->regularization = regularization;
}

void create_matern_kernel_system_assembler_object(struct matern_kernel_system_assembler** assem, double regularization)
{
	create_matern_kernel_system_assembler_object_kernel<<<1,1>>>(assem, regularization);
	cudaThreadSynchronize();
	checkCUDAError("create_matern_kernel_system_assembler_object");
}


__global__ void destroy_matern_kernel_system_assembler_object_kernel(struct matern_kernel_system_assembler** assem)
{
	delete *assem;
}

void destroy_matern_kernel_system_assembler_object(struct matern_kernel_system_assembler** assem)
{
	destroy_matern_kernel_system_assembler_object_kernel<<<1,1>>>(assem);
	cudaThreadSynchronize();
 	checkCUDAError("destroy_matern_kernel_system_assembler_object");
}



