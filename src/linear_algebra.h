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


#ifndef LINEAR_ALGEBRA_H_
#define LINEAR_ALGEBRA_H_


#define MATRIX_ENTRY_BLOCK_SIZE 512

#include "morton.h"
#include "tree.h"
#include <thrust/device_ptr.h>
#include "cublas_v2.h"

#include "magma_v2.h"
#include "magma_lapack.h"


#ifndef CHECK_CUDA_ERROR
#define CHECK_CUDA_ERROR
extern void checkCUDAError(const char* msg);
#endif

#define KT_GAUSSIAN 1
#define KT_MATERN_1 2
#define KT_MATERN_2 3

extern void sort_mat_vec_data(struct work_item* mat_vec_data, int mat_vec_data_count);

extern void apply_dense_matrix_for_current_work_item(double* x, double* y, struct work_item current_mat_vec_data, struct point_set* input_set1, struct point_set* input_set2, cublasStatus_t stat, cublasHandle_t handle, struct system_assembler* assem);

extern double compute_frobenius_norm_of_low_rank_matrix(double* U, double* V, int m1, int m2, int k, cublasStatus_t stat, cublasHandle_t handle);

extern void apply_aca_for_current_work_item(double* x, double* y, struct work_item current_mat_vec_data, struct point_set* input_set1, struct point_set* input_set2,  cublasStatus_t stat, cublasHandle_t handle, double eta, double epsilon, int k, struct system_assembler* assem);

extern void compute_batched_norms(double* batched_norms, int* norm_count, double* x, int m_total, thrust::device_ptr<int> work_item_map_ptr, int block_size);

extern void compute_batched_norms_with_keys_output(double* batched_norms, int* keys_output, int* norm_count, double* x, int m_total, thrust::device_ptr<int> work_item_map_ptr, int block_size);

extern void compute_batched_products_for_kxk_matrices(double* batched_products, int* products_count, double* C, double* D, int m_total, thrust::device_ptr<int> work_item_map_ptr, int block_size, bool* stop_aca_as_soon_as_possible);

extern void batched_low_rank_mvp(double* x, double* y, double* U, double* V, int m1_total, int m2_total, int* m1_h, int* m2_h, int mat_vec_data_count, int batch_count, int k, int* k_per_item, cudaStream_t *streams, cublasStatus_t stat, cublasHandle_t handle , int* point_map_offsets1_h, int* point_map_offsets2_h, int* point_map1, int* point_map2, int* work_item_map1 );

extern bool do_stop_based_on_batched_frobenius_norm(double* U, double* V, double* u_r, double* v_r, int m1_total, int m2_total, int* point_map_offsets1_h, int* point_map_offsets2_h, bool* stop_aca_as_soon_as_possible, bool* stop_aca_as_soon_as_possible_h, int* work_item_map1, int* work_item_map2, int batch_count, int r, int mat_vec_data_count, int* m1_h, int* m2_h, double eta, double epsilon, cudaStream_t *streams, cublasStatus_t stat, cublasHandle_t handle );

//--------------------------------------------------------------
// compute mapping of batch data entries to global point indices
//--------------------------------------------------------------
extern void compute_point_map(int* point_map1, int* point_map2, int m1_total, int m2_total, int* m1, int* m2, int* point_map_offsets1, int* point_map_offsets2, struct work_item* mat_vec_data, int mat_vec_data_count);

// --------------------------------------------------------------
// compute mapping of rows in batched data to index in work_queue	
// --------------------------------------------------------------
extern void compute_work_item_maps(int* work_item_map1, int* work_item_map2, int m1_total, int m2_total, int* point_map_offsets1, int* point_map_offsets2, int* m1, int* m2, struct work_item* mat_vec_data, int mat_vec_data_count);

// ------------------------------------------------------------------------------------------------------------
// creating map between work item list (including invalid entries) and batch set list (without invalid entries)
// ------------------------------------------------------------------------------------------------------------
extern void compute_work_item_to_batch_map(int* work_item_to_batch_map, struct work_item* mat_vec_data, int mat_vec_data_count, int* batch_count);

extern void compute_m1_m2(int* m1, int* m2, struct work_item* mat_vec_data, int mat_vec_data_count);

extern void fill_matrix_fun(double* matrix, struct work_item current_mat_vec_data, struct point_set* input_set1, struct point_set* input_set2, int m1, int m2, int grid_size, int block_size, struct system_assembler* assem);


extern void create_maps_and_indices(int* m1, int* m2, int m1_total, int m2_total, int* point_map_offsets1, int* point_map_offsets2, int* point_map1, int* point_map2, int* work_item_map1, int* work_item_map2, int* work_item_to_batch_map, int* batch_count, struct work_item* mat_vec_data, int mat_vec_data_count);



extern void compute_current_batched_v_r(double* v_r, double* U, double* V, int m1_total, int m2_total, struct work_item* mat_vec_data, int mat_vec_data_count, int* compute_v_r, int* i_r, int* point_map1, int* point_map2, int* point_map_offsets1, int* point_map_offsets2, int* work_item_map2, struct point_set* input_set1, struct point_set* input_set2, int* k_per_item, int r, struct system_assembler* assem);


extern void compute_current_batched_u_r(double* u_r, double* v_r, double* U, double* V, int m1_total, int m2_total, struct work_item* mat_vec_data, int mat_vec_data_count, int* point_map1, int* point_map2, int* work_item_map1, int* work_item_map2, struct point_set* input_set1, struct point_set* input_set2, int* k_per_item, int* j_r_global, int* work_item_to_batch_map, int r, struct system_assembler* assem);


extern void apply_batched_aca(double* x, double* y, struct work_item* mat_vec_data, int mat_vec_data_count, struct point_set* input_set1, struct point_set* input_set2, cublasStatus_t stat, cublasHandle_t handle, double eta, double epsilon, int k, struct system_assembler* assem);

struct mat_vec_data_info
{
	int dense_count;
	int aca_count;
	int total_count;
};


extern void organize_mat_vec_data(struct work_item* mat_vec_data, int mat_vec_data_count, struct mat_vec_data_info* mat_vec_info);

extern void precompute_work_sizes(int** dense_work_size, int** aca_work_size, int* dense_batch_count, int* aca_batch_count, struct work_item* mat_vec_data, struct mat_vec_data_info* mat_vec_info, int max_batched_dense_size, int max_batched_aca_size);

extern int compute_current_dense_work_size(struct work_item* mat_vec_data_h, struct mat_vec_data_info* mat_vec_info, int max_batched_size, double batching_ratio, int current_work_item_index);

extern int compute_current_aca_work_size(struct work_item* mat_vec_data_h, struct mat_vec_data_info* mat_vec_info, int max_batched_size, int current_work_item_index);

extern void apply_batched_dense(double* x, double* y, struct work_item* mat_vec_data, int mat_vec_data_count, struct point_set* input_set1, struct point_set* input_set2, cublasStatus_t stat, cublasHandle_t handle, struct system_assembler* assem);

extern void precompute_aca_for_h_matrix_mvp(struct work_item* mat_vec_data, struct mat_vec_data_info* mat_vec_info, struct point_set* input_set1, struct point_set* input_set2, double eta, double epsilon, int k, double** U, double** V, struct system_assembler* assem);

extern void h_matrix_mvp(double* x, double* y, struct work_item* mat_vec_data, struct mat_vec_data_info* mat_vec_info, struct point_set* input_set1, struct point_set* input_set2, double eta, double epsilon, int k, double* dA, double* U, double* V, struct system_assembler* assem, int max_batched_dense_size, double dense_batching_ratio, int max_batched_aca_size, magma_queue_t magma_queue, int* dense_work_size, int* aca_work_size, int dense_batch_count, int aca_batch_count, bool use_precomputed_aca, bool use_precomputed_dense);

extern void sequential_h_matrix_mvp_without_batching(double* x, double* y, struct work_item* mat_vec_data, struct mat_vec_data_info* mat_vec_info, struct point_set* input_set1, struct point_set* input_set2, double eta, double epsilon, int k, struct system_assembler* assem);
#endif /* LINEAR_ALGEBRA_H_ */
