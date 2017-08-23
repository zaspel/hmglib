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

#ifndef HMGLIB_H
#define HMGLIB_H


#include "morton.h"
#include "tree.h"
#include "linear_algebra.h"

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

	int bits;

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

	int kernel_type;

	int max_batched_dense_size;
	int max_batched_aca_size;

	double dense_batching_ratio;

	double* U;
	double* V;
};

extern void init_h_matrix_data(struct h_matrix_data* data, int point_count[2], int dim, int bits);

extern void setup_h_matrix(struct h_matrix_data* data);

extern void apply_h_matrix_mvp(double* x, double* y, struct h_matrix_data* data);

extern void apply_h_matrix_mvp_without_batching(double* x, double* y, struct h_matrix_data* data);

extern void destroy_h_matrix_data(struct h_matrix_data* data);

extern void precompute_aca(struct h_matrix_data* data);

extern void apply_full_mvp(double* x, double* y, struct h_matrix_data* data);

extern void set_gaussian_kernel_rhs(double* b, struct h_matrix_data* data);


#endif
