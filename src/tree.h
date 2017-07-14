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

#ifndef TREE_H
#define TREE_H

#include "morton.h"


#ifndef CHECK_CUDA_ERROR
#define CHECK_CUDA_ERROR
extern void checkCUDAError(const char* msg);
#endif



#define WARP_SIZE 32
#define BLOCK_SIZE 512
#define WT_ACA 1
#define WT_DENSE 2

#define MAX_POINT_DIMENSION 5

struct work_item
{
	int set1_l; // lower index bound of the points belonging to set 1
		    // -> this index is the index mapping towards the point set, which is sorted by the Z order curve
	int set1_u; // upper index bound of the points belonging to set 1
	int set2_l; // lower index bound of the points belonging to set 2
	int set2_u; // upper index bound of the points belonging to set 2
	int work_type;  // work type, i.e. WT_ACA or WT_DENSE
	int is_in_use;  // whether this work item is in use (DEBUG: is this ever used????)
	double max1[MAX_POINT_DIMENSION]; // max part of the bounding box for set 1  (assuming an upper bound of MAX_POINT_DIMENSION dimensions)
	double min1[MAX_POINT_DIMENSION]; // mim part of the bounding box for set 1  (assuming an upper bound of  dimensions)
	double max2[MAX_POINT_DIMENSION]; // max part of the bounding box for set 2  (assuming an upper bound of  dimensions)
	double min2[MAX_POINT_DIMENSION]; // mim part of the bounding box for set 2  (assuming an upper bound of  dimensions)

	int dim;
};



extern void print_work_items(struct work_item* work_items, int work_item_count);

extern void write_work_items(char* file_name, struct work_item* work_items, int work_item_count);


//cudaEvent_t sstart, sstop;
//float mmilliseconds;
//
//#define TIME_sstart {cudaEventCreate(&sstart); cudaEventCreate(&sstop); cudaEventRecord(sstart);}
//#define TIME_sstop(a) {cudaEventRecord(sstop); cudaEventSynchronize(sstop); cudaEventElapsedTime(&mmilliseconds, sstart, sstop); printf("%s: Elapsed time: %lf ms\n", a, mmilliseconds); }

extern void traverse_with_arrays(struct work_item root_h, struct work_item* mat_vec_data, int* mat_vec_data_count, struct morton_code* input_set1_codes, struct morton_code* input_set2_codes, struct point_set* input_set1, struct point_set* input_set2, double eta, int max_level, int c_leaf, int max_elements_in_array);

extern void traverse_with_dynamic_arrays(struct work_item root_h, struct work_item* mat_vec_data, int* mat_vec_data_count, struct morton_code* input_set1_codes, struct morton_code* input_set2_codes, struct point_set* input_set1, struct point_set* input_set2, double eta, int max_level, int c_leaf, int max_elements_in_array);

extern void compute_bounding_boxes_fun_old(struct work_item* current_level_data, int total_children, struct point_set* input_set1, struct point_set* input_set2);

extern void print_int(int* array, int n);

extern void print_double(double* array, int n);

extern void print_bool(bool* array, int n);

// this method computes a map from the work item index to the bounding box computation lookup table entry
extern void compute_map_for_lookup_table(int* map, struct work_item* current_level_data, int total_children, struct point_set* input_set1, struct point_set* input_set2, int set_nr);

// this function computes the lookup table for the bounding box computation results
extern void compute_lookup_table(double*** lookup_table_min, double*** lookup_table_max, int* lookup_table_size, struct work_item* current_level_data, int total_children, struct point_set* input_set1, struct point_set* input_set2, int set_nr);

// this method computes the bounding boxes for each work_item in current_level_data (i.e. for each node on the current
// tree level and sets the boxes as parameters in each work_item / node
extern void compute_bounding_boxes_fun(struct work_item* current_level_data, int total_children, struct point_set* input_set1, struct point_set* input_set2);

extern void traverse_with_dynamic_arrays_dynamic_output(struct work_item root_h, struct work_item** mat_vec_data, int* mat_vec_data_count, int* mat_vec_data_array_size, struct morton_code* input_set1_codes, struct morton_code* input_set2_codes, struct point_set* input_set1, struct point_set* input_set2, double eta, int max_level, int c_leaf, int max_elements_in_array);


#endif
