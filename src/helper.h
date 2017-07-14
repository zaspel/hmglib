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

#ifndef HELPER_H
#define HELPER_H

#include "morton.h"


#ifndef CHECK_CUDA_ERROR
#define CHECK_CUDA_ERROR
void checkCUDAError(const char* msg);
#endif

extern void print_binary(uint64_t val);

extern void get_min_and_max(double* min, double* max, double* values, int size);

extern void compute_minmax(struct point_set* points_d);

extern void get_morton_ordering(struct point_set* points_d, struct morton_code* morton_d, uint64_t* order);

extern void print_points(struct point_set* points_d);

extern void write_points(struct point_set* points_d, char* file_name);

extern void write_vector(double* x, int n, char* file_name);
	
extern void reorder_point_set(struct point_set* points_d, uint64_t* order);

extern void reorder_vector(double* vector, int vector_length, uint64_t* order);

extern void reorder_back_vector(double* vector, int vector_length, uint64_t* order);

extern void print_points_with_morton_codes(struct point_set* points_d, struct morton_code* code_d);

//extern void allocate_work_queue(struct work_queue **new_work_queue_h, struct work_queue **new_work_queue, int work_queue_size);
//
//extern void delete_work_queue(struct work_queue *new_work_queue_h, struct work_queue *new_work_queue);

#endif
