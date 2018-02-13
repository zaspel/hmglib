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

#ifndef MORTON_H
#define MORTON_H

#include <stdint.h>

struct point_set
{
	double** coords;
	int	dim;
	int	size;
	double* max_per_dim;
	double* min_per_dim;
	unsigned int* point_ids;
};

struct morton_code
{
	uint64_t* code;
	int	size;
	int	dim;
	int	bits;
	double*	max_values;
	double* min_values;
};
	
#define MAX_DIM 20

	
extern void get_morton_code(struct point_set* points, struct morton_code* morton, int grid_size, int block_size);

#endif
