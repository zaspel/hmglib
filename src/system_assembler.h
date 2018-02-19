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

#ifndef system_assembler_h__
#define system_assembler_h__

#include "morton.h"

class system_assembler
{

	public:
//		int max_row_count_per_dgemv;

		virtual	__device__ double get_matrix_entry(int i, int j, struct point_set* point_set1_d, struct point_set* point_set2_d) =0;
//		virtual __device__ double get_rhs_entry(int i) =0;
//
};


#endif


