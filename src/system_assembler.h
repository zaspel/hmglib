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


