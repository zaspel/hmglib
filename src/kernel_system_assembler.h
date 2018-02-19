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

#ifndef kernel_system_assembler_h__
#define kernel_system_assembler_h__
        


#include "hmglib.h"
#include "system_assembler.h"

class gaussian_kernel_system_assembler : public system_assembler
{
	public:

		double regularization;


		__device__ double gaussian_kernel(double r)
		{
			return exp(-r*r);
		}

		__device__ double get_matrix_entry(int i, int j, struct point_set* point_set1_d, struct point_set* point_set2_d)
		{
			double result = 0.0;
			for (int d=0; d<point_set1_d->dim; d++)
			{
				result += (point_set1_d->coords[d][i]-point_set2_d->coords[d][j])*(point_set1_d->coords[d][i]-point_set2_d->coords[d][j]);
			}
			result =  sqrt(result);

			if (i==j)
				return gaussian_kernel(result) + regularization;
			else
				return gaussian_kernel(result);

		}

//		__device__ double get_rhs_entry(int i)
//		{
//			double result = 1.0;
//
//			for (int d=0; d<dim; d++)
//				result = result * (sqrt(M_PI)/2.0) * (erf(1.0- points[d][i]) - erf(0.0 - points[d][i]));
//
//			return result;
//		}

};

extern void create_gaussian_kernel_system_assembler_object(struct gaussian_kernel_system_assembler** assem, double regularization);

extern void destroy_gaussian_kernel_system_assembler_object(struct gaussian_kernel_system_assembler** assem);

class matern_kernel_system_assembler : public system_assembler
{
	public:

		double regularization;


		__device__ double matern_kernel(double r, int dim)
		{
			return (yn(1, r+1.0e-15)*pow(r,1.0))/(pow(2.0,(double)dim/2.0)*tgamma(1.0+(double)dim/2.0));
		}

		__device__ double get_matrix_entry(int i, int j, struct point_set* point_set1_d, struct point_set* point_set2_d)
		{
			double result = 0.0;
			for (int d=0; d<point_set1_d->dim; d++)
			{
				result += (point_set1_d->coords[d][i]-point_set2_d->coords[d][j])*(point_set1_d->coords[d][i]-point_set2_d->coords[d][j]);
			}
			result =  sqrt(result);

			if (i==j)
				return matern_kernel(result, point_set1_d->dim) + regularization;
			else
				return matern_kernel(result, point_set1_d->dim);

		}

};

extern void create_matern_kernel_system_assembler_object(struct matern_kernel_system_assembler** assem, double regularization);

extern void destroy_matern_kernel_system_assembler_object(struct matern_kernel_system_assembler** assem);

#endif
