# Copyright (C) 2016 Peter Zaspel
#
# This file is part of hmglib.
#
# hmglib is free software: you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# hmglib is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with hmglib.  If not, see <http://www.gnu.org/licenses/>.

CFLAGS = -Xcompiler="-fPIC -DADD_" -O3 -gencode arch=compute_60,code=sm_60 -DADD_
GSL_LIB = -I/home-2/pzaspel/gsllib/include -L/home-2/pzaspel/gsllib/lib -lgsl -lgslcblas -lm

MAGMA_LIB= -L/home-2/pzaspel/magmalib/lib -lmagma -L/home-2/pzaspel/OpenBLASlib/lib -lopenblas -lcusparse -lcudart -lcudadevrt
MAGMA_INC= -I/home-2/pzaspel/magmalib/include

all: hmglib_test paper_convergence_test paper_batching_benchmark

hmglib_test: hmglib_test.cu libhmglib.so
	nvcc $(CFLAGS) $(MAGMA_INC) $(MAGMA_LIB) -L. -lhmglib -lcurand -lcublas hmglib_test.cu -o hmglib_test

paper_convergence_test: paper_convergence_test.cu libhmglib.so
	nvcc $(CFLAGS) $(GSL_LIB) $(MAGMA_INC) $(MAGMA_LIB) -L. -lhmglib -lcurand -lcublas paper_convergence_test.cu -o paper_convergence_test

paper_batching_benchmark: paper_batching_benchmark.cu libhmglib.so
	nvcc $(CFLAGS) $(GSL_LIB) $(MAGMA_INC) $(MAGMA_LIB) -L. -lhmglib -lcurand -lcublas paper_batching_benchmark.cu -o paper_batching_benchmark

morton.o: morton.cu morton.h
	nvcc $(CFLAGS) -c morton.cu -o morton.o

linear_algebra.o: linear_algebra.cu linear_algebra.h
	nvcc $(CFLAGS) $(MAGMA_INC) -c linear_algebra.cu -o linear_algebra.o

tree.o: tree.cu tree.h
	nvcc $(CFLAGS) -c tree.cu -o tree.o

hmglib.o: hmglib.cu hmglib.h
	nvcc $(CFLAGS) $(MAGMA_INC) -c hmglib.cu -o hmglib.o

helper.o: helper.cu helper.h
	nvcc $(CFLAGS) -c helper.cu -o helper.o

libhmglib.so: morton.o tree.o linear_algebra.o hmglib.o helper.o
	nvcc $(CFLAGS) $(MAGMA_LIB) -o libhmglib.so --shared morton.o tree.o linear_algebra.o hmglib.o helper.o
