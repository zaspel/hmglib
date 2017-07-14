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

CFLAGS = -O3 -arch=sm_35 -g -Xcompiler -fPIC

all: hmglib_test

hmglib_test: hmglib_test.cu libhmglib.so
	nvcc $(CFLAGS) -L. -lhmglib -lcurand -lcublas hmglib_test.cu -o hmglib_test

morton.o: morton.cu morton.h
	nvcc $(CFLAGS) -c morton.cu -o morton.o

linear_algebra.o: linear_algebra.cu linear_algebra.h
	nvcc $(CFLAGS) -c linear_algebra.cu -o linear_algebra.o

tree.o: tree.cu tree.h
	nvcc $(CFLAGS) -c tree.cu -o tree.o

hmglib.o: hmglib.cu hmglib.h
	nvcc $(CFLAGS) -c hmglib.cu -o hmglib.o

helper.o: helper.cu helper.h
	nvcc $(CFLAGS) -c helper.cu -o helper.o

libhmglib.so: morton.o tree.o linear_algebra.o hmglib.o helper.o
	CC -lcudart -fPIC -o libhmglib.so --shared morton.o tree.o linear_algebra.o hmglib.o helper.o
