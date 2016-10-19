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

all:
	nvcc -O3 -g -arch=sm_50 -lcublas -lcurand tree_test.cu -o tree_test
