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
#include "cub/cub.cuh"
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/unique.h>

#ifndef CHECK_CUDA_ERROR
#define CHECK_CUDA_ERROR
void checkCUDAError(const char* msg) {
cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}
#endif


//! computation of the split within a morton code based tree
//! this codes is taken from
//! "Thinking Parallel, Part III: Tree Construction on the GPU"
__device__ __forceinline__ int findSplit( struct morton_code *codes, int first, int last)
{
    // Identical Morton codes => split the range in the middle.

    uint64_t firstCode = codes->code[first];
    uint64_t lastCode = codes->code[last];

    if (firstCode == lastCode)
        return (first + last) >> 1;

    // Calculate the number of highest bits that are the same
    // for all objects, using the count-leading-zeros intrinsic.

    int commonPrefix = __clzll(firstCode ^ lastCode);

    // Use binary search to find where the next bit differs.
    // Specifically, we are looking for the highest object that
    // shares more than commonPrefix bits with the first one.

    int split = first; // initial guess
    int step = last - first;

    do
    {
        step = (step + 1) >> 1; // exponential decrease
        int newSplit = split + step; // proposed new position

        if (newSplit < last)
        {
        	uint64_t splitCode = codes->code[newSplit];
            int splitPrefix = __clzll(firstCode ^ splitCode);
            if (splitPrefix > commonPrefix)
                split = newSplit; // accept proposal
        }
    }
    while (step > 1);

    return split;
}





#define WARP_SIZE 32
#define BLOCK_SIZE 512
#define WT_ACA 1
#define WT_DENSE 2
struct work_item
{
	int set1_l;
	int set1_u;
	int set2_l;
	int set2_u;
	int work_type;
	int is_in_use;
	double max1[5];
	double min1[5];
	double max2[5];
	double min2[5];
	int dim;
};

struct work_queue
{
	struct work_item* head;
	struct work_item* tail;
	struct work_item* data;
	int queue_size;

	__host__ __forceinline__ void init(int _queue_size)
	{
		queue_size = _queue_size;
		cudaMalloc((void**)&data, queue_size * sizeof(struct work_item));
		head = data;
		tail = data;
		printf("init: %p\n",data);
	}

	__host__ __forceinline__ void destroy()
	{
		cudaFree(data);
	}

	__host__ __device__ __forceinline__ bool is_empty()
	{
		return head >= tail;
	}

	__device__ __forceinline__ void warp_single_put(struct work_item item, int warp_lane)
	{
		// TODO: ignoring queue limits check
		if (warp_lane==0)
		{
			struct work_item* old_tail = (struct work_item*)atomicAdd((unsigned long long*)&tail, sizeof(struct work_item));
			old_tail[0] = item;
		}
	}



//	__host__ __device__ __forceinline__ void warp_put(T* new_data, int warp_lane)
//	{
//		atomicAdd(tail, WARP_SIZE);
//		tail[warp_idx] = new_data
//		// TODO: hier fehlt abfrage, ob am Ende angekommen
//		
//		
//	}

	__device__ __forceinline__ bool warp_get(struct work_item& destination, int warp_lane)
	{
		if (this->is_empty())  // make sure that there is at least a single element in the queue
		{
			destination.set1_l=-1;
			return false;
		}
		else
		{
/*
			// data extraction with bounds checking
			if (&(tail[warp_lane-(WARP_SIZE)])>=head)  // valid memory access
				destination = tail[warp_lane-WARP_SIZE];
			else
				destination.set1_l=-1;  // avoid invalid memory access and set work item to be invalid

			// move tail back (with bounds checking)
			int jump_offset = max((long)(-WARP_SIZE), head-tail); // calculate offset (either the full warp size or offset necessary to achieve tail == head)
			if (warp_lane == 0)
				atomicAdd((unsigned long long*)&tail, jump_offset*sizeof(struct work_item));
*/


			// push head forward (with bounds checking)
			struct work_item* old_head;
//			int jump_offset = min((long)(WARP_SIZE), tail-head); // calculate offset (either the full warp size or offset necessary to achieve tail == head)

			// TODO: a fixed jump size of WARP_SIZE might cause problems !!!!!!
			if (warp_lane == 0)
				old_head = (struct work_item*)atomicAdd((unsigned long long int *)&head, WARP_SIZE*sizeof(struct work_item));
			struct work_item* new_head = head;  // BUG: This is not thread-safe!!!
			// broadcast old_head to all threads of this warp
			old_head = (struct work_item*)__shfl((unsigned long)old_head, 0);



			// wait until either there is a valid working item in the work item set
			// that was taken by this warp, or until the queue is empty
			while ((!this->is_empty()) && (old_head[0].set1_l==-1)) {;}

			// if I get to this point, I have either an empty queue or some work
			// in case of the empty queue (and no useful work): stop
			if (this->is_empty() && (old_head[0].set1_l==-1))
			{
				return false
;
			}

			// second case: work is available (with or without (recently become) empty queue)
			// data extraction with bounds checking
//			if (warp_lane < new_head-old_head)  // valid memory access
				destination = old_head[warp_lane];
/*			else
			{
				destination.set1_l=-1;  // avoid invalid memory access and set work item to be invalid
				destination.set1_u=-1;  // avoid invalid memory access and set work item to be invalid
				destination.set2_l=-1;  // avoid invalid memory access and set work item to be invalid
				destination.set2_u=-1;  // avoid invalid memory access and set work item to be invalid
			}*/
//			if (warp_lane==0)
//				printf("Warp lane 0 get data at pointer %p\n", &old_head[warp_lane]);
	
			return true;
		}
	}

	__device__ __forceinline__ bool single_get(struct work_item& destination, int warp_lane)
	{
		if (this->is_empty())  // make sure that there is at least a single element in the queue
		{
			destination.set1_l=-1;
			return false;
		}
		else
		{
			// push head forward (with bounds checking)
			struct work_item* old_head;

			if (head<tail)
				old_head = (struct work_item*)atomicAdd((unsigned long long int *)&head, sizeof(struct work_item));
			else
				return false;

//			// wait until working item gets valid or until the queue is empty
//			while ((!this->is_empty()) && (old_head[0].set1_l==-1)) {;}

//			// if I get to this point, I have either an empty queue or some work
//			// in case of the empty queue (and no useful work): stop
//			if (this->is_empty() && (old_head[0].set1_l==-1))
//			{
//				return false;
//			}

			// second case: work is available (with or without (recently become) empty queue)
			// data extraction with bounds checking
			destination = old_head[0];

			return true;
		}
	}



};



__host__ __device__ __forceinline__ bool is_valid_work_item(struct work_item item)
{
	return (item.set1_l!=-1);
//	return ((item.set1_l!=-1) && (item.set1_u!=-1) && (item.set2_l!=-1) && (item.set2_u!=-1));
//	return ((item.set1_l>0) && (item.set1_l<100));
}


__host__ __device__ __forceinline__ bool at_least_one_block_smaller_than_threshold(struct work_item item, int c_leaf)
{
	return (((item.set1_u-item.set1_l+1)<=c_leaf) || ((item.set2_u-item.set2_l+1)<=c_leaf));
}


__host__ __device__ __forceinline__ double compute_diameter(double* min, double* max, int dim)
{
	double diam = 0.0;

	for (int d=0; d<dim; d++)
	{
		diam += (max[d]-min[d]) * (max[d]-min[d]);
	}
	diam = sqrt(diam);

	return diam;
}

__host__ __device__ __forceinline__ double compute_distance(double* min1, double* max1, double* min2, double* max2, int dim)
{
	double dist = 0.0;

	for (int d=0; d<dim; d++)
	{
		dist += (fmax(0.0, min1[d]-max2[d]) * fmax(0.0, min1[d]-max2[d])) + (fmax(0.0, min2[d]-max1[d]) * fmax(0.0, min2[d]-max1[d]));
	}
	dist = sqrt(dist);

	return dist;
}


__host__ __device__ __forceinline__ bool bounding_box_admissibility(struct work_item item, struct point_set* input_set1, struct point_set* input_set2, double eta)
{
/*
	int l1 = item.set1_l;
	int u1 = item.set1_u;
	int l2 = item.set2_l;
	int u2 = item.set2_u;

	int dim = input_set1->dim;
	double min1[5];
	double max1[5];
	double min2[5];
	double max2[5];

	for (int d=0; d<dim; d++)
	{
		min1[d] = input_set1->coords[d][l1];
		max1[d] = input_set1->coords[d][l1];
		min2[d] = input_set2->coords[d][l2];
		max2[d] = input_set2->coords[d][l2];
		for (int i=l1+1; i<=u1; i++)
		{
			min1[d] = fmin(min1[d], input_set1->coords[d][i]);
			max1[d] = fmax(max1[d], input_set1->coords[d][i]);
		}
		for (int i=l2+1; i<=u2; i++)
		{
			min2[d] = fmin(min2[d], input_set2->coords[d][i]);
			max2[d] = fmax(max2[d], input_set2->coords[d][i]);
		}
	}
*/
	double diam1 = compute_diameter(item.min1, item.max1, item.dim);
	double diam2 = compute_diameter(item.min2, item.max2, item.dim);

	double dist = compute_distance(item.min1, item.max1, item.min2, item.max2, item.dim);

	return (fmin(diam1, diam2) <= (eta*dist));
}

	__global__ void traverse(work_queue *mat_vec_work_queue, struct work_queue **tree_work_queue, struct morton_code* input_set1_codes, struct morton_code* input_set2_codes, struct point_set* input_set1, struct point_set* input_set2, double eta, int max_level, int c_leaf)
{
//	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int warp_lane = threadIdx.x % WARP_SIZE;
	int warp_id = threadIdx.x / WARP_SIZE;
	
	
	struct work_item work;
	int child_count;  // does not make any sense
	int mat_vec_work_item_count;
	int mat_vec_work_item_output_offset;
	int mat_vec_work_item_total_output;
	int output_offset;  // does not make any sense
	int total_output;

	struct work_item* old_tree_work_queue_tail;
	struct work_item* old_mat_vec_work_queue_tail;


	typedef cub::WarpScan<int> WarpScan;

	__shared__ typename WarpScan::TempStorage temp_storage[BLOCK_SIZE/WARP_SIZE];
	
	bool is_admissible;


	for ( int current_level = 0; current_level < max_level; current_level++  )
	{	
//		if (current_level > 0)
//			while(!tree_work_queue[current_level]->is_empty()) {;}

		while (!tree_work_queue[current_level]->is_empty())
		{
			if (tree_work_queue[current_level]->warp_get(work, warp_lane))
			{

			// TODO: max_level management -> what to do if I am on level max_level-1 an want to split to a next level ????  <= done ???

			// TODO: Initialize the task queue with invalid work items <-- necessary ??
			// TODO: set work items to invalid before dequeueing  <-- necessary ??

			// start by counting children
			if (!is_valid_work_item(work))
			{
				child_count = 0;
				mat_vec_work_item_count = 0;
			}
			else // we are having a valid work item
			{
	
				is_admissible = bounding_box_admissibility(work, input_set1, input_set2, eta);
			
				if (is_admissible)
				{
					// ACA
					child_count = 0;
					mat_vec_work_item_count = 1;
				}			
				else if (at_least_one_block_smaller_than_threshold(work, c_leaf) || (current_level+1)>=max_level )
				{
					// dense MVP
					child_count = 0;
					mat_vec_work_item_count = 1;
				}
				else
				{
					// create children
					child_count = 4;
					mat_vec_work_item_count = 0;
				}
			}

			// compute offset for children and do updating (in parallel)
			// this update process is a warp-wide collective operation and therefore has to include invalid
			// work items (-> warp lanes on invalid work items still have to take part in the collective operation)

			WarpScan(temp_storage[warp_id]).ExclusiveSum( child_count, output_offset, total_output );

			__syncthreads();

			if (total_output > 0)  // this implicitly makes sure that there is no atomicAdd on the highest level (wrt. the highest + 1 level)
								   // also, this is a useful sanity check
			{
				if (warp_lane == 0)
				{
					old_tree_work_queue_tail = (struct work_item*)atomicAdd((unsigned long long*)&(tree_work_queue[current_level+1]->tail), total_output*sizeof(struct work_item));
				}
				// broadcast old_tree_work_queue_tail to all warps
				old_tree_work_queue_tail = (struct work_item*)__shfl((unsigned long)old_tree_work_queue_tail, 0);
			}

			WarpScan(temp_storage[warp_id]).ExclusiveSum( mat_vec_work_item_count, mat_vec_work_item_output_offset, mat_vec_work_item_total_output );

			__syncthreads();

			if (warp_lane == 0)
			{
				old_mat_vec_work_queue_tail = (struct work_item*)atomicAdd((unsigned long long*)&(mat_vec_work_queue->tail), mat_vec_work_item_total_output*sizeof(struct work_item));
			}
			// broadcast old_mat_vec_work_queue_tail to all warps
			old_mat_vec_work_queue_tail = (struct work_item*)__shfl((unsigned long)old_mat_vec_work_queue_tail, 0);


			// the real work is now only done for the valid work items
			if (is_valid_work_item(work))
			{
				// do work
	
				if (is_admissible)
				{
					// ACA
//					atomicAdd((unsigned long long*)&(mat_vec_work_queue->tail),sizeof(struct work_item));
	
					struct work_item aca_item;
					aca_item = work;
					aca_item.work_type = WT_ACA;
	
					old_mat_vec_work_queue_tail[mat_vec_work_item_output_offset] = aca_item;
				}
				else if (at_least_one_block_smaller_than_threshold(work, c_leaf) || (current_level+1)>=max_level )
				{
					// dense MVP
//					atomicAdd((unsigned long long*)&(mat_vec_work_queue->tail),sizeof(struct work_item));
	
					struct work_item dense_item;
					dense_item = work;
					dense_item.work_type = WT_DENSE;

					old_mat_vec_work_queue_tail[mat_vec_work_item_output_offset] = dense_item;
				}
				else
				{
					// create children
					struct work_item child11, child12, child21, child22;
					
					int split_set1 = findSplit( input_set1_codes, work.set1_l, work.set1_u);
					int split_set2 = findSplit( input_set2_codes, work.set2_l, work.set2_u);
					
					child11.set1_l = work.set1_l;
					child11.set1_u = split_set1;
					child11.set2_l = work.set2_l;
					child11.set2_u = split_set2;
	
					child12.set1_l = work.set1_l;
					child12.set1_u = split_set1;
					child12.set2_l = split_set2+1;
					child12.set2_u = work.set2_u;

					child21.set1_l = split_set1+1;
					child21.set1_u = work.set1_u;
					child21.set2_l = work.set1_l;
					child21.set2_u = split_set2;
	
					child22.set1_l = split_set1+1;  // correkt ?
					child22.set1_u = work.set1_u;
					child22.set2_l = split_set2+1;
					child22.set2_u = work.set2_u;
		
					// insert new children in work queue
					old_tree_work_queue_tail[output_offset]=child11;				
					old_tree_work_queue_tail[output_offset+1]=child12;				
					old_tree_work_queue_tail[output_offset+2]=child21;				
					old_tree_work_queue_tail[output_offset+3]=child22;				
				}
			}
			}
		}

	}
}

/*__global__ void traverse_single(work_queue *mat_vec_work_queue, struct work_queue **tree_work_queue, struct morton_code* input_set1_codes, struct morton_code* input_set2_codes, struct point_set* input_set1, struct point_set* input_set2, double eta, int max_level, int c_leaf)
{
//	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int warp_lane = threadIdx.x % WARP_SIZE;
	int warp_id = threadIdx.x / WARP_SIZE;


	struct work_item work;
	int child_count;  // does not make any sense
	int mat_vec_work_item_count;
	int mat_vec_work_item_output_offset;
	int mat_vec_work_item_total_output;
	int output_offset;  // does not make any sense
	int total_output;

	struct work_item* old_tree_work_queue_tail;
	struct work_item* old_mat_vec_work_queue_tail;


	bool is_admissible;


	for ( int current_level = 0; current_level < max_level; current_level++  )
	{
//		if (current_level > 0)
//			while(!tree_work_queue[current_level]->is_empty()) {;}

		while (!tree_work_queue[current_level]->is_empty())
		{
			if (tree_work_queue[current_level]->single_get(work, warp_lane))
			{

			// TODO: max_level management -> what to do if I am on level max_level-1 an want to split to a next level ????  <= done ???

			// TODO: Initialize the task queue with invalid work items <-- necessary ??
			// TODO: set work items to invalid before dequeueing  <-- necessary ??

			// start by counting children
			if (!is_valid_work_item(work))
			{
				child_count = 0;
				mat_vec_work_item_count = 0;
			}
			else // we are having a valid work item
			{

				is_admissible = bounding_box_admissibility(work, input_set1, input_set2, eta);

				if (is_admissible)
				{
					// ACA
					child_count = 0;
					mat_vec_work_item_count = 1;
				}
				else if (at_least_one_block_smaller_than_threshold(work, c_leaf) || (current_level+1)>=max_level )
				{
					// dense MVP
					child_count = 0;
					mat_vec_work_item_count = 1;
				}
				else
				{
					// create children
					child_count = 4;
					mat_vec_work_item_count = 0;
				}
			}

			// compute offset for children and do updating (in parallel)
			// this update process is a warp-wide collective operation and therefore has to include invalid
			// work items (-> warp lanes on invalid work items still have to take part in the collective operation)

			if (child_count > 0)  // this implicitly makes sure that there is no atomicAdd on the highest level (wrt. the highest + 1 level)
								   // also, this is a useful sanity check
			{
				old_tree_work_queue_tail = (struct work_item*)atomicAdd((unsigned long long*)&(tree_work_queue[current_level+1]->tail), child_count*sizeof(struct work_item));
			}

			old_mat_vec_work_queue_tail = (struct work_item*)atomicAdd((unsigned long long*)&(mat_vec_work_queue->tail), mat_vec_work_item_count*sizeof(struct work_item));

			// the real work is now only done for the valid work items
			if (is_valid_work_item(work))
			{
				// do work

				if (is_admissible)
				{
					// ACA
//					atomicAdd((unsigned long long*)&(mat_vec_work_queue->tail),sizeof(struct work_item));

					struct work_item aca_item;
					aca_item = work;
					aca_item.work_type = WT_ACA;

					old_mat_vec_work_queue_tail[0] = aca_item;
				}
				else if (at_least_one_block_smaller_than_threshold(work, c_leaf) || (current_level+1)>=max_level )
				{
					// dense MVP
//					atomicAdd((unsigned long long*)&(mat_vec_work_queue->tail),sizeof(struct work_item));

					struct work_item dense_item;
					dense_item = work;
					dense_item.work_type = WT_DENSE;

					old_mat_vec_work_queue_tail[0] = dense_item;
				}
				else
				{
					// create children
					struct work_item child11, child12, child21, child22;

					int split_set1 = findSplit( input_set1_codes, work.set1_l, work.set1_u);
					int split_set2 = findSplit( input_set2_codes, work.set2_l, work.set2_u);

					child11.set1_l = work.set1_l;
					child11.set1_u = split_set1;
					child11.set2_l = work.set2_l;
					child11.set2_u = split_set2;

					child12.set1_l = work.set1_l;
					child12.set1_u = split_set1;
					child12.set2_l = split_set2+1;
					child12.set2_u = work.set2_u;

					child21.set1_l = split_set1+1;
					child21.set1_u = work.set1_u;
					child21.set2_l = work.set1_l;
					child21.set2_u = split_set2;

					child22.set1_l = split_set1+1;  // correkt ?
					child22.set1_u = work.set1_u;
					child22.set2_l = split_set2+1;
					child22.set2_u = work.set2_u;

					// insert new children in work queue
					old_tree_work_queue_tail[0]=child11;
					old_tree_work_queue_tail[1]=child12;
					old_tree_work_queue_tail[2]=child21;
					old_tree_work_queue_tail[3]=child22;
				}
			}
			}
		}

	}
}
*/

/*
__global__ void traverse_single_single_queue(work_queue *mat_vec_work_queue, struct work_queue **tree_work_queue, struct morton_code* input_set1_codes, struct morton_code* input_set2_codes, struct point_set* input_set1, struct point_set* input_set2, double eta, int max_level, int c_leaf)
{
//	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int warp_lane = threadIdx.x % WARP_SIZE;
	int warp_id = threadIdx.x / WARP_SIZE;


	struct work_item work;
	int child_count;  // does not make any sense
	int mat_vec_work_item_count;
	int mat_vec_work_item_output_offset;
	int mat_vec_work_item_total_output;
	int output_offset;  // does not make any sense
	int total_output;

	struct work_item* old_tree_work_queue_tail;
	struct work_item* old_mat_vec_work_queue_tail;


	bool is_admissible;

	int current_level = 0;

//		if (current_level > 0)
//			while(!tree_work_queue[current_level]->is_empty()) {;}

 		while (!tree_work_queue[current_level]->is_empty())
		{
			if (tree_work_queue[current_level]->single_get(work, warp_lane))
			{

			// TODO: max_level management -> what to do if I am on level max_level-1 an want to split to a next level ????  <= done ???

			// TODO: Initialize the task queue with invalid work items <-- necessary ??
			// TODO: set work items to invalid before dequeueing  <-- necessary ??

			// start by counting children
			if (!is_valid_work_item(work))
			{
				child_count = 0;
				mat_vec_work_item_count = 0;
			}
			else // we are having a valid work item
			{

				is_admissible = bounding_box_admissibility(work, input_set1, input_set2, eta);

				if (is_admissible)
				{
					// ACA
					child_count = 0;
					mat_vec_work_item_count = 1;
				}
				else if (at_least_one_block_smaller_than_threshold(work, c_leaf) || (current_level+1)>=max_level )
				{
					// dense MVP
					child_count = 0;
					mat_vec_work_item_count = 1;
				}
				else
				{
					// create children
					child_count = 4;
					mat_vec_work_item_count = 0;
				}
			}

			// compute offset for children and do updating (in parallel)
			// this update process is a warp-wide collective operation and therefore has to include invalid
			// work items (-> warp lanes on invalid work items still have to take part in the collective operation)

			if (child_count > 0)  // this implicitly makes sure that there is no atomicAdd on the highest level (wrt. the highest + 1 level)
								   // also, this is a useful sanity check
			{
				old_tree_work_queue_tail = (struct work_item*)atomicAdd((unsigned long long*)&(tree_work_queue[current_level]->tail), child_count*sizeof(struct work_item));
			}

			old_mat_vec_work_queue_tail = (struct work_item*)atomicAdd((unsigned long long*)&(mat_vec_work_queue->tail), mat_vec_work_item_count*sizeof(struct work_item));

			// the real work is now only done for the valid work items
			if (is_valid_work_item(work))
			{
				// do work
http://www.tagesschau.de/
				if (is_admissible)
				{
					// ACA
//					atomicAdd((unsigned long long*)&(mat_vec_work_queue->tail),sizeof(struct work_item));

					struct work_item aca_item;
					aca_item = work;
					aca_item.work_type = WT_ACA;

					old_mat_vec_work_queue_tail[0] = aca_item;
				}
				else if (at_least_one_block_smaller_than_threshold(work, c_leaf) || (current_level+1)>=max_level )
				{
					// dense MVP
//					atomicAdd((unsigned long long*)&(mat_vec_work_queue->tail),sizeof(struct work_item));

					struct work_item dense_item;
					dense_item = work;
					dense_item.work_type = WT_DENSE;

					old_mat_vec_work_queue_tail[0] = dense_item;
				}
				else
				{
					// create children
					struct work_item child11, child12, child21, child22;

					int split_set1 = findSplit( input_set1_codes, work.set1_l, work.set1_u);
					int split_set2 = findSplit( input_set2_codes, work.set2_l, work.set2_u);

					child11.set1_l = work.set1_l;
					child11.set1_u = split_set1;
					child11.set2_l = work.set2_l;
					child11.set2_u = split_set2;

					child12.set1_l = work.set1_l;
					child12.set1_u = split_set1;
					child12.set2_l = split_set2+1;
					child12.set2_u = work.set2_u;

					child21.set1_l = split_set1+1;
					child21.set1_u = work.set1_u;
					child21.set2_l = work.set1_l;
					child21.set2_u = split_set2;

					child22.set1_l = split_set1+1;  // correkt ?
					child22.set1_u = work.set1_u;
					child22.set2_l = split_set2+1;
					child22.set2_u = work.set2_u;

					// insert new children in work queue
					old_tree_work_queue_tail[0]=child11;
					old_tree_work_queue_tail[1]=child12;
					old_tree_work_queue_tail[2]=child21;
					old_tree_work_queue_tail[3]=child22;
				}
			}
			}
			__syncthreads();
		}

}
*/

void print_work_items(struct work_item* work_items, int work_item_count)
{
	for (int i=0; i<work_item_count; i++)
	{
		struct work_item item;
		cudaMemcpy(&item, &work_items[i], sizeof(struct work_item), cudaMemcpyDeviceToHost);
		printf("S1_L: %d,  S1_U: %d,  S2_L: %d,  S2_U: %d,  work_type: %s\n", item.set1_l, item.set1_u, item.set2_l, item.set2_u, (item.work_type==WT_ACA) ? "ACA" : "DENSE");
	}
}

void write_work_items(char* file_name, struct work_item* work_items, int work_item_count)
{
	FILE* f= fopen(file_name, "w");
	for (int i=0; i<work_item_count; i++)
	{
		struct work_item item;
		cudaMemcpy(&item, &work_items[i], sizeof(struct work_item), cudaMemcpyDeviceToHost);
		fprintf(f, "S1_L: %d,  S1_U: %d,  S2_L: %d,  S2_U: %d,  work_type: %s\n", item.set1_l, item.set1_u, item.set2_l, item.set2_u, (item.work_type==WT_ACA) ? "ACA" : "DENSE");
	}
	fclose(f);
}



__global__ void init_tree_array_root(struct work_item *current_level_data, struct work_item item)
{
	current_level_data[0] = item;
}

__global__ void invalidate_array(struct work_item* items, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx>=size)
		return;

	items[idx].set1_l = -1;
	items[idx].set1_u = -1;
	items[idx].set2_l = -1;
	items[idx].set2_u = -1;
}

__global__ void set_array(int* a, int value, int count)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx>=count)
		return;

	a[idx] = value;
}

__global__ void count_for_new_level(struct work_item *current_level_data, struct work_item* next_level_data, int* new_mat_vec_counts, int* new_child_counts, int total_children, struct morton_code* input_set1_codes, struct morton_code* input_set2_codes, struct point_set* input_set1, struct point_set* input_set2, double eta, int current_level, int max_level, int c_leaf)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= total_children)
		return;

	struct work_item work = current_level_data[idx];

	// start by counting children
	if (!is_valid_work_item(work))
	{
		new_child_counts[idx] = 0;
		new_mat_vec_counts[idx] = 0;
	}
	else // we are having a valid work item
	{
		bool is_admissible = bounding_box_admissibility(work, input_set1, input_set2, eta);

		if (is_admissible)
		{
			// ACA
			new_child_counts[idx] = 0;
			new_mat_vec_counts[idx] = 1;
		}
		else if (at_least_one_block_smaller_than_threshold(work, c_leaf) || (current_level+1)>=max_level )
		{
			// dense MVP
			new_child_counts[idx] = 0;
			new_mat_vec_counts[idx] = 1;
		}
		else
		{
			// create children
			new_child_counts[idx] = 4;
			new_mat_vec_counts[idx] = 0;
		}
	}

}

__global__ void compute_bounding_boxes(struct work_item *current_level_data, int total_children, struct point_set* input_set1, struct point_set* input_set2)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= total_children)
		return;

	struct work_item* work = &current_level_data[idx];

	int l1 = work->set1_l;
	int u1 = work->set1_u;
	int l2 = work->set2_l;
	int u2 = work->set2_u;

	int dim = input_set1->dim;
	work->dim = dim;

	double min1[5];
	double max1[5];
	double min2[5];
	double max2[5];

	for (int d=0; d<dim; d++)
	{
		min1[d] = input_set1->coords[d][l1];
		max1[d] = input_set1->coords[d][l1];
		min2[d] = input_set2->coords[d][l2];
		max2[d] = input_set2->coords[d][l2];
		for (int i=l1+1; i<=u1; i++)
		{
			min1[d] = fmin(min1[d], input_set1->coords[d][i]);
			max1[d] = fmax(max1[d], input_set1->coords[d][i]);
		}
		for (int i=l2+1; i<=u2; i++)
		{
			min2[d] = fmin(min2[d], input_set2->coords[d][i]);
			max2[d] = fmax(max2[d], input_set2->coords[d][i]);
		}
	}

	for (int d=0; d<dim; d++)
	{
		work->min1[d] = min1[d];
		work->max1[d] = max1[d];
		work->min2[d] = min2[d];
		work->max2[d] = max2[d];
	}

}



__global__ void generate_new_level(struct work_item *current_level_data, struct work_item* next_level_data, struct work_item* mat_vec_data_at_current_offset, int* child_counts, int total_children, int* new_mat_vec_offsets, int* new_child_offsets, struct morton_code* input_set1_codes, struct morton_code* input_set2_codes, struct point_set* input_set1, struct point_set* input_set2, double eta, int current_level, int max_level, int c_leaf)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= total_children)
		return;
	// compute offset for children and do updating (in parallel)
	// this update process is a warp-wide collective operation and therefore has to include invalid
	// work items (-> warp lanes on invalid work items still have to take part in the collective operation)

	struct work_item work = current_level_data[idx];

	if (child_counts[idx]>0)
	{
		int offset = new_child_offsets[idx];

		// create children
		struct work_item child11, child12, child21, child22;

		int split_set1 = findSplit( input_set1_codes, work.set1_l, work.set1_u);
		int split_set2 = findSplit( input_set2_codes, work.set2_l, work.set2_u);

		child11.set1_l = work.set1_l;
		child11.set1_u = split_set1;
		child11.set2_l = work.set2_l;
		child11.set2_u = split_set2;

		child12.set1_l = work.set1_l;
		child12.set1_u = split_set1;
		child12.set2_l = split_set2+1;
		child12.set2_u = work.set2_u;

		child21.set1_l = split_set1+1;
		child21.set1_u = work.set1_u;
		child21.set2_l = work.set2_l;
		child21.set2_u = split_set2;

		child22.set1_l = split_set1+1;  // correct ?
		child22.set1_u = work.set1_u;
		child22.set2_l = split_set2+1;
		child22.set2_u = work.set2_u;

		// insert new children in work queue
		next_level_data[offset]=child11;
		next_level_data[offset+1]=child12;
		next_level_data[offset+2]=child21;
		next_level_data[offset+3]=child22;
	}
	else
	{
		bool is_admissible = bounding_box_admissibility(work, input_set1, input_set2, eta);

		if (is_admissible)
			work.work_type = WT_ACA;
		else
			work.work_type = WT_DENSE;

		mat_vec_data_at_current_offset[new_mat_vec_offsets[idx]] = work;
	}

}

cudaEvent_t sstart, sstop;
float mmilliseconds;

#define TIME_sstart {cudaEventCreate(&sstart); cudaEventCreate(&sstop); cudaEventRecord(sstart);}
#define TIME_sstop(a) {cudaEventRecord(sstop); cudaEventSynchronize(sstop); cudaEventElapsedTime(&mmilliseconds, sstart, sstop); printf("%s: Elapsed time: %lf ms\n", a, mmilliseconds); }

void traverse_with_arrays(struct work_item root_h, struct work_item* mat_vec_data, int* mat_vec_data_count, struct morton_code* input_set1_codes, struct morton_code* input_set2_codes, struct point_set* input_set1, struct point_set* input_set2, double eta, int max_level, int c_leaf, int max_elements_in_array)
{
	struct work_item* current_level_data;
	struct work_item* next_level_data;

	cudaMalloc((void**)&current_level_data, max_elements_in_array*sizeof(struct work_item));
	cudaMalloc((void**)&next_level_data, max_elements_in_array*sizeof(struct work_item));

	// calculate GPU thread configuration
	int block_size = 512;
	int grid_size = (max_elements_in_array + (block_size - 1)) / block_size;

	// invalidate tree level data arrays
	// TODO: The following two kernel calls require a relatively large amount of runtime; what to do?
	invalidate_array<<<grid_size, block_size>>>(current_level_data, max_elements_in_array);
	checkCUDAError("invalidate_array0");
	invalidate_array<<<grid_size, block_size>>>(next_level_data, max_elements_in_array);
	checkCUDAError("invalidate_array1");

	// fill initial node into first tree level
	init_tree_array_root<<<1, 1>>>(current_level_data, root_h);
	checkCUDAError("init_tree_array_root");

	int* new_mat_vec_counts;   // number of MatVecs that will be generated per valid node of the current tree level
	int* new_child_counts;     // number of child nodes that will be generated per valid node of the current tree level
	int* new_child_offsets;    // storage for the offsets for the new child nodes in the next tree level

	// allocation
	cudaMalloc((void**)&new_mat_vec_counts, max_elements_in_array*sizeof(int));
	cudaMalloc((void**)&new_child_counts, max_elements_in_array*sizeof(int));
	cudaMalloc((void**)&new_child_offsets, max_elements_in_array*sizeof(int));
	// pointer fun
	thrust::device_ptr<int> new_mat_vec_counts_ptr(new_mat_vec_counts);
	thrust::device_ptr<int> new_child_counts_ptr(new_child_counts);
	thrust::device_ptr<int> new_child_offsets_ptr(new_child_offsets);

	int total_new_mat_vecs;  // temp field to store total number of new MatVecs of the current level

	int total_children = 1;  // number of nodes on current level
	int total_new_children;  // temp field to store total number of new nodes on next level
	int grid_size_for_children = (total_children + (block_size - 1)) / block_size;  // field to store the grid size for
																					// kernels that follow the node count

	struct work_item* mat_vec_data_at_current_offset = mat_vec_data;	// array to store the current tail of the queue / array that holds the MatVecs

	for (int current_level=0; current_level<max_level; current_level++)  // run over all arrays
	{
		set_array<<<grid_size_for_children, block_size>>>(new_child_counts, 0, total_children);  //  will compute new child counts for total_children nodes
		checkCUDAError("set_array");

		// find number of children & MatVecs per node on current level
		count_for_new_level<<<grid_size_for_children, block_size>>>(current_level_data, next_level_data, new_mat_vec_counts, new_child_counts, total_children, input_set1_codes, input_set2_codes, input_set1, input_set2, eta, current_level, max_level, c_leaf);
		checkCUDAError("count_for_new_level");

		// compute total number of new children & MatVecs
		total_new_mat_vecs = thrust::reduce(new_mat_vec_counts_ptr, new_mat_vec_counts_ptr+total_children);
		total_new_children = thrust::reduce(new_child_counts_ptr, new_child_counts_ptr+total_children);

		*mat_vec_data_count = *mat_vec_data_count + total_new_mat_vecs;

		// compute node offsets in new level & offsets for MatVecs
		thrust::exclusive_scan(new_mat_vec_counts_ptr, new_mat_vec_counts_ptr+total_children, new_mat_vec_counts_ptr);   // here, I reuse the field to store the offsets (for memory efficiency reasons)
		thrust::exclusive_scan(new_child_counts_ptr, new_child_counts_ptr+total_children, new_child_offsets_ptr);  // here, I store the offsets in a dedicated field

		// generate new level with nodes and write new MatVecs into queue
		generate_new_level<<<grid_size_for_children, block_size>>>(current_level_data, next_level_data, mat_vec_data_at_current_offset, new_child_counts, total_children, new_mat_vec_counts, new_child_offsets, input_set1_codes, input_set2_codes, input_set1, input_set2, eta, current_level, max_level, c_leaf);
		checkCUDAError("generate_new_level");

		// move forward tail of MatVecs queue
		mat_vec_data_at_current_offset = &mat_vec_data_at_current_offset[total_new_mat_vecs];

		// data on current level is no longer needed -> cleanup
		invalidate_array<<<grid_size_for_children, block_size>>>(current_level_data, total_children);
		checkCUDAError("invalidate_array2");

		// switch to next level by flipping pointers to array
		struct work_item* tmp_level_data_pointer;
		tmp_level_data_pointer = next_level_data;
		next_level_data = current_level_data;
		current_level_data = tmp_level_data_pointer;
		total_children = total_new_children;

		// compute new compute configuration for children computations
		grid_size_for_children = (total_children + (block_size - 1)) / block_size;

		if (total_children==0) // stopping when no more children are generated
			break;
	}
}

void traverse_with_dynamic_arrays(struct work_item root_h, struct work_item* mat_vec_data, int* mat_vec_data_count, struct morton_code* input_set1_codes, struct morton_code* input_set2_codes, struct point_set* input_set1, struct point_set* input_set2, double eta, int max_level, int c_leaf, int max_elements_in_array)
{
	struct work_item* current_level_data = 0;
	struct work_item* next_level_data = 0;

//	cudaMalloc((void**)&current_level_data, max_elements_in_array*sizeof(struct work_item));
//	cudaMalloc((void**)&next_level_data, max_elements_in_array*sizeof(struct work_item));

	int* new_mat_vec_counts;   // number of MatVecs that will be generated per valid node of the current tree level
	int* new_child_counts;     // number of child nodes that will be generated per valid node of the current tree level
	int* new_child_offsets;    // storage for the offsets for the new child nodes in the next tree level

	int total_new_mat_vecs;  // temp field to store total number of new MatVecs of the current level

	int total_children = 1;  // number of nodes on current level
	int total_new_children;  // temp field to store total number of new nodes on next level
	int block_size = 512;
	int grid_size_for_children = (total_children + (block_size - 1)) / block_size;  // field to store the grid size for
																					// kernels that follow the node count

	struct work_item* mat_vec_data_at_current_offset = mat_vec_data;	// array to store the current tail of the queue / array that holds the MatVecs

	// allocate array for current level
	cudaMalloc((void**)&current_level_data, total_children*sizeof(struct work_item));
	checkCUDAError("cudaMalloc0");
	invalidate_array<<<(total_children + (block_size - 1)) / block_size, block_size>>>(current_level_data, total_children);
	checkCUDAError("invalidate_array0");

	// fill initial node into first tree level
	init_tree_array_root<<<1, 1>>>(current_level_data, root_h);
	checkCUDAError("init_tree_array_root");

	for (int current_level=0; current_level<max_level; current_level++)  // run over all arrays
	{
		// allocation
		cudaMalloc((void**)&new_mat_vec_counts, total_children*sizeof(int));
		checkCUDAError("cudaMalloc01");
		cudaMalloc((void**)&new_child_counts, total_children*sizeof(int));
		checkCUDAError("cudaMalloc02");
		cudaMalloc((void**)&new_child_offsets, total_children*sizeof(int));
		checkCUDAError("cudaMalloc03");
		// pointer fun
		thrust::device_ptr<int> new_mat_vec_counts_ptr(new_mat_vec_counts);
		thrust::device_ptr<int> new_child_counts_ptr(new_child_counts);
		thrust::device_ptr<int> new_child_offsets_ptr(new_child_offsets);

		set_array<<<grid_size_for_children, block_size>>>(new_child_counts, 0, total_children);  //  will compute new child counts for total_children nodes
		checkCUDAError("set_array");

		// find number of children & MatVecs per node on current level
		count_for_new_level<<<grid_size_for_children, block_size>>>(current_level_data, next_level_data, new_mat_vec_counts, new_child_counts, total_children, input_set1_codes, input_set2_codes, input_set1, input_set2, eta, current_level, max_level, c_leaf);
		checkCUDAError("count_for_new_level");

		// compute total number of new children & MatVecs
		total_new_mat_vecs = thrust::reduce(new_mat_vec_counts_ptr, new_mat_vec_counts_ptr+total_children);
		total_new_children = thrust::reduce(new_child_counts_ptr, new_child_counts_ptr+total_children);

		*mat_vec_data_count = *mat_vec_data_count + total_new_mat_vecs;

		// compute node offsets in new level & offsets for MatVecs
		thrust::exclusive_scan(new_mat_vec_counts_ptr, new_mat_vec_counts_ptr+total_children, new_mat_vec_counts_ptr);   // here, I reuse the field to store the offsets (for memory efficiency reasons)
		thrust::exclusive_scan(new_child_counts_ptr, new_child_counts_ptr+total_children, new_child_offsets_ptr);  // here, I store the offsets in a dedicated field

		// allocate array for next level
		cudaMalloc((void**)&next_level_data, total_new_children*sizeof(struct work_item));
		checkCUDAError("cudaMalloc1");
		if (total_new_children > 0)  // handle case in which no new level is generated, does not work with 0 grid size
		{
			invalidate_array<<<(total_new_children + (block_size - 1)) / block_size, block_size>>>(next_level_data, total_new_children);
			checkCUDAError("invalidate_array1");
		}

		// generate new level with nodes and write new MatVecs into queue
		generate_new_level<<<grid_size_for_children, block_size>>>(current_level_data, next_level_data, mat_vec_data_at_current_offset, new_child_counts, total_children, new_mat_vec_counts, new_child_offsets, input_set1_codes, input_set2_codes, input_set1, input_set2, eta, current_level, max_level, c_leaf);
		checkCUDAError("generate_new_level");

		// move forward tail of MatVecs queue
		mat_vec_data_at_current_offset = &mat_vec_data_at_current_offset[total_new_mat_vecs];

		// data on current level is no longer needed -> cleanup
		cudaFree(current_level_data);
		cudaFree(new_mat_vec_counts);
		cudaFree(new_child_counts);
		cudaFree(new_child_offsets);

		// switch to next level
		current_level_data = next_level_data;
		next_level_data = 0;
		total_children = total_new_children;

		// compute new compute configuration for children computations
		grid_size_for_children = (total_children + (block_size - 1)) / block_size;

		if (total_children==0) // stopping when no more children are generated
		{
			cudaFree(current_level_data);
			break;
		}
	}
}

struct min_or_m2_in_second_argument : public thrust::binary_function<int, int, int>
{
	__host__ __device__ int operator()(int a, int b)
	{
		return (b!=-2) ? max(a,b) : -2;
	}
};

struct equals_zero : public thrust::unary_function<int, int>
{
	__host__ __device__ int operator()(int x) { return x==0; }
};

void compute_bounding_boxes_fun_old(struct work_item* current_level_data, int total_children, struct point_set* input_set1, struct point_set* input_set2)
{
	int block_size = 512;
	compute_bounding_boxes<<<(total_children + (block_size - 1)) / block_size, block_size>>>(current_level_data, total_children, input_set1, input_set2);
	checkCUDAError("compute_bounding_boxes");
}

__global__ void set_bounds_for_keys(int* keys, struct work_item* current_level_data, int total_children, int set_num)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= total_children)
		return;

	struct work_item* work = &current_level_data[idx];

	int l, u;

	if (set_num==1)
	{
		l = work->set1_l;
		u = work->set1_u;
	}
	else
	{
		l = work->set2_l;
		u = work->set2_u;
	}

	keys[l] = idx;
	keys[u] = -2;
}

__global__ void set_bounds_for_keys_using_limits(int* keys, int* l, int* u, int total_children, int set_num)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= total_children)
		return;

	keys[l[idx]] = (idx+1);
	keys[u[idx]] = -(idx+1);
}

__global__ void correct_upper_bound(int* keys, struct work_item* current_level_data, int total_children, int set_num)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= total_children)
		return;

	struct work_item* work = &current_level_data[idx];

	int u;

	if (set_num==1)
		u = work->set1_u;
	else
		u = work->set2_u;

	keys[u] = idx;

}

__global__ void correct_upper_bound_using_limits(int* keys, int* l, int* u, int total_children, int set_num)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= total_children)
		return;

	keys[u[idx]] = (idx+1);

}

__global__ void set_bounding_box_minmax(double* maxs, int* output_keys, int d, int dim, int maxs_size, struct work_item* current_level_data, int type){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= maxs_size)
		return;

	int output_pos = output_keys[idx];

	if (output_pos==-1)
	{
		return;
	}

	struct work_item* work = &current_level_data[output_pos];

	switch (type)
	{
		case 1: work->max1[d] = maxs[idx]; break;
		case 2: work->max2[d] = maxs[idx]; break;
		case 3: work->min1[d] = maxs[idx]; break;
		case 4: work->min2[d] = maxs[idx]; break;
	}
	work->dim = dim;
}

__global__ void get_point_count_dim(int* point_count, int* dim, struct point_set* input_set1)
{
	*point_count = input_set1->size;
	*dim = input_set1->dim;
}

__global__ void get_coords_pointer(double** coords, int dim, struct point_set* input_set)
{
	coords[0] = input_set->coords[dim];
}

void print_int(int* array, int n)
{
	int* array_h = new int[n];
	cudaMemcpy(array_h, array, n*sizeof(int), cudaMemcpyDeviceToHost);
	for (int i=0; i<n; i++)
		printf("% d ", array_h[i]);
	printf("\n");
	delete[] array_h;
}

void print_double(double* array, int n)
{
	double* array_h = new double[n];
	cudaMemcpy(array_h, array, n*sizeof(double), cudaMemcpyDeviceToHost);
	for (int i=0; i<n; i++)
		printf("% .5ef ", array_h[i]);
	printf("\n");
	delete[] array_h;
}

void print_bool(bool* array, int n)
{
	bool* array_h = new bool[n];
	cudaMemcpy(array_h, array, n*sizeof(bool), cudaMemcpyDeviceToHost);
	for (int i=0; i<n; i++)
		printf("%d ", array_h[i]);
	printf("\n");
	delete[] array_h;
}



__global__ void get_work_item_point_set_limits(int* l, int* u, struct work_item* current_level_data, int total_children, int point_set_nr)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= total_children)
		return;

	if (point_set_nr==1)
	{
		l[idx] = current_level_data[idx].set1_l;
		u[idx] = current_level_data[idx].set1_u;
	}
	else
	{
		l[idx] = current_level_data[idx].set2_l;
		u[idx] = current_level_data[idx].set2_u;
	}

}

__global__ void initialize_lookup_map(int* map, int* l, int total_children)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= total_children)
		return;

	if (idx==0)
	{
		map[idx] = 0;
	}
	else if (l[idx]!=l[idx-1])
	{
		map[idx] = 1;
	}
	else
		map[idx] = 0;
}

__global__ void apply_permutation_to_map(int* map, int* tmp_map, int* permutation, int total_children)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= total_children)
		return;

	map[permutation[idx]] = tmp_map[idx];
}

void compute_map_for_lookup_table(int* map, struct work_item* current_level_data, int total_children, struct point_set* input_set1, struct point_set* input_set2, int set_nr)
{
	int block_size = 512;

	thrust::device_ptr<int> map_ptr(map);

	int* permutation;
	cudaMalloc((void**)&permutation, total_children*sizeof(int));
	thrust::device_ptr<int> permutation_ptr(permutation);
	thrust::sequence(permutation_ptr, permutation_ptr+total_children);

	int* l;
	int* u;
	cudaMalloc((void**)&l, total_children*sizeof(int));
	cudaMalloc((void**)&u, total_children*sizeof(int));
	thrust::device_ptr<int> l_ptr(l);
	thrust::device_ptr<int> u_ptr(u);

	get_work_item_point_set_limits<<<(total_children + (block_size - 1)) / block_size, block_size>>>(l, u, current_level_data, total_children, set_nr);
	cudaThreadSynchronize();
	checkCUDAError("get_work_item_point_set_limits");

	cudaFree(u);  // TODO u is not really used; remove?

	thrust::sort_by_key(l_ptr, l_ptr+total_children, permutation_ptr);

	int* tmp_map;
	cudaMalloc((void**)&tmp_map, total_children*sizeof(int));
	thrust::device_ptr<int> tmp_map_ptr(tmp_map);

	initialize_lookup_map<<<(total_children + (block_size - 1)) / block_size, block_size>>>(tmp_map, l, total_children);
	cudaThreadSynchronize();
	checkCUDAError("initialize_lookup_map");

	cudaFree(l);

	thrust::inclusive_scan(tmp_map_ptr, tmp_map_ptr+total_children, tmp_map_ptr);

	apply_permutation_to_map<<<(total_children + (block_size - 1)) / block_size, block_size>>>(map, tmp_map, permutation, total_children);
	cudaThreadSynchronize();
	checkCUDAError("apply_permutation_to_map");

	cudaFree(tmp_map);
	cudaFree(permutation);
}

void compute_lookup_table(double*** lookup_table_min, double*** lookup_table_max, int* lookup_table_size, struct work_item* current_level_data, int total_children, struct point_set* input_set1, struct point_set* input_set2, int set_nr)
{
	int block_size = 512;

	int* l;
	int* u;
	cudaMalloc((void**)&l, total_children*sizeof(int));
	cudaMalloc((void**)&u, total_children*sizeof(int));
	thrust::device_ptr<int> l_ptr(l);
	thrust::device_ptr<int> u_ptr(u);

	get_work_item_point_set_limits<<<(total_children + (block_size - 1)) / block_size, block_size>>>(l, u, current_level_data, total_children, set_nr);
	cudaThreadSynchronize();
	checkCUDAError("get_work_item_point_set_limits");

	thrust::sort(l_ptr, l_ptr+total_children);
	thrust::sort(u_ptr, u_ptr+total_children);

	thrust::device_ptr<int> new_end_unique = thrust::unique(l_ptr, l_ptr+total_children);
	thrust::unique(u_ptr, u_ptr+total_children);

	*lookup_table_size = new_end_unique - l_ptr;



	// very dirty way to get point count and dimensionality of points
	int point_count,dim;
	int* point_count_d; cudaMalloc((void**)&point_count_d, sizeof(int));
	int* dim_d; cudaMalloc((void**)&dim_d, sizeof(int));
	get_point_count_dim<<<1,1>>>(point_count_d, dim_d, input_set1);
	cudaMemcpy(&point_count, point_count_d, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&dim, dim_d, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(point_count_d); cudaFree(dim_d);

	double** lookup_table_min_h = new double*[dim];
	double** lookup_table_max_h = new double*[dim];
	cudaMalloc((void**)lookup_table_min, dim*sizeof(double*));
	cudaMalloc((void**)lookup_table_max, dim*sizeof(double));
	for (int d=0; d<dim; d++)
		cudaMalloc((void**)&(lookup_table_min_h[d]), lookup_table_size[0]*sizeof(double));
	for (int d=0; d<dim; d++)
		cudaMalloc((void**)&(lookup_table_max_h[d]), lookup_table_size[0]*sizeof(double));
	cudaMemcpy(*lookup_table_min, lookup_table_min_h, dim*sizeof(double*), cudaMemcpyHostToDevice);
	cudaMemcpy(*lookup_table_max, lookup_table_max_h, dim*sizeof(double*), cudaMemcpyHostToDevice);

//	printf("lookup_table_size: %d\n", lookup_table_size[0]);


	double** coords_pointer;
	double* coords_pointer_h;
	cudaMalloc((void**)&coords_pointer, sizeof(double*));

	int* keys;
	cudaMalloc((void**)&keys,point_count*sizeof(int));
	thrust::device_ptr<int> keys_ptr(keys);

	int* output_keys;
	cudaMalloc((void**)&output_keys,point_count*sizeof(int)); // point_count is a bad upper bound
	thrust::device_ptr<int> output_keys_ptr(output_keys);

	struct min_or_m2_in_second_argument op;

	thrust::pair<thrust::device_ptr<int>, thrust::device_ptr<double> > new_end;

//	print_work_items(current_level_data, total_children);

	set_array<<<(point_count + (block_size - 1)) / block_size, block_size>>>(keys, 0, point_count);
	cudaThreadSynchronize();
	checkCUDAError("set_array");

//	printf("keys after set array:\n");
//	print_int(keys,point_count);

//	printf("l,u:\n");
//	print_int(l,*lookup_table_size);
//	print_int(u,*lookup_table_size);


	set_bounds_for_keys_using_limits<<<(*lookup_table_size + (block_size - 1)) / block_size, block_size>>>(keys, l, u, *lookup_table_size, set_nr);
	cudaThreadSynchronize();
	checkCUDAError("set_bounds_for_keys");

//	printf("keys after set bounds for keys:\n");
//	print_int(keys,point_count);


	thrust::inclusive_scan(keys_ptr, keys_ptr+point_count, keys_ptr);


//	printf("keys after inclusive scan:\n");
//	print_int(keys,point_count);

	correct_upper_bound_using_limits<<<(*lookup_table_size + (block_size - 1)) / block_size, block_size>>>(keys, l, u, *lookup_table_size, set_nr);
	cudaThreadSynchronize();
	checkCUDAError("correct_upper_bound");

//	printf("keys after correct upper bound:\n");
//	print_int(keys,point_count);

//	printf("point_count %d\n", point_count);
	double* tmp_lookup_table;
	cudaMalloc((void**)&tmp_lookup_table, point_count*sizeof(double));
	thrust::device_ptr<double> tmp_lookup_table_ptr(tmp_lookup_table);

//	printf("Computing maximum in set %d\n", set_nr);
	for (int d=0; d<dim; d++)
	{
		get_coords_pointer<<<1,1>>>(coords_pointer, d, input_set2);

		cudaMemcpy(&coords_pointer_h, coords_pointer, sizeof(double*), cudaMemcpyDeviceToHost);
		checkCUDAError("cudaMemcpy11");
		thrust::device_ptr<double> coords_current_dim_ptr(coords_pointer_h);

//		printf("keys, coords_current_dim\n");
//		print_int(keys, point_count);
//		print_double(coords_pointer_h, point_count);

		thrust::device_ptr<double> lookup_table_max_current_dim_ptr(lookup_table_max_h[d]);

		new_end = thrust::reduce_by_key(keys_ptr, keys_ptr+point_count, coords_current_dim_ptr, output_keys_ptr, tmp_lookup_table_ptr, thrust::equal_to<int>(), thrust::maximum<double>());

		int output_size = new_end.first - output_keys_ptr;
//		printf("first_output_size %d\n", output_size);

		// remove empty entries in lookup table
		thrust::device_ptr<double> new_end_without_empty_entries;
		new_end_without_empty_entries = thrust::remove_if(tmp_lookup_table_ptr, tmp_lookup_table_ptr+output_size, output_keys_ptr, equals_zero());
		checkCUDAError("remove_if");

		output_size = new_end_without_empty_entries - tmp_lookup_table_ptr;

		// !!!!!!!!!!!!!!!
		// ALLOCATION FOR LOOKUP_TABLE is TOOOO SMALLL (empty entries are neglected)
		// !!!!!!!!!!!!!!!

//		printf("output_size %d\n", output_size);
		cudaMemcpy(lookup_table_max_h[d], tmp_lookup_table, output_size*sizeof(double), cudaMemcpyHostToDevice);
		checkCUDAError("cudaMemcpy12");

//		printf("lookup table max for d=%d\n", d);
//		print_double(lookup_table_max_h[d], output_size);

//		set_bounding_box_minmax<<<(output_size + (block_size - 1)) / block_size, block_size>>>(minmaxs, output_keys, d, dim, output_size, current_level_data, 1);
//		cudaThreadSynchronize();
//		checkCUDAError("set_bounding_box_minmax");
	}

//	printf("Computing minimum in set %d\n", set_nr);
	for (int d=0; d<dim; d++)
	{
		get_coords_pointer<<<1,1>>>(coords_pointer, d, input_set2);

		cudaMemcpy(&coords_pointer_h, coords_pointer, sizeof(double*), cudaMemcpyDeviceToHost);
		checkCUDAError("cudaMemcpy21");
		thrust::device_ptr<double> coords_current_dim_ptr(coords_pointer_h);

//		printf("keys, coords_current_dim\n");
//		print_int(keys, point_count);
//		print_double(coords_pointer_h, point_count);

		thrust::device_ptr<double> lookup_table_min_current_dim_ptr(lookup_table_min_h[d]);

		new_end = thrust::reduce_by_key(keys_ptr, keys_ptr+point_count, coords_current_dim_ptr, output_keys_ptr, tmp_lookup_table_ptr, thrust::equal_to<int>(), thrust::minimum<double>());

		int output_size = new_end.first - output_keys_ptr;

		// remove empty entries in lookup table
		thrust::device_ptr<double> new_end_without_empty_entries;
		new_end_without_empty_entries = thrust::remove_if(tmp_lookup_table_ptr, tmp_lookup_table_ptr+output_size, output_keys_ptr, equals_zero());

		output_size = new_end_without_empty_entries - tmp_lookup_table_ptr;

		cudaMemcpy(lookup_table_min_h[d], tmp_lookup_table, output_size*sizeof(double), cudaMemcpyHostToDevice);
		checkCUDAError("cudaMemcpy22");

//		printf("lookup table min for d=%d\n", d);
//		print_double(lookup_table_min_h[d], output_size);

//		printf("min for d=%d\n", d);
//		print_double(minmaxs, output_size);

//		set_bounding_box_minmax<<<(output_size + (block_size - 1)) / block_size, block_size>>>(minmaxs, output_keys, d, dim, output_size, current_level_data, 3);
//		cudaThreadSynchronize();
//		checkCUDAError("set_bounding_box_minmax");

	}

	cudaFree(tmp_lookup_table);
}

__global__ void set_bounding_box_minmax_using_lookup_table(double** lookup_table_min, double** lookup_table_max, int* map, int dim, struct work_item* current_level_data, int total_children, int type)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= total_children)
		return;

	int map_index = map[idx];

	struct work_item* work = &current_level_data[idx];

	if (type==1)
	{
		for (int d=0; d<dim; d++)
		{
			work->max1[d] = lookup_table_max[d][map_index];
			work->min1[d] = lookup_table_min[d][map_index];
		}
	}
	else
	{
		for (int d=0; d<dim; d++)
		{
			work->max2[d] = lookup_table_max[d][map_index];
			work->min2[d] = lookup_table_min[d][map_index];
		}
	}

	work->dim = dim;
}


void compute_bounding_boxes_fun(struct work_item* current_level_data, int total_children, struct point_set* input_set1, struct point_set* input_set2)
{
	// very dirty way to get point count and dimensionality of points
	int point_count,dim;
	int* point_count_d; cudaMalloc((void**)&point_count_d, sizeof(int));
	int* dim_d; cudaMalloc((void**)&dim_d, sizeof(int));
	get_point_count_dim<<<1,1>>>(point_count_d, dim_d, input_set1);
	cudaMemcpy(&point_count, point_count_d, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&dim, dim_d, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(point_count_d); cudaFree(dim_d);

	int* map;
	cudaMalloc((void**)&map, total_children*sizeof(double));

	double** lookup_table_min;
	double** lookup_table_max;
	int lookup_table_size;

//	printf("WORKING ON SET 1...\n");

	compute_map_for_lookup_table(map, current_level_data, total_children, input_set1, input_set2, 1);

	compute_lookup_table(&lookup_table_min, &lookup_table_max, &lookup_table_size, current_level_data, total_children, input_set1, input_set2, 1);


	int block_size = 512;
//	printf("bla %d %d\n",(total_children + (block_size - 1)) / block_size, block_size);
	set_bounding_box_minmax_using_lookup_table<<<(total_children + (block_size - 1)) / block_size, block_size>>>(lookup_table_min, lookup_table_max, map, dim, current_level_data, total_children, 1);
	cudaThreadSynchronize();
	checkCUDAError("set_bounding_box_minmax_using_lookup_table");

	double** tmp_array = new double*[dim];
	cudaMemcpy(tmp_array, lookup_table_min, dim*sizeof(double*), cudaMemcpyDeviceToHost);
	checkCUDAError("cudaMemcpy1");
	for (int d=0; d<dim; d++)
	{
		cudaFree(tmp_array[d]);
		checkCUDAError("cudaFree1");
	}
	cudaMemcpy(tmp_array, lookup_table_max, dim*sizeof(double*), cudaMemcpyDeviceToHost);
	checkCUDAError("cudaMemcpy2");
	for (int d=0; d<dim; d++)
	{
		cudaFree(tmp_array[d]);
		checkCUDAError("cudaFree2");
	}

//	printf("WORKING ON SET 2...\n");

	compute_map_for_lookup_table(map, current_level_data, total_children, input_set1, input_set2, 2);

	compute_lookup_table(&lookup_table_min, &lookup_table_max, &lookup_table_size, current_level_data, total_children, input_set1, input_set2, 2);


//	printf("bla %d %d\n",(total_children + (block_size - 1)) / block_size, block_size);
	set_bounding_box_minmax_using_lookup_table<<<(total_children + (block_size - 1)) / block_size, block_size>>>(lookup_table_min, lookup_table_max, map, dim, current_level_data, total_children, 2);
	cudaThreadSynchronize();
	checkCUDAError("set_bounding_box_minmax_using_lookup_table");




	cudaMemcpy(tmp_array, lookup_table_min, dim*sizeof(double*), cudaMemcpyDeviceToHost);
	checkCUDAError("cudaMemcpy1");
	for (int d=0; d<dim; d++)
	{
		cudaFree(tmp_array[d]);
		checkCUDAError("cudaFree1");
	}
	cudaMemcpy(tmp_array, lookup_table_max, dim*sizeof(double*), cudaMemcpyDeviceToHost);
	checkCUDAError("cudaMemcpy2");
	for (int d=0; d<dim; d++)
	{
		cudaFree(tmp_array[d]);
		checkCUDAError("cudaFree2");
	}

	delete[] tmp_array;

}

/*

void compute_bounding_boxes_fun(struct work_item* current_level_data, int total_children, struct point_set* input_set1, struct point_set* input_set2)
{
	int block_size = 512;

	// very dirty way to get point count and dimensionality of points
	int point_count,dim;
	int* point_count_d; cudaMalloc((void**)&point_count_d, sizeof(int));
	int* dim_d; cudaMalloc((void**)&dim_d, sizeof(int));
	get_point_count_dim<<<1,1>>>(point_count_d, dim_d, input_set1);
	cudaMemcpy(&point_count, point_count_d, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&dim, dim_d, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(point_count_d); cudaFree(dim_d);

	double** coords_pointer;
	double* coords_pointer_h;
	cudaMalloc((void**)&coords_pointer, sizeof(double*));

	int* keys;
	cudaMalloc((void**)&keys,point_count*sizeof(int));
	thrust::device_ptr<int> keys_ptr(keys);

	double* minmaxs;
	int* output_keys;
	cudaMalloc((void**)&minmaxs,point_count*sizeof(double)); // point_count is a bad upper bound
	cudaMalloc((void**)&output_keys,point_count*sizeof(int)); // point_count is a bad upper bound

	thrust::device_ptr<double> minmaxs_ptr(minmaxs);
	thrust::device_ptr<int> output_keys_ptr(output_keys);

	struct min_or_m2_in_second_argument op;

	thrust::pair<thrust::device_ptr<int>, thrust::device_ptr<double> > new_end;

	print_work_items(current_level_data, total_children);

	set_array<<<(point_count + (block_size - 1)) / block_size, block_size>>>(keys, -1, point_count);
	cudaThreadSynchronize();
	checkCUDAError("set_array");

	printf("keys after set array:\n");
	print_int(keys,point_count);

	set_bounds_for_keys<<<(total_children + (block_size - 1)) / block_size, block_size>>>(keys, current_level_data, total_children, 1);
	cudaThreadSynchronize();
	checkCUDAError("set_bounds_for_keys");

	printf("keys after set bounds for keys:\n");
	print_int(keys,point_count);

	thrust::inclusive_scan(keys_ptr, keys_ptr+point_count, keys_ptr, op);

	printf("keys after inclusive scan:\n");
	print_int(keys,point_count);

	correct_upper_bound<<<(total_children + (block_size - 1)) / block_size, block_size>>>(keys, current_level_data, total_children, 1);
	cudaThreadSynchronize();
	checkCUDAError("correct_upper_bound");

	printf("keys after correct upper bound:\n");
	print_int(keys,point_count);

	for (int d=0; d<dim; d++)
	{
		get_coords_pointer<<<1,1>>>(coords_pointer, d, input_set1);

		cudaMemcpy(&coords_pointer_h, coords_pointer, sizeof(double*), cudaMemcpyDeviceToHost);
		thrust::device_ptr<double> coords_current_dim_ptr(coords_pointer_h);

		printf("keys, coords_current_dim\n");
		print_int(keys, point_count);
		print_double(coords_pointer_h, point_count);

		new_end = thrust::reduce_by_key(keys_ptr, keys_ptr+point_count, coords_current_dim_ptr, output_keys_ptr, minmaxs_ptr, thrust::equal_to<int>(), thrust::maximum<double>());

		int output_size = new_end.first - output_keys_ptr;

		printf("max for d=%d\n", d);
		print_double(minmaxs, output_size);


		set_bounding_box_minmax<<<(output_size + (block_size - 1)) / block_size, block_size>>>(minmaxs, output_keys, d, dim, output_size, current_level_data, 1);
		cudaThreadSynchronize();
		checkCUDAError("set_bounding_box_minmax");
	}

	for (int d=0; d<dim; d++)
	{
		get_coords_pointer<<<1,1>>>(coords_pointer, d, input_set1);

		cudaMemcpy(&coords_pointer_h, coords_pointer, sizeof(double*), cudaMemcpyDeviceToHost);
		thrust::device_ptr<double> coords_current_dim_ptr(coords_pointer_h);

//		printf("keys, coords_current_dim\n");
//		print_int(keys, point_count);
//		print_double(coords_pointer_h, point_count);

		new_end = thrust::reduce_by_key(keys_ptr, keys_ptr+point_count, coords_current_dim_ptr, output_keys_ptr, minmaxs_ptr, thrust::equal_to<int>(), thrust::minimum<double>());

		int output_size = new_end.first - output_keys_ptr;

//		printf("min for d=%d\n", d);
//		print_double(minmaxs, output_size);

		set_bounding_box_minmax<<<(output_size + (block_size - 1)) / block_size, block_size>>>(minmaxs, output_keys, d, dim, output_size, current_level_data, 3);
		cudaThreadSynchronize();
		checkCUDAError("set_bounding_box_minmax");

	}

	set_array<<<(point_count + (block_size - 1)) / block_size, block_size>>>(keys, -1, point_count);
	cudaThreadSynchronize();
	checkCUDAError("set_array");

	set_bounds_for_keys<<<(total_children + (block_size - 1)) / block_size, block_size>>>(keys, current_level_data, total_children, 2);
	cudaThreadSynchronize();
	checkCUDAError("set_bounds_for_keys");

	thrust::inclusive_scan(keys_ptr, keys_ptr+point_count, keys_ptr, op);

	correct_upper_bound<<<(total_children + (block_size - 1)) / block_size, block_size>>>(keys, current_level_data, total_children, 2);
	cudaThreadSynchronize();
	checkCUDAError("correct_upper_bound");

	for (int d=0; d<dim; d++)
	{
		get_coords_pointer<<<1,1>>>(coords_pointer, d, input_set2);
		cudaMemcpy(&coords_pointer_h, coords_pointer, sizeof(double*), cudaMemcpyDeviceToHost);
		thrust::device_ptr<double> coords_current_dim_ptr(coords_pointer_h);

//		printf("keys, coords_current_dim\n");
//		print_int(keys, point_count);
//		print_double(coords_pointer_h, point_count);

		new_end = thrust::reduce_by_key(keys_ptr, keys_ptr+point_count, coords_current_dim_ptr, output_keys_ptr, minmaxs_ptr, thrust::equal_to<int>(), thrust::maximum<double>());

		int output_size = new_end.first - output_keys_ptr;

//		printf("max for d=%d\n", d);
//		print_double(minmaxs, output_size);

		set_bounding_box_minmax<<<(output_size + (block_size - 1)) / block_size, block_size>>>(minmaxs, output_keys, d, dim, output_size, current_level_data, 2);
		cudaThreadSynchronize();
		checkCUDAError("set_bounding_box_minmax");
	}

	for (int d=0; d<dim; d++)
	{
		get_coords_pointer<<<1,1>>>(coords_pointer, d, input_set2);
		cudaMemcpy(&coords_pointer_h, coords_pointer, sizeof(double*), cudaMemcpyDeviceToHost);
		thrust::device_ptr<double> coords_current_dim_ptr(coords_pointer_h);

//		printf("keys, coords_current_dim\n");
//		print_int(keys, point_count);
//		print_double(coords_pointer_h, point_count);

		new_end = thrust::reduce_by_key(keys_ptr, keys_ptr+point_count, coords_current_dim_ptr, output_keys_ptr, minmaxs_ptr, thrust::equal_to<int>(), thrust::minimum<double>());

		int output_size = new_end.first - output_keys_ptr;

//		printf("min for d=%d\n", d);
//		print_double(minmaxs, output_size);

		set_bounding_box_minmax<<<(output_size + (block_size - 1)) / block_size, block_size>>>(minmaxs, output_keys, d, dim, output_size, current_level_data, 4);
		cudaThreadSynchronize();
		checkCUDAError("set_bounding_box_minmax");
	}
	cudaThreadSynchronize();

	cudaFree(keys);
	checkCUDAError("cudaFree");
	cudaFree(minmaxs);
	checkCUDAError("cudaFree");
	cudaFree(output_keys);
	checkCUDAError("cudaFree");
	cudaFree(coords_pointer);
	checkCUDAError("cudaFree");
}

*/

void traverse_with_dynamic_arrays_dynamic_output(struct work_item root_h, struct work_item** mat_vec_data, int* mat_vec_data_count, int* mat_vec_data_array_size, struct morton_code* input_set1_codes, struct morton_code* input_set2_codes, struct point_set* input_set1, struct point_set* input_set2, double eta, int max_level, int c_leaf, int max_elements_in_array)
{
	struct work_item* current_level_data = 0;
	struct work_item* next_level_data = 0;

//	cudaMalloc((void**)&current_level_data, max_elements_in_array*sizeof(struct work_item));
//	cudaMalloc((void**)&next_level_data, max_elements_in_array*sizeof(struct work_item));

	int* new_mat_vec_counts;   // number of MatVecs that will be generated per valid node of the current tree level
	int* new_child_counts;     // number of child nodes that will be generated per valid node of the current tree level
	int* new_child_offsets;    // storage for the offsets for the new child nodes in the next tree level

	int total_new_mat_vecs;  // temp field to store total number of new MatVecs of the current level

	int total_children = 1;  // number of nodes on current level
	int total_new_children;  // temp field to store total number of new nodes on next level
	int block_size = 512;
	int grid_size_for_children = (total_children + (block_size - 1)) / block_size;  // field to store the grid size for
																					// kernels that follow the node count

	struct work_item* mat_vec_data_at_current_offset = *mat_vec_data;	// array to store the current tail of the queue / array that holds the MatVecs

	// allocate array for current level
	cudaMalloc((void**)&current_level_data, total_children*sizeof(struct work_item));
	checkCUDAError("cudaMalloc0");
	invalidate_array<<<(total_children + (block_size - 1)) / block_size, block_size>>>(current_level_data, total_children);
	cudaThreadSynchronize();
	checkCUDAError("invalidate_array0");

	// fill initial node into first tree level
	init_tree_array_root<<<1, 1>>>(current_level_data, root_h);
	cudaThreadSynchronize();
	checkCUDAError("init_tree_array_root");

	for (int current_level=0; current_level<max_level; current_level++)  // run over all arrays
	{
		TIME_sstart;
//		compute_bounding_boxes_fun_old(current_level_data, total_children, input_set1, input_set2);
		compute_bounding_boxes_fun(current_level_data, total_children, input_set1, input_set2);
		TIME_sstop("compute_bounding_boxes");

		cudaThreadSynchronize();
		checkCUDAError("cudaThreadSynchronize");

		// allocation
		cudaMalloc((void**)&new_mat_vec_counts, total_children*sizeof(int));
		checkCUDAError("cudaMalloc01");
		cudaMalloc((void**)&new_child_counts, total_children*sizeof(int));
		checkCUDAError("cudaMalloc02");
		cudaMalloc((void**)&new_child_offsets, total_children*sizeof(int));
		checkCUDAError("cudaMalloc03");
		// pointer fun
		thrust::device_ptr<int> new_mat_vec_counts_ptr(new_mat_vec_counts);
		thrust::device_ptr<int> new_child_counts_ptr(new_child_counts);
		thrust::device_ptr<int> new_child_offsets_ptr(new_child_offsets);

		set_array<<<grid_size_for_children, block_size>>>(new_child_counts, 0, total_children);  //  will compute new child counts for total_children nodes
		cudaThreadSynchronize();
		checkCUDAError("set_array");

		// find number of children & MatVecs per node on current level
		count_for_new_level<<<grid_size_for_children, block_size>>>(current_level_data, next_level_data, new_mat_vec_counts, new_child_counts, total_children, input_set1_codes, input_set2_codes, input_set1, input_set2, eta, current_level, max_level, c_leaf);
		cudaThreadSynchronize();
		checkCUDAError("count_for_new_level");

		// compute total number of new children & MatVecs
		total_new_mat_vecs = thrust::reduce(new_mat_vec_counts_ptr, new_mat_vec_counts_ptr+total_children);
		total_new_children = thrust::reduce(new_child_counts_ptr, new_child_counts_ptr+total_children);

		// compute node offsets in new level & offsets for MatVecs
		thrust::exclusive_scan(new_mat_vec_counts_ptr, new_mat_vec_counts_ptr+total_children, new_mat_vec_counts_ptr);   // here, I reuse the field to store the offsets (for memory efficiency reasons)
		thrust::exclusive_scan(new_child_counts_ptr, new_child_counts_ptr+total_children, new_child_offsets_ptr);  // here, I store the offsets in a dedicated field

		// dynamically increase the size of the MatVec data array, if necessary
		// WARNING: This is still not the best possible implementation since (starting from a specific size) it always requires to reallocate memory
		if ((*mat_vec_data_count + total_new_mat_vecs)> *mat_vec_data_array_size)
		{
			struct work_item* new_array;  // pointer for new array
			struct work_item* old_array = *mat_vec_data;  // save pointer to old array
			cudaMalloc((void**) &new_array, (*mat_vec_data_count + total_new_mat_vecs)*sizeof(struct work_item));  // allocate new, larger array
			cudaMemcpy(new_array, old_array, (*mat_vec_data_count)*sizeof(struct work_item), cudaMemcpyDeviceToDevice);  // transfer data to new array
			*mat_vec_data = new_array;  // set new field as standard mat_vec_data array
			*mat_vec_data_array_size = *mat_vec_data_count + total_new_mat_vecs;  // store size of new array
			cudaFree(old_array);  // delete old array
			mat_vec_data_at_current_offset = &new_array[*mat_vec_data_count];  // re-set tail of output queue
		}

		// allocate array for next level
		cudaMalloc((void**)&next_level_data, total_new_children*sizeof(struct work_item));
		checkCUDAError("cudaMalloc1");
		if (total_new_children > 0)  // handle case in which no new level is generated, does not work with 0 grid size
		{
			invalidate_array<<<(total_new_children + (block_size - 1)) / block_size, block_size>>>(next_level_data, total_new_children);
			cudaThreadSynchronize();
			checkCUDAError("invalidate_array1");
		}


		// generate new level with nodes and write new MatVecs into queue
		generate_new_level<<<grid_size_for_children, block_size>>>(current_level_data, next_level_data, mat_vec_data_at_current_offset, new_child_counts, total_children, new_mat_vec_counts, new_child_offsets, input_set1_codes, input_set2_codes, input_set1, input_set2, eta, current_level, max_level, c_leaf);
		cudaThreadSynchronize();
		checkCUDAError("generate_new_level");


		// move forward tail of MatVecs queue and update size
		mat_vec_data_at_current_offset = &mat_vec_data_at_current_offset[total_new_mat_vecs];
		*mat_vec_data_count = *mat_vec_data_count + total_new_mat_vecs;

		// data on current level is no longer needed -> cleanup
		cudaFree(current_level_data);
		cudaFree(new_mat_vec_counts);
		cudaFree(new_child_counts);
		cudaFree(new_child_offsets);

		// switch to next level
		current_level_data = next_level_data;
		next_level_data = 0;
		total_children = total_new_children;

		// compute new compute configuration for children computations
		grid_size_for_children = (total_children + (block_size - 1)) / block_size;

		if (total_children==0) // stopping when no more children are generated
		{
			cudaFree(current_level_data);
			break;
		}
	}
}



#endif
