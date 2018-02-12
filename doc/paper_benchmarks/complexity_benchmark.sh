#!/usr/bin/env bash

#SBATCH --qos=short
#SBATCH -N 1
#SBATCH -p p8_p100
#SBATCH -t 240
#     #SBATCH -o complexity_benchmark.out


export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/home-2/pzaspel/gsllib/lib:/home-2/pzaspel/magmalib/lib:/home-2/pzaspel/OpenBLASlib/lib:.
export PATH=/usr/local/cuda/bin:$PATH

cd /home-2/pzaspel/hmglib/src

for d in {2..3}
do

	echo -n "" > results/complexity_benchmark_with_precomputing_dim_${d}.txt
	for i in {14..20}
	do
		echo $i
		./paper_benchmark $[2**$i] $[2**$i] 16 2048 1.5 1 $d $[2**27] $[2**25] 1 1 > results/complexity_benchmark_${i}_with_precomputing_dim_${d}.txt
		
		echo $i `cat results/complexity_benchmark_${i}_with_precomputing_dim_${d}.txt | ./extract_average_h_mvp.sh` >> results/complexity_benchmark_with_precomputing_dim_${d}.txt
	done
	
	echo -n "" > results/complexity_benchmark_without_precomputing_dim_${d}.txt
	for i in {14..26}
	do
		./paper_benchmark $[2**$i] $[2**$i] 16 2048 1.5 1 $d $[2**27] $[2**25] 0 1 > results/complexity_benchmark_${i}_without_precomputing_dim_${d}.txt
		
		echo $i `cat results/complexity_benchmark_${i}_without_precomputing_dim_${d}.txt | ./extract_average_nonbatch_h_mvp.sh` >> results/complexity_benchmark_without_precomputing_dim_${d}.txt
	done

done

