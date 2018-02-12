#! /bin/bash
grep "apply_h_matrix_mvp" | cut -d' ' -f4 | paste -s -d+ - | bc -l | awk '{print $1"/5.0"}' | bc -l
