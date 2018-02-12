#! /bin/bash
grep "dense blocks" | cut -d' ' -f7 | paste -s -d+ - | bc -l | awk '{print $1"/5.0"}' | bc -l
