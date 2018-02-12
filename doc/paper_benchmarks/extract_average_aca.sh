#! /bin/bash
grep "ACA blocks" | cut -d' ' -f5 | paste -s -d+ - | bc -l | awk '{print $1"/5.0"}' | bc -l
