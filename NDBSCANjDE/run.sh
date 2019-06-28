#!/bin/bash
declare -a func=(1)
for j in {0..0}
do
	python3 ndbjde.py -acc 0.001 -a 1 -flag 0 -r 10 -p 150 -f "${func[$j]}"
done
