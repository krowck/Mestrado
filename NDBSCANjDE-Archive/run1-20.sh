#!/bin/bash
declare -a func=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
for j in {0..19}
do
	python3 ndbjde.py -acc 0.001 -a 1 -flag 0 -r 10 -p 150 -f "${func[$j]}" &
done
