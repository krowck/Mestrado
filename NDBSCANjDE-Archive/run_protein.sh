for j in {0..30}
do
        taskset -c "$j" python3 ndbjde.py -acc 0.001 -a 1 -flag 0 -r 1 -p 150 -f 21 &
done