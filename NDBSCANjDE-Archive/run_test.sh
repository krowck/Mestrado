N=4

for i in {1..20}
do
	for j in {0..3}
	do
		((k=k%N)); ((k++==0)) && wait
		taskset -c "$j" python3 ndbjde.py -acc 0.001 -a 1 -flag 0 -r 1 -p 150 -f "$i" &
	done
done 
