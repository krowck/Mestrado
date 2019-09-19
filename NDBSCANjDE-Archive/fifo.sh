N=4


(
for thing in {0..9999999}; do 
   ((i=i%N)); ((i++==0)) && wait
   echo "$thing" & 
done
)