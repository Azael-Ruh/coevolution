#!/bin/bash
# 0.2 \"Normal(0, 2)\" 100000 500"
rVect=(0 1 2 3 4 5 10 15 20 30 40 50 60 80 100)
R0Vect=(1.05 1.1 1.2 1.3 1.4 1.5 1.8 2 2.4 2.8 3.5 4.2 5 6 7 8 9 10)

printf "" > /home/zayas-orihuela/coevolution/scripts/1D/params.txt

i=0
for r in "${rVect[@]}" 
do 
    for R0 in "${R0Vect[@]}"
    do
        printf "%s\n" "$r $R0" >> /home/zayas-orihuela/coevolution/scripts/1D/params.txt
        ((i++))
    done
done

echo "produced $i different combinations"
