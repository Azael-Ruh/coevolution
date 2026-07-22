#!/bin/bash
# 0.2 \"Normal(0, 2)\" 100000 500"
r=40
R0=1.2
muVect=(0.06 0.08 0.10 0.12 0.14 0.16 0.18 0.20 0.22 0.24 0.26 0.28 0.30 0.32 0.34 0.36 0.38 0.40)
mutKernel="Normal(0,1)"
Nh=10000000
tmax=500
totalRuns=8000

printf "" > /home/zayas-orihuela/coevolution/scripts/1D/mutantGrowth/cluster/params.txt

i=0
for mu in "${muVect[@]}" 
do 
    printf "%s\n" "$r $R0 $mu $mutKernel $Nh $tmax $totalRuns" >> /home/zayas-orihuela/coevolution/scripts/1D/mutantGrowth/cluster/params.txt
    ((i++))    
done

echo "produced $i different combinations"
