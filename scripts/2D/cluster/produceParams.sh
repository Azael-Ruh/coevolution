#!/bin/bash
#SBATCH -p R-Infinite

r=30
R0=2.5
muVect=(0.10 0.12 0.14 0.16 0.18 0.20)
nonLocalMutProb="2e-6"
nonLocalJumpVect=(0 10 20 30 40 50 60 70)
Nh=10000000
tmax=300
nCycles=100

printf "" > /home/zayas-orihuela/coevolution/scripts/2D/cluster/params.txt

i=0
for nonLocalJump in "${nonLocalJumpVect[@]}" 
do 
    for mu in "${muVect[@]}"
    do
        printf "%s\n" "$r $R0 $mu $nonLocalJump $nonLocalMutProb $Nh $tmax $nCycles" >> /home/zayas-orihuela/coevolution/scripts/2D/cluster/params.txt
        ((i++))
    done
done

echo "produced $i different combinations"
