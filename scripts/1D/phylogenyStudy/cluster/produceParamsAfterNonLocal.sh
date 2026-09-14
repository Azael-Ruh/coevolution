#!/bin/bash
#SBATCH -p R-Infinite

r=19
R0=1.4
muVect=(0.14)
nonLocalMutProbVect=("5e-6")
nonLocalJump=35
localKernel="Normal(0,1)"
Nh=10000000
tmax=400
xmax=300
runs=200
tAfterNonLocalVect=(10 20 30 40 50 60 70 80 90 100)

printf "" > /home/zayas-orihuela/coevolution/scripts/1D/phylogenyStudy/cluster/params.txt

i=0
for nonLocalMutProb in "${nonLocalMutProbVect[@]}" 
do 
    for mu in "${muVect[@]}"
    do
        for tAfterNonLocal in "${tAfterNonLocalVect[@]}"
        do
            for run in $(seq $runs)
            do
                printf "%s\n" "$r $R0 $mu $localKernel $nonLocalJump $nonLocalMutProb $Nh $tmax $xmax $tAfterNonLocal $run" >> /home/zayas-orihuela/coevolution/scripts/1D/phylogenyStudy/cluster/params.txt
            done
            ((i++))
        done
    done
done

echo "produced $i different combinations $runs times each"
