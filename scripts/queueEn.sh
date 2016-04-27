#!/bin/sh
#echo $1

# precomp helper structures 
#matlab -nodesktop -nosplash -r "precomputeForCluster('$PBS_JOBNAME'); quit;";

#qsub -t 1-20 -j oe -l walltime=08:00:00 -l vmem=16GB -W depend=afterokarray:69420 -o logs -N $1 trainModel.sh
#qsub -t 1-20 -j oe -l nodes=acvt-node00+acvt-node01+acvt-node02+acvt-node03+acvt-node04+acvt-node05+acvt-node06+acvt-node07+acvt-node08+acvt-node09+acvt-node10+acvt-node11+acvt-node20+acvt-node11+acvt-node22+acvt-node23+acvt-node24+acvt-node25+acvt-node26+acvt-node27+acvt-node28+acvt-node29+acvt-node30+acvt-node31+acvt-node32+acvt-node33 -l walltime=32:00:00 -l vmem=6GB -o logs -N $1 trainModel.sh

subscript='subTrainEn.sh'
#subscript='subTestV.sh'
subscript=`cat ../config/subscript.txt`
echo $subscript

#sleep 1
if [[ $# -eq 2 ]]; then
	qsub -t 1-$2 -j oe -l nodes=1:type3:ppn=1 -l walltime=8:00:00 -l vmem=16GB -o ../logs -N $1 $subscript
elif [[ $# -eq 3 ]]; then
	# qsub -t 1-$2 -j oe -l host=!acvt-node16 -l host=!acvt-node17 -l host=!acvt-node18 -l host=!acvt-node19 -l nodes=1:ppn=$3 -l walltime=48:00:00 -l vmem=16GB -o logs -N $1 subTest.sh
#	qsub -t 1-$2 -j oe -l nodes=1:type3:ppn=$3 -l walltime=96:00:00 -l vmem=64GB -o logs -N $1 $subscript
echo "?"
	
else
        echo "Illegal number of parameters"
fi
