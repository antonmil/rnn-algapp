#!/bin/sh
source ~/.bashrc
cd $PBS_O_WORKDIR

cat $PBS_NODEFILE

# e.g. 4
echo ${PBS_ARRAYID}

# eg 58973[4].moby.cs.adelaide.edu.au
echo ${PBS_JOBID}

# eg 0923a-4
echo $PBS_JOBNAME

name=$PBS_JOBNAME
echo $name

cd /home/h3/amilan/research/projects/rnn-algapp/src

th train.lua -config ../config/$name.txt
