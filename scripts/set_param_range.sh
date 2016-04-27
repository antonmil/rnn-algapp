#!/bin/bash
# set a range of parameter in a range of config files
# needs five inputs: name, low val, high val, n_files, config batch
# e.g. sh set_param.sh rnn_size 50 0926c

param=$1
from=$2
to=$3
n=$4
name=$5

for i in $(eval echo "{1..$n}")
do
#   echo $i
  ipol=`echo "($i-1) * ($to-$from) / ($n-1)" | bc -l`
#   echo $ipol
  val=`echo "$from+$ipol" | bc -l`
  echo $val
  
  # replace
  sed -i s/'\s'$param'\s'.*/' '$param'  '$val/g ../config/$name-$i.txt
done;
# sed -i s/'\s'$1.*/' '$1'  '$2/g config/$3*
