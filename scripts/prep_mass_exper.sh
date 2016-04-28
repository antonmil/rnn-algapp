#!/bin/sh

echo 'SUBMITTING...'
subscript=`cat ../config/subscript.txt`
echo $subscript
sleep 3

basename=`date +%m%d`$1
# basename='1109'$1
echo $basename
name=$basename'a'

conftempfile='conf_template.txt'
rangetempfile='range_template.txt'
if [[ $# -eq 3 ]]; then
  conftempfile=$3'.txt'
  rangetempfile=$3'-range.txt'
  cp ../config/$conftempfile ../config/conf_template.txt
  cp ../config/$rangetempfile ../config/range_template.txt
fi
cp ../config/conf_template.txt ../config/$basename'.txt'
cp ../config/range_template.txt ../config/$basename'-range.txt'


# cp config/conf_template.txt config/$name-1.txt
# ./clone_conf.sh $name
# ./set_param_range.sh lrng_rate 0.00020 0.00180 9 $name

while IFS=, read set pname from to n logsp;do
  name=$basename$set
  cp ../config/conf_template.txt ../config/$name-1.txt
  ./clone_conf.sh $name $n
  ./set_param_range.sh $pname $from $to $n $name $logsp

  echo $name
  echo $set
  echo $pname
  echo $from
  echo $to
  echo $n
  echo $logsp

  ##SUBMIT!!!
  if [[ $2 -eq 1 ]]
  then
    echo "Submitting " $name $n
    sh queueEn.sh $name $n
  fi
#   echo "aa"
done < ../config/range_template.txt
