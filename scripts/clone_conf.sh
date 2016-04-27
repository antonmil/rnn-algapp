#!/bin/sh

#base='0930b';
base=$1;
for i in $(seq 2 1 $2); 
do 
  cp ../config/$base-1.txt ../config/$base-$i.txt; 
done;
