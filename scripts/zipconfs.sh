#!/bin/sh


if [[ $# -eq 1 ]]; then
  mkdir -p ../config/zipped
  
  # zip confs
  zip ../config/zipped/$1.zip -q ../config/$1*.txt; 
  
  # remove originals
  rm -f ../config/$1*.txt;
else
  echo "Illegal number of parameters"
fi

