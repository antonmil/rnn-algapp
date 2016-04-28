#!/bin/sh


if [[ $# -eq 1 ]]; then
  mkdir -p ../logs/zipped
  
  # remove broken EOL characters
  sed -i 's/\[0m//g' ../logs/$1*
  
  # zip logs
  zip ../logs/zipped/$1.zip -q ../logs/$1*; 
  
  # remove originals
  rm -f ../logs/$1*;
else
  echo "Illegal number of parameters"
fi

