#!/bin/bash

if [[ $# -eq 2 ]]; then
	qstat -u amilan -t | grep $1 | while read -r line; 
	do
	  echo ${line:0:9};
	  qalter ${line:0:9} -l vmem=$2GB
	done
else
	echo "Illegeal number of parameters"
fi
