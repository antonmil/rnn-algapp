#!/bin/bash

if [[ $# -ne 2 ]]; then
  echo "Illegal number of parameters"
  exit;
fi


njobs=`qstat -u amilan | grep $1 | wc -l`



if [[ $2 -gt 0 ]]; then
  echo "Deleting "$njobs" jobs..."
  qstat -u amilan | grep $1 | while read -r line; do
    jobid=${line:0:6}
    echo $jobid
    qdel $jobid[]
  done
  echo "Done!"
  if [[ $2 -eq 2 ]]; then
	  echo "Deleting logs..."
	  sleep 1
	  rm ../logs/$1*
	  echo "Done!"
  fi

  if [[ $2 -eq 2 ]]; then
	echo "Deleting tmp and out ..."
	sleep 1
	rm -rf ../out/$1*
	rm -rf ../tmp/$1*
	echo "Done!"
  fi
else
  echo "Delete "$njobs" jobs?"
fi
