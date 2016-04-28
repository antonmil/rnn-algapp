#!/bin/bash
# get max mota from certain experiment

maxobj=-1000
rm -f tmpmota

bmfile='be.txt'
#find tmp/$1* -name bm.txt
tmpex=0
tmpex=`find ../tmp/$1* -name $bmfile -nowarn | wc -l`
#echo $tmpex
if [[ $tmpex -gt 0 ]] 
then
	find ../tmp/$1* -name $bmfile | xargs cat | while read -r line; do
	#   echo $line
	#   echo "$line > $maxobj" | bc -l
	  cmp=`echo "$line > $maxobj" | bc -l`
	  if [[ $cmp -eq 1 ]]
	  then
	    maxobj=$line
	#     echo $maxobj
	  fi
	  echo $maxobj > tmpmota
  
	done
fi

# echo $OUTPUT
if [ -f tmpmota ]
then
  maxobj=`cat tmpmota`
fi
printf %.1f $(echo "$maxobj" | bc -l)
# echo "done"
# echo $maxobj
