#!/bin/bash
# get max mota from certain experiment

minen=100
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
	#   echo "$line > $minen" | bc -l
	  cmp=`echo "$line < $minen" | bc -l`
	  if [[ $cmp -eq 1 ]]
	  then
	    minen=$line
	#     echo $minen
	  fi
	  echo $minen > tmpmota
  
	done
fi

# echo $OUTPUT
if [ -f tmpmota ]
then
  minen=`cat tmpmota`
fi
printf %.1f $(echo "$minen" | bc -l)
# echo "done"
# echo $minen
