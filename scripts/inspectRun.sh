#!/bin/sh
cd ../src/

#find . -name mota* | sort | grep $1
# th inspectRun.lua -model_name $1


# ## declare an array variable
# declare -a arr=("a" "b" "c" "d" "e" "f" "g" "h" "i" "j" "k" "l" "m" "n" "o" "p" "q" "r" "x" "y" "z")
# 
# ## now loop through the above array
# for i in "${arr[@]}"
# do
#   echo $1$i
#   dirsEx=`ls tmp/$1$i*/bm.txt -R 2>/dev/null | wc -l`	# -R >... redirects errors to null
#   if [[ $dirsEx -ne 0 ]]
#   then
#     th inspectRun.lua -model_name $1$i
#   fi
# done

if [[ $# -eq 1 ]]; then
	th inspectRun.lua -main_name $1
elif [[ $# -eq 2 ]]; then
	th inspectRunNP.lua -main_name $1
fi

sleep 1
scp ../tmp/$1*.png amilan@129.127.10.41:~/Dropbox/research/rnn-algapp/tmp/
