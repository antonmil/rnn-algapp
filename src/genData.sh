#!/bin/bash

cd matlab
matlab -nosplash -nodesktop -r "genQBP($1, $2, $3, $4); quit"