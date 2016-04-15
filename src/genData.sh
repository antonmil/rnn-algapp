#!/bin/bash

cd matlab
matlab -nosplash -nodesktop -r "addpath(genpath('.')); cd Matching/; DemoCar_Anton; cd ..; genQBP; quit"