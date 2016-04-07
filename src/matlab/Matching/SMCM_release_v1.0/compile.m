cd utils
mex assignmentoptimal.cpp
cd ..

% SMCM
cd Methods
cd SMCM
mex SMC_eccv.cpp particles_eccv.cpp
cd ../..

% RRWM
cd Methods
cd RRWM
mex mexBistocNormalize_match_slack.cpp 
cd ..
cd ..