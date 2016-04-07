MATLAB demo code of Sequential Monte Carlo Graph Matching of ECCV 2012

Yumin Suh, Minsu Cho, and Kyoung Mu Lee,
Graph Matching via Sequential Monte Carlo, 
Proc. European Conference on Computer Vision (ECCV), 2012
http://cv.snu.ac.kr/research/~SMCM/

Please cite our work if you find this code useful in your research. 

written by Yumin Suh & Minsu Cho & Jungmin Lee, 2012, Seoul National University, Korea
http://cv.snu.ac.kr/~ysuh/
http://cv.snu.ac.kr/~minsucho/
http://cv.snu.ac.kr/~jungminlee/


Date: 02/08/2013
Version: 1.0

For image matching demo, combine this with some additional codes and data provided in our site.
==================================================================================================


1. Overview

do_GraphMatchingTest.m   : main script for graph matching demo
do_PointMatchingTest.m   : main script for point matching demo

Current test parameters are set to outlier-varing experiments.
For other tests, modify 'setRandomGraph.m' or 'setPointMatching.m'.
Refer to our paper for the settings in our experiments. 
Performance of each algorithm is represented in terms of accuracy, score,and time.
   Accuracy: the ratio between # of true positive matches and # of groundtruth matches.
   Score: the sum of all affinity values related to the matching result.               

setMethods.m        : script for settings of algorithms being tested
                      the list of methods and their options for visualization

SMCM.m              : Matlab function of Sequential Monte Carlo Graph Matching 

If you want to add your own algorithm for comparison, three steps are required:
1. Create 'YOUR_ALGORITHM_NAME' folder in 'Methods' folder. Then put your code in it.
2. Add the folder in the script 'setPath.m' so that your method can be called.
3. Modify 'setMethods.m' for your method. Note that you should follow the 'methods' structure. 

If you find this code useful in your research, please cite our paper
Yumin Suh, Minsu Cho, and Kyoung Mu Lee, Graph Matching via Sequential Monte Carlo, 
Proc. European Conference on Computer Vision (ECCV), 2012


2. References

This code includes our implementation of two graph matching algorithms for comparison:
M. Leordeanu and M. Hebert. "A Spectral Technique for Correspondence Problems Using Pairwise Constraints", ICCV 2005. 
M. Cho, J. Lee, and K.M. Lee, "Reweighted Random Walks for Graph Matching", ECCV 2010.

We utilized some functions of the following public implementations;

bistocastic normalization functions of Timothee Cour's: 
http://www.seas.upenn.edu/~timothee/software/graph_matching/graph_matching.html

Hungarian algorithm of Markus Buehren's for final discretization (optional):
http://www.markusbuehren.de/