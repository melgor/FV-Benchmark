25/08/2015
author: Bartosz Ludwiczuk


This file is explaining how matching between images should be calculated.

1. You should firstly run several first line of IPython notebook: "Create Gallery"
2. You should produce file "verification_felix.npy"

Some explanation of process: this code calcualte matching between all images in same class to choose best image along all.
As there is a lot of matching to calculate, we have to divide it to several subtask. Firstly, divide "verification_felix.npy" to smaller set using:
python divide_verification.py
It will divide it to 100 parts (you can change that parameter), producing *_data.npy files


Then, as we have 100 parts, we would like to run calculating matching between images. 
Run parallel computing by:
ls *_data.npy | parallel python calculate_matches.py
This will take some time and produce files scores_*_data.npy.

Then, we will create txt file with pathches to it by:
for data in scores*.npy; do CONTAININGDIR=$(realpath ${data%/*}); echo $CONTAININGDIR; done > all_scores.txt

After all process, return to IPython notebook. 

