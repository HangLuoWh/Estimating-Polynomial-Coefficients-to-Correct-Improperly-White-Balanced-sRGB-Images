# Estimating-Polynomial-Coefficients-to-Correct-Improperly-White-Balanced-sRGB-Images
This is the  official python implementation of the paper &lt;Estimating Polynomial Coefficients to Correct Improperly White-Balanced sRGB Images> published in IEEE Signal Processing Letters 2021.
# Prerequisites
1. pytorch
2. numpy
3. opencv
4. numpy
5. tqdm
6. tensorboard
7. Pillow
8. torchvision
# Dataset preparation
1. Download the [Rendered WB dataset](https://cvil.eecs.yorku.ca/projects/public_html/sRGB_WB_correction/dataset.html).
2. create a directory 'train_set', and arrange these datasets as:  
train_set  
----set_no_chart  
--------ground_truth  
--------ground_truth_metadata  
--------input  
--------input_metadata  
----set2  
--------ground_truth  
--------input  
----cube  
--------ground_truth  
--------input  
3. Put test.txt, train.txt and vali.txt into 'train_set/set_no_chart'
