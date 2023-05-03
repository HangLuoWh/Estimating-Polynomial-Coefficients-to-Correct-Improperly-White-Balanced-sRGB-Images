# Estimating-Polynomial-Coefficients-to-Correct-Improperly-White-Balanced-sRGB-Images
This is the  official python implementation of the paper &lt;Estimating Polynomial Coefficients to Correct Improperly White-Balanced sRGB Images> published in IEEE Signal Processing Letters 2021.
# Prerequisites
numpy  
pytorch  
opencv  
tensorboard  
torchvision  
pillow  
scikit-image  
tqdm  
# Dataset Preparation
1. Download the [Rendered WB dataset](http://cvil.eecs.yorku.ca/projects/public_html/sRGB_WB_correction/dataset.html).
2. Organize the folder 'train_set' as:  
   train_set  
   ----set1_no_chart  
   --------ground_truth  
   --------input  
   --------ground_truth_metadata  
   --------input_metadata  
   --------train.txt  
   --------vali.txt  
   --------test.txt  
   ----set2  
   --------ground_truth  
   --------input  
   ----cube  
   --------ground_truth  
   --------input  
3. Copy the datasets into corresponding folds.
# Code
# Train form the scratch
Open a terminator and use command 'python train.py'
# Test the method
use command 'python test.py'
# 
