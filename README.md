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
# Train form the scratch
Open a terminator and use command 'python train.py'
# Test the method
Download a checkpoint, and we have provided one [here (extraction code: eguf)](https://pan.baidu.com/s/1Qv7xRfWcv42-5BM6gsFqHg). Open a terminator and use command 'python test.py'  
\
If you use our codes please cite our paper:  
H. Luo and X. Wan, "Estimating Polynomial Coefficients to Correct Improperly White-Balanced sRGB Images," in IEEE Signal Processing Letters, vol. 28, pp. 1709-1713, 2021, doi: 10.1109/LSP.2021.3102527.
# Contact
If you have any question, please feel free to contact Hang Luo at hluo@wtu.edu.cn.
# Acknowledge
Some codes in our project are directly taken from [Wb_sRGB](https://github.com/mahmoudnafifi/WB_sRGB) and [Deep White Balance](https://github.com/mahmoudnafifi/Deep_White_Balance).
