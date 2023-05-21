# CED 
# CEDFlow: Latent Contour Enhancement for Dark Optical Flow Estimation
This is CEDFlow code.

# Dataset
FCDN & vVBOF can be accessed here https://github.com/mf-zhang/Optical-Flow-in-the-Dark

# Setup
Our code is based on pytorch 1.9.0, CUDA 10.2 and python 3.8. Higher version pytorch should also work well.

# Demos
All pretrained models can be downloaded from BaiDu drive.
link：https://pan.baidu.com/s/1McXLMPoQyNvt57_fCUbzKw 
code：w3gc

FCDNed is trained on FCDN only; Mixed is trained on the FCDN and VBOF in a RTX 3080Ti.

# Training 
You can run a trained model on a sequence of images and visualize the results
after set of stage & lr & batch_size, run train.py file

# Evaluation
evaluation.py is epe test  
evaluation_single.py is dark flow estimation.
