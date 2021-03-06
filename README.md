# End-to-End-JDNDMSR
End-to-End Learning for Joint Image Demosaicing, Denoising and Super-Resolution ([CVPR 2021](http://cvpr2021.thecvf.com))

## Enviornment Requirements
* Python >= 3
* Keras = 2.3.1
* Tensorflow-gpu = 1.14.0
* Python packages: `pip install opencv-python scipy scikit-image`
```
conda create --name JDNDMSR
conda activate JDNDMSR
conda install keras=2.3.1 tensorflow-gpu=1.14.0 tensorflow   
pip install opencv-python scipy scikit-image
```
## Quick Test
1. Clone github repository
```
git clone https://github.com/xingwz/End-to-End-JDNDMSR.git JDNDMSR
cd JDNDMSR
```
2. Preprocess test data: run `preprocess/preprocess.m` to downscale the image.
Remember to change folders: `Your_GT_Path` and `Your_LR_Path`.
3. Test our [JDNDMSR+ model](https://github.com/xingwz/End-to-End-JDNDMSR/blob/main/models/jdndmsr%2B_model.h5) (trained with data whose scale factor is 2 and noise level is from 0 to 20.)
```
python test.py -test_image_folder_path_LR "Your_LR_Path" -noise 10.0 -scale_factor 2
```
4. If input raw images, remember to shift the pixel to match our model's Bayer pattern (RGGB).
```
python test.py -test_image_folder_path_LR "Your_LR_Path" -input_raw True -offset_x 1
```
## Train Your Model
1. Preprocess data: run `preprocess/preprocess.m`
2. Train your model. The super-resolution can be turned off by set `scale_factor` as 1.
Remember to change folders: `Your_GT_Path` and `Your_LR_Path`.
```
python train.py
```
3. You can transfer parameters from our pre-trained [JDMSR model](https://github.com/xingwz/End-to-End-JDNDMSR/blob/main/models/jdmsr_model.h5) by set parameter `transfer` as `True`.
