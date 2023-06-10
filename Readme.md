# UW-CycleGAN: Model-driven CycleGAN for Underwater Image Restoration
This is an implement of the UW_CycleGAN,
**“[UW-CycleGAN: Model-driven CycleGAN for Underwater Image Restoration]”**, 
Haorui Yan , Zhenwei Zhang , and Yuping Duan* ,IEEE Transactions on Circuits and Systems for Video Technology, 2023.

## Overview
![avatar](network.PNG)

## Prerequisites
- PyTorch 1.11.0
- Python 3.7
- NVIDIA GPU + CUDA cuDNN

## Installation
Type the command:
```
pip install -r requirements.txt
```

## Testing
We use several datasets: 
[UIEB](https://li-chongyi.github.io/proj_benchmark.html), 
[EUVP](http://irvlab.cs.umn.edu/resources/euvp-dataset), 
[U45](https://github.com/IPNUISTlegal/underwater-test-dataset-U45-), 
[SQUID]( http://csms.haifa.ac.il/profiles/tTreibitz/datasets), 
[RUIE](https://github.com/dlut-dimt/RealworldUnderwater-Image-Enhancement-RUIE-Benchmark) 
for testing.

After downloading the dataset, put it in the folder _./test_img_
Then modify the address of the dataset in _test_cycle.py_

Change the address of weight file of the model you want to test in _test_cycle.py_, 
our trained model weights are in _./model_ folder

run 
​```
python test_cycle.py
​``` 

The test results will be in _./test_result_ folder.

## Training
We use the in-air dataset RGB-D [NYU-V1](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v1.html) 
and the [UIEB](https://li-chongyi.github.io/proj_benchmark.html) for unpaired training.

The training sets are put in the _./dataset_ folder.

To train the model, run
​```
python train_cycle.py
​``` 

You can set the pretrained weights in _train_cycle.py_ and train on the pretrained model.

The running files will be saved to _./runs/exp1_ folder.

## Contact
Should you have any question, please contact [Haorui Yan].

[Haorui Yan]: yanhaorui1520@163.com