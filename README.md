# My Paper Title

This repository is the official implementation of [My Paper Title](https://arxiv.org/abs/2030.12345). 

## Requirements

Experiments were done with the following package versions for Python 3.6:
 - PyTorch (`torch`) v1.2.0 with CUDA 9.0;
 - Torchvision (`torchvision`) v0.4.0;
 - Numpy (`numpy`) v1.14.5;
 - Imblearn (`imblearn`) v0.5.0;
 - Cnn_finetune (`cnn_finetune`) v0.6.0;
 - PIL (`PIL`) v6.1.0;
 - Scikit-learn (`sklearn`) v0.21.3;
 - Scipy (`scipy`) v1.1.0;
 - Csv (`csv`) v1.0.
 - math, os, collections, pylab.
 
## Datasets

The datasets manipulated in this code can be downloaded on the following locations:
1. LFWface:
  - Baiduyun: https://pan.baidu.com/s/1QIM8ZVbitRopECfGCKTaZg,    Extraction code: 39t9;

2. animal:
  - Baiduyun: https://pan.baidu.com/s/1dTd80kqvT7TnSZZcVdeVZA,    Extraction code: ixqe.  

Download correspond dataset to folder 'data'.

## Training

1. To train the baseline model: 
 - LFWface
```train
cd /focus/code_face
python train_baseline.py
```
 - animal
 ```train
cd /focus/code_animal
python train_baseline.py
```
2. To train the best model in the paper, run this command:
 - LFWface
```train
cd /focus/code_face
python train_caps_cam.py
```
 - animal
 ```train
cd /focus/code_animal
python train_caps_cam.py
```
When training this model, if training accuracy does not increase, multiply the learning rate by 0.1.

## Evaluation

1. To evaluate the baseline model on LFWface/animal, run:
```eval
cd /focus/code_face or cd /focus/code_animal
python test_baseline.py 
```
2. To evaluate our model on LFWface/animal, run:
```eval
cd /focus/code_face or cd /focus/code_animal
python test_caps_cam.py 
```

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

> 📋Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### Focus Classification on LFWface

|     Model name     | Accuracy | Precision | Recall | F1-Score |
| ------------------ |--------- | --------- |------- |--------- |
|      baseline      |       %  |        %  |        |          |
| My awesome model   |  84.71%  |        %  |        |          |

### Focus Classification on animal

|      Model name    | Accuracy | Precision | Recall | F1-Score |
| ------------------ |--------- | --------- |------- |--------- |
|       baseline     |       %  |        %  |        |          |
| ------------------ |--------- | --------- |------- |--------- |
| My awesome model   |  77.44%  |        %  |        |          |

## Contributing

Our paper is submited to NeurIPS 2020. If use our codes, please cite our paper.
