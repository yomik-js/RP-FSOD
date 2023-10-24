# RP-FSOD
<div align="center"><img src="assets/header.png" width="840"></div>

## Introduction

This repository contains the source code for our paper "Retentive Compensation and Personality Filtering for Few-Shot Remote Sensing Object Detection" by Jiashan Wu, Chunbo Lang, Gong Cheng, Xingxing Xie, and Junwei Han..


## Quick Start

**1. Check Requirements**
* Linux with Python >= 3.8
* [PyTorch](https://pytorch.org/get-started/locally/) >= 1.6 & [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch version.
* CUDA 10.1
* GCC >= 4.9

**2. Build RP-FSOD**
* Clone Code
  ```angular2html
  git clone https://github.com/yomik-js/RP-FSOD.git
  cd RCPF
  ```
* Create a virtual environment (optional)
  ```angular2html
  conda create -n RP-FSOD python==3.8.0
  conda activate RP-FSOD
  ```
* Install PyTorch 1.7.1 with CUDA 10.1 
  ```shell
  pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html 

  ```
* Install Detectron2
  ```angular2html
  python3 -m pip install detectron2==0.3 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.7/index.html
  ```
  - If you use other version of PyTorch/CUDA, check the latest version of Detectron2 in this page: [Detectron2](https://github.com/facebookresearch/detectron2/releases). 
  - Sorry for that I don’t have enough time to test on more versions, if you run into problems with other versions, please let me know.
* Install other requirements. 
  ```angular2html
  python3 -m pip install -r requirements.txt
  ```

**3. Prepare Data and Weights**
* Data Preparation
  - We evaluate our models on DIOR and NWPU VHR-10.v2, put them into `datasets` and put it into your project directory:
    ```angular2html
      ...
      datasets
        | -- DIOR (JPEGImages/*.jpg, ImageSets/, Annotations/*.xml)
        | -- DIORsplit
        | -- NWPU
        | -- NWPUsplit
      RCPF
      tools
      ...
    ```
* Weights Preparation
  - We use the imagenet pretrain weights to initialize our model. Download the same models from here: [GoogleDrive](https://drive.google.com/file/d/1rsE20_fSkYeIhFaNU04rBfEDkMENLibj/view?usp=sharing)

**4. Training and Evaluation**

For ease of training and evaluation over multiple runs, we integrate the whole pipeline of few-shot object detection into one script `run_*.sh`, including base pre-training and novel-finetuning.
* To reproduce the results on VOC, `EXP_NAME` can be any string and `SPLIT_ID` must be `1 or 2 or 3 or 4` .
  ```angular2html
  bash run_voc.sh EXP_NAME SPLIT_ID
  ```
* Please read the details of few-shot object detection pipeline in `run_*.sh`, you need change `IMAGENET_PRETRAIN*` to your path.

## Acknowledgement
This repo is developed based on [DeFRCN](https://github.com/er-muyue/DeFRCN). Please check it for more details and features.
