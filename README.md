# TopNet-Object-Placement
An implementation of the paper "TopNet: Transformer-based Object Placement Network for Image Compositing", CVPR 2023.

## Setup
All the code have been tested on PyTorch 1.7.0. Follow the instructions to run the project.

First, clone the repository:
```
git clone git@github.com:bcmi/TopNet-Object-Placement.git
```
Then, install Anaconda and create a virtual environment:
```
conda create -n TopNet
conda activate TopNet
```
Install PyTorch 1.7.0 (higher version should be fine):
```
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.2 -c pytorch
```
Install necessary packages:
```
pip install -r requirements.txt
```

## Data Preparation
Download and extract data from  [Baidu Cloud](https://pan.baidu.com/s/10JBpXBMZybEl5FTqBlq-hQ)(access code: 4zf9) or [Google Drive](https://drive.google.com/file/d/1VBTCO3QT1hqzXre1wdWlndJR97SI650d/view?pli=1). Put them in "data/data". It should contain the following directories and files:
```
<data/data>
  bg/                         # background images
  fg/                         # foreground images
  mask/                       # foreground masks
  train(test)_pair_new.json   # json annotations 
  train(test)_pair_new.csv    # csv files
```
## Training
Before training, modify "config.py" according to your need. After that, run:
```
python train.py
```
## Test
To get the F1 score and balanced accuracy of a specified model, run:
```
python test.py --load_path <PATH_TO_MODEL> 
```
## Evalution on Discriminative Task

## Evalution on Generation Task
