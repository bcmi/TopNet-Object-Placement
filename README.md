# TopNet-Object-Placement
This is an unofficial implementation of the paper "TopNet: Transformer-based Object Placement Network for Image Compositing", CVPR 2023. Its idea is similar to our earlier work [FOPA](https://github.com/bcmi/FOPA-Fast-Object-Placement-Assessment). 

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
Install PyTorch 2.0.1 (higher version should be fine):
```
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==0.7.0 cudatoolkit=11.7 -c pytorch
```
Install necessary packages:
```
pip install -r requirements.txt
```

## Data Preparation
Download and extract data from  [Baidu Cloud](https://pan.baidu.com/s/10JBpXBMZybEl5FTqBlq-hQ)(access code: 4zf9) or [Dropbox](https://www.dropbox.com/scl/fi/c05wk038piy224sba6jpi/data.rar?rlkey=tghrxjjgo2g93le64tb1xymvq&st=iaduern9&dl=0). Download the SOPA encoder from [Baidu Cloud](https://pan.baidu.com/s/1hQGm3ryRONRZpNpU66SJZA?_at_=1715224325066) (access code: 1x3n) or [Dropbox](https://www.dropbox.com/scl/fi/tlkbmqebokjloe0i1yfpy/SOPA.pth.tar?rlkey=8mzzc53wy6rjqz69o5lkzusau&st=sos1y1t4&dl=0). Put them in "data/data". It should contain the following directories and files:
```
<data/data>
  bg/                         # background images
  fg/                         # foreground images
  mask/                       # foreground masks
  train(test)_pair_new.json   # json annotations 
  train(test)_pair_new.csv    # csv files
  SOPA.pth.tar                # SOPA encoder
```
## Training
Before training, modify "config.py" according to your need. After that, run:
```
python train.py
```
You can download our pretrained model from [Baidu Cloud](https://pan.baidu.com/s/1TXsNIG4pRAw-wGQUv3doPA?pwd=jx6u) (access code: jx6u) or [Dropbox](https://www.dropbox.com/scl/fi/q3i6fryoumzr15piuq9pr/best_weight.pth?rlkey=wahho3h18k3ntsaw9pvdyfvea&st=ej16vjag&dl=0). 
## Test
To get the F1 score and balanced accuracy of a specified model, run:
```
python test.py --load_path <PATH_TO_MODEL> 
```
## Evalution on Discriminative Task
We show the results on discriminate task compared with SOPA and FOPA.
| Method   | F1   | bAcc   |
|----------|-------|-------|
| [SOPA](https://arxiv.org/abs/2107.01889)   | 0.780 | 0.842 |
| [FOPA](https://arxiv.org/abs/2205.14280)     | 0.776 | 0.840 |
| TopNet   | 0.741 | 0.815 |

## Evalution on Generation Task
Following [FOPA](https://arxiv.org/abs/2205.14280), given each background-foreground pair in the test set, we predict 16 rationality score maps for 16 foreground scales and generate composite images with top 50 rationality scores. Then, we randomly sample one from 50 generated composite images per background-foreground pair for Acc and FID evaluation, using the test scripts provided by [GracoNet](https://arxiv.org/abs/2207.11464).
| Method   | Acc   | FID   |
|----------|-------|-------|
| [TERSE](https://arxiv.org/abs/1904.05475)    | 0.679 | 46.94 |
| [PlaceNet](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580562.pdf) | 0.683 | 36.69 |
| [GracoNet](https://arxiv.org/abs/2207.11464) | 0.847 | 27.75 |
| [IOPRE](https://openreview.net/pdf?id=hwHBaL7wur)    | 0.895 | 21.59 |
| [FOPA](https://arxiv.org/abs/2205.14280)     | 0.932 | 19.76 |
| TopNet   | 0.910 | 23.49 |

## Other Resources

+ [Awesome-Object-Placement](https://github.com/bcmi/Awesome-Object-Placement)
+ [Awesome-Image-Composition](https://github.com/bcmi/Awesome-Object-Insertion)
