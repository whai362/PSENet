# Shape Robust Text Detection with Progressive Scale Expansion Network

## Requirements
* python 2.7
* PyTorch v0.4.1 or Install PyTorch v1.0.0
* pyclipper
* Polygon2
* OpenCV 3+ (for c++ version pse)

# Todo
* CTW1500 train and test

## Introduction
Progressive Scale Expansion Network (PSENet) is a text detector which is able to well detect the arbitrary-shape text in natural scene.

## Training
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_ic15.py
```

## Testing
```
CUDA_VISIBLE_DEVICES=4 python test_ic15.py --scale 1 --resume [path of model]
```


## Performance (new version paper)
### [ICDAR 2015](http://rrc.cvc.uab.es/?ch=4&com=evaluation&task=1)
| Method | Extra Data | Precision (%) | Recall (%) | F-measure (%) | Model |
| - | - | - | - | - | - |
| PSENet-1s (ResNet50) | - | 81.49 | 79.68 | 80.57 | todo |
| PSENet-1s (ResNet50) | pretrain on IC17 MLT | 86.92 | 84.5 | 85.69 | todo |
| PSENet-4s (ResNet50) | pretrain on IC17 MLT | 86.1 | 83.77 | 84.92 | todo |

## Performance (old version paper on arxiv)
### [ICDAR 2015](http://rrc.cvc.uab.es/?ch=4&com=evaluation&task=1) (training with ICDAR 2017 MLT)
| Method | Precision (%) | Recall (%) | F-measure (%) | 
| - | - | - | - |
| PSENet-4s (ResNet152) | 87.98 | 83.87 | 85.88 |
| PSENet-2s (ResNet152) | 89.30 | 85.22 | 87.21 |
| PSENet-1s (ResNet152) | 88.71 | 85.51 | 87.08 |

### [ICDAR 2017 MLT](http://rrc.cvc.uab.es/?ch=8&com=evaluation&task=1)
| Method | Precision (%) | Recall (%) | F-measure (%) | 
| - | - | - | - |
| PSENet-4s (ResNet152) | 75.98 | 67.56 | 71.52 |
| PSENet-2s (ResNet152) | 76.97 | 68.35 | 72.40 |
| PSENet-1s (ResNet152) | 77.01 | 68.40 | 72.45 |

### [SCUT-CTW1500](https://github.com/Yuliang-Liu/Curve-Text-Detector)
| Method | Precision (%) | Recall (%) | F-measure (%) | 
| - | - | - | - |
| PSENet-4s (ResNet152) | 80.49 | 78.13 | 79.29 |
| PSENet-2s (ResNet152) | 81.95 | 79.30 | 80.60 |
| PSENet-1s (ResNet152) | 82.50 | 79.89 | 81.17 |

### [ICPR MTWI 2018 Challenge 2](https://tianchi.aliyun.com/competition/rankingList.htm?spm=5176.100067.5678.4.65166a80jnPm5W&raceId=231651)
| Method | Precision (%) | Recall (%) | F-measure (%) | 
| - | - | - | - |
| PSENet-1s (ResNet152) | 78.5 | 72.1 | 75.2 |

## Results
<div align="center">
  <img src="https://github.com/whai362/PSENet/blob/master/figure/res0.png">
</div>
<p align="center">
  Figure 3: The results on ICDAR 2015, ICDAR 2017 MLT and SCUT-CTW1500
</p>
