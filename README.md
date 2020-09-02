## News
![#f03c15](https://placehold.it/15/f03c15/000000?text=+) Python3 implementations of PSENet [1], PAN [2] and PAN++ [3] are released at https://github.com/whai362/pan_pp.pytorch.

[1] W. Wang, E. Xie, X. Li, W. Hou, T. Lu, G. Yu, and S. Shao. Shape robust text detection with progressive scale expansion network. In Proc. IEEE Conf. Comp. Vis. Patt. Recogn., pages 9336–9345, 2019.<br>
[2] W. Wang, E. Xie, X. Song, Y. Zang, W. Wang, T. Lu, G. Yu, and C. Shen. Efficient and accurate arbitrary-shaped text detection with pixel aggregation network. In Proc. IEEE Int. Conf. Comp. Vis., pages 8440–8449, 2019.<br>
[3] Paper is in preparation.

# Shape Robust Text Detection with Progressive Scale Expansion Network

## Requirements
* Python 2.7
* PyTorch v0.4.1+
* pyclipper
* Polygon2
* OpenCV 3.4 (for c++ version pse)
* opencv-python 3.4

## Introduction
Progressive Scale Expansion Network (PSENet) is a text detector which is able to well detect the arbitrary-shape text in natural scene.

## Training
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_ic15.py
```

## Testing
```
CUDA_VISIBLE_DEVICES=0 python test_ic15.py --scale 1 --resume [path of model]
```

## Eval script for ICDAR 2015 and SCUT-CTW1500
```
cd eval
sh eval_ic15.sh
sh eval_ctw1500.sh
```


## Performance (new version paper)
### [ICDAR 2015](http://rrc.cvc.uab.es/?ch=4&com=evaluation&task=1)
| Method | Extra Data | Precision (%) | Recall (%) | F-measure (%) | FPS (1080Ti) | Model |
| - | - | - | - | - | - | - |
| PSENet-1s (ResNet50) | - | 81.49 | 79.68 | 80.57 | 1.6 | [baiduyun](https://pan.baidu.com/s/17FssfXd-hjsU5i2GGrKD-g)(extract code: rxti); [OneDrive](https://1drv.ms/u/s!Ai5Ldd26Lrzkkx3E1OTZzcNlMz5T) |
| PSENet-1s (ResNet50) | pretrain on IC17 MLT | 86.92 | 84.5 | 85.69 | 1.6 | [baiduyun](https://pan.baidu.com/s/1oKVxHKuT3hdzDUmksbcgAQ)(extract code: aieo); [OneDrive](https://1drv.ms/u/s!Ai5Ldd26Lrzkkx44xvpay4rbV4nW) |
| PSENet-4s (ResNet50) | pretrain on IC17 MLT | 86.1 | 83.77 | 84.92 | 3.8 | [baiduyun](https://pan.baidu.com/s/1oKVxHKuT3hdzDUmksbcgAQ)(extract code: aieo); [OneDrive](https://1drv.ms/u/s!Ai5Ldd26Lrzkkx44xvpay4rbV4nW) |

### [SCUT-CTW1500](https://github.com/Yuliang-Liu/Curve-Text-Detector)
| Method | Extra Data | Precision (%) | Recall (%) | F-measure (%) | FPS (1080Ti) | Model |
| - | - | - | - | - | - | - |
| PSENet-1s (ResNet50) | - | 80.57 | 75.55 | 78.0 | 3.9 | [baiduyun](https://pan.baidu.com/s/1BqJspFwBmHjoqlE0jOrJQg)(extract code: ksv7); [OneDrive](https://1drv.ms/u/s!Ai5Ldd26LrzkkxtlTb-yqBPd1PCn) |
| PSENet-1s (ResNet50) | pretrain on IC17 MLT | 84.84| 79.73 | 82.2 | 3.9 | [baiduyun](https://pan.baidu.com/s/1zonNEABLk4ifseeJtQeS4w)(extract code: z7ac); [OneDrive](https://1drv.ms/u/s!Ai5Ldd26LrzkkxxJcfU1a__6nJTT) |
| PSENet-4s (ResNet50) | pretrain on IC17 MLT | 82.09 | 77.84 | 79.9 | 8.4 | [baiduyun](https://pan.baidu.com/s/1zonNEABLk4ifseeJtQeS4w)(extract code: z7ac); [OneDrive](https://1drv.ms/u/s!Ai5Ldd26LrzkkxxJcfU1a__6nJTT) |

## Performance (old version paper)
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

## Paper Link
[new version paper] [https://arxiv.org/abs/1903.12473](https://arxiv.org/abs/1903.12473)

[old version paper] [https://arxiv.org/abs/1806.02559](https://arxiv.org/abs/1806.02559)

## Other Implements
[tensorflow version (thanks @[liuheng92](https://github.com/liuheng92))] [https://github.com/liuheng92/tensorflow_PSENet](https://github.com/liuheng92/tensorflow_PSENet)

## Citation
```
@inproceedings{wang2019shape,
  title={Shape Robust Text Detection With Progressive Scale Expansion Network},
  author={Wang, Wenhai and Xie, Enze and Li, Xiang and Hou, Wenbo and Lu, Tong and Yu, Gang and Shao, Shuai},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={9336--9345},
  year={2019}
}
```
