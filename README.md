# Shape Robust Text Detection with Progressive Scale Expansion Network
Paper: [arXiv](http://arxiv.org/abs/1806.02559)

by Xiang Li, Wenhai Wang, Wenbo Hou, Ruo-Ze Liu, Tong Lu, Jian Yang

DeepInsight@PCALab, Nanjing University of Science and Technology.
IMAGINE Lab@National Key Laboratory for Novel Software Technology, Nanjing University.

## Introduction
Progressive Scale Expansion Network (PSENet) is a text detector which is able to well detect the arbitrary-shape text in natural scene.

Code are coming soon.
<div align="center">
  <img src="https://github.com/whai362/PSENet/blob/master/figure/pipeline.png">
</div>
<p align="center">
  Figure 1: Illustration of our overall pipeline.
</p>

<div align="center">
  <img src="https://github.com/whai362/PSENet/blob/master/figure/pse.png">
</div>
<p align="center">
  Figure 2: The procedure of progressive scale expansion algorithm.
</p>

## Performance
### [ICDAR 2015](http://rrc.cvc.uab.es/?ch=4&com=evaluation&task=1)
| Method | Precision (%) | Recall (%) | F-measure (%) | 
| - | - | - | - |
| PSENet-4s | 87.98 | 83.87 | 85.88 |
| PSENet-2s | 89.30 | 85.22 | 87.21 |
| PSENet-1s | 88.71 | 85.51 | 87.08 |

### [ICDAR 2017 MLT](http://rrc.cvc.uab.es/?ch=8&com=evaluation&task=1)
| Method | Precision (%) | Recall (%) | F-measure (%) | 
| - | - | - | - |
| PSENet-4s | 75.98 | 67.56 | 71.52 |
| PSENet-2s | 76.97 | 68.35 | 72.40 |
| PSENet-1s | 77.01 | 68.40 | 72.45 |

### [SCUT-CTW1500](https://github.com/Yuliang-Liu/Curve-Text-Detector)
| Method | Precision (%) | Recall (%) | F-measure (%) | 
| - | - | - | - |
| PSENet-4s | 80.49 | 78.13 | 79.29 |
| PSENet-2s | 81.95 | 79.30 | 80.60 |
| PSENet-1s | 82.50 | 79.89 | 81.17 |

### [ICPR MTWI 2018 Challenge 2](https://tianchi.aliyun.com/competition/rankingList.htm?spm=5176.100067.5678.4.65166a80jnPm5W&raceId=231651)
| Method | Precision (%) | Recall (%) | F-measure (%) | 
| - | - | - | - |
| PSENet-1s | 78.5 | 72.1 | 75.2 |

## Results
<div align="center">
  <img src="https://github.com/whai362/PSENet/blob/master/figure/res0.png">
</div>
<p align="center">
  Figure 3: The results on ICDAR 2015, ICDAR 2017 MLT and SCUT-CTW1500
</p>