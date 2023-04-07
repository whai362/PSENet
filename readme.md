## News
- PSENet is included in [MMOCR](https://github.com/open-mmlab/mmocr).
- We have upgraded PSENet from python2 to python3. You can find the old version [here](https://github.com/whai362/PSENet/tree/python2).
- We have implemented PSENet using Paddle. Visit it [here](https://github.com/RoseSakurai/PSENet_paddle).
- You can find code of PAN [here](https://github.com/whai362/pan_pp.pytorch).
- Another group also implemented PSENet using Paddle. You can visit it [here](https://github.com/PaddleEdu/OCR-models-PaddlePaddle/tree/main/PSENet). You can also have a try online with all the environment ready [here](https://aistudio.baidu.com/aistudio/projectdetail/1945560).

## Introduction
Official Pytorch implementations of PSENet [1].

[1] W. Wang, E. Xie, X. Li, W. Hou, T. Lu, G. Yu, and S. Shao. Shape robust text detection with progressive scale expansion network. In Proc. IEEE Conf. Comp. Vis. Patt. Recogn., pages 9336–9345, 2019.<br>


## Recommended environment
```
Python 3.6+
Pytorch 1.1.0
torchvision 0.3
mmcv 0.2.12
editdistance
Polygon3
pyclipper
opencv-python 3.4.2.17
Cython
```

## Install
```shell script
pip install -r requirement.txt
./compile.sh
```

## Training
```shell script
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py ${CONFIG_FILE}
```
For example:
```shell script
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py config/psenet/psenet_r50_ic15_736.py
```

## Test
```
python test.py ${CONFIG_FILE} ${CHECKPOINT_FILE}
```
For example:
```shell script
python test.py config/psenet/psenet_r50_ic15_736.py checkpoints/psenet_r50_ic15_736/checkpoint.pth.tar
```

## Speed
```shell script
python test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --report_speed
```
For example:
```shell script
python test.py config/psenet/psenet_r50_ic15_736.py checkpoints/psenet_r50_ic15_736/checkpoint.pth.tar --report_speed
```

## Evaluation
## Introduction
The evaluation scripts of ICDAR 2015 (IC15), Total-Text (TT) and CTW1500 (CTW) datasets.
## [ICDAR 2015](https://rrc.cvc.uab.es/?ch=4)
Text detection
```shell script
./eval_ic15.sh
```


## [Total-Text](https://github.com/cs-chan/Total-Text-Dataset)
Text detection
```shell script
./eval_tt.sh
```

## [CTW1500](https://github.com/Yuliang-Liu/Curve-Text-Detector)
Text detection
```shell script
./eval_ctw.sh
```

## Benchmark 
## Results 

[ICDAR 2015](https://rrc.cvc.uab.es/?ch=4)

| Method | Backbone | Fine-tuning | Scale | Config | Precision (%) | Recall (%) | F-measure (%) | Model |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| PSENet | ResNet50 | N | Shorter Side: 736 | [psenet_r50_ic15_736.py](https://github.com/whai362/PSENet/blob/python3/config/psenet/psenet_r50_ic15_736.py) | 83.6 | 74.0 | 78.5 | [Releases](https://github.com/whai362/PSENet/releases/download/checkpoint/psenet_r50_ic15_736.pth.tar) |
| PSENet | ResNet50 | N | Shorter Side: 1024 | [psenet_r50_ic15_1024.py](https://github.com/whai362/PSENet/blob/python3/config/psenet/psenet_r50_ic15_1024.py) | 84.4 | 76.3 | 80.2 | [Releases](https://github.com/whai362/PSENet/releases/download/checkpoint/psenet_r50_ic15_1024.pth.tar) |
| PSENet (paper) | ResNet50 | N | Longer Side: 2240 | - | 81.5 | 79.7 | 80.6 | - | 
| PSENet | ResNet50 | Y | Shorter Side: 736 | [psenet_r50_ic15_736_finetune.py](https://github.com/whai362/PSENet/blob/python3/config/psenet/psenet_r50_ic15_736_finetune.py) | 85.3 | 76.8 | 80.9 | [Releases](https://github.com/whai362/PSENet/releases/download/checkpoint/psenet_r50_ic15_736_finetune.pth.tar) |
| PSENet | ResNet50 | Y | Shorter Side: 1024 | [psenet_r50_ic15_1024_finetune.py](https://github.com/whai362/PSENet/blob/python3/config/psenet/psenet_r50_ic15_1024_finetune.py) | 86.2 | 79.4 | 82.7 | [Releases](https://github.com/whai362/PSENet/releases/download/checkpoint/psenet_r50_ic15_1024_finetune.pth.tar) |
| PSENet (paper) | ResNet50 | Y | Longer Side: 2240 | - | 86.9 | 84.5 | 85.7 | - | 

[CTW1500](https://github.com/Yuliang-Liu/Curve-Text-Detector)

| Method | Backbone | Fine-tuning | Config | Precision (%) | Recall (%) | F-measure (%) | Model |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| PSENet | ResNet50 | N | [psenet_r50_ctw.py](https://github.com/whai362/PSENet/blob/python3/config/psenet/psenet_r50_ctw.py) | 82.6 | 76.4 | 79.4 | [Releases](https://github.com/whai362/PSENet/releases/download/checkpoint/psenet_r50_ctw.pth.tar) |
| PSENet (paper) | ResNet50 | N | - | 80.6 | 75.6 | 78 | - | 
| PSENet | ResNet50 | Y | [psenet_r50_ctw_finetune.py](https://github.com/whai362/PSENet/blob/python3/config/psenet/psenet_r50_ctw_finetune.py) | 84.5 | 79.2 | 81.8 | [Releases](https://github.com/whai362/PSENet/releases/download/checkpoint/psenet_r50_ctw_finetune.pth.tar) |
| PSENet (paper) | ResNet50 | Y | - | 84.8 | 79.7 | 82.2 | - | 

[Total-Text](https://github.com/cs-chan/Total-Text-Dataset)

| Method | Backbone | Fine-tuning | Config | Precision (%) | Recall (%) | F-measure (%) | Model |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| PSENet | ResNet50 | N | [psenet_r50_tt.py](https://github.com/whai362/PSENet/blob/python3/config/psenet/psenet_r50_tt.py) | 87.3 | 77.9 | 82.3 | [Releases](https://github.com/whai362/PSENet/releases/download/checkpoint/psenet_r50_tt.pth.tar) |
| PSENet (paper) | ResNet50 | N | - | 81.8 | 75.1 | 78.3 | - | 
| PSENet | ResNet50 | Y | [psenet_r50_tt_finetune.py](https://github.com/whai362/PSENet/blob/python3/config/psenet/psenet_r50_tt_finetune.py) | 89.3 | 79.6 | 84.2 | [Releases](https://github.com/whai362/PSENet/releases/download/checkpoint/psenet_r50_tt_finetune.pth.tar) |
| PSENet (paper) | ResNet50 | Y | - | 84.0 | 78.0 | 80.9 | - | 

## Citation
```
@inproceedings{wang2019shape,
  title={Shape robust text detection with progressive scale expansion network},
  author={Wang, Wenhai and Xie, Enze and Li, Xiang and Hou, Wenbo and Lu, Tong and Yu, Gang and Shao, Shuai},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={9336--9345},
  year={2019}
}
```

## License
This project is developed and maintained by [IMAGINE Lab@National Key Laboratory for Novel Software Technology, Nanjing University](https://cs.nju.edu.cn/lutong/ImagineLab.html).

<img src="logo.jpg" alt="IMAGINE Lab">

This project is released under the [Apache 2.0 license](https://github.com/whai362/pan_pp.pytorch/blob/master/LICENSE).
