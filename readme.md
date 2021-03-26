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

| Method | Backbone | Min_Scale | Short_Size | Precision (%) | Recall (%) | F-measure (%) |
| :----: | :------: | :-------: | :--------: | ------------- | :--------: | :-----------: |
| PSENet | ResNet50 | 0.4       | 736        | 83.6          | 74.0       | 78.5          |
| PSENet | ResNet50 | 0.4       | 1024       | 83.4          | 75.5       | 79.3          |


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
This project is released under the [Apache 2.0 license](https://github.com/whai362/pan_pp.pytorch/blob/master/LICENSE).
