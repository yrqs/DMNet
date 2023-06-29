# Decoupled Metric Network for Single-Stage Few-Shot Object Detection

### Introduction
Our project is based on the public detection toolbox and benchmark [MMDetection v1.1.0](https://github.com/open-mmlab/mmdetection/tree/v1.1.0).

### Quick Start
1. Build
- Clone Code

```
git clone https://github.com/yrqs/DMNet.git
cd DMNet
```
- Install MMDetection (see [MMDet_README](https://gitee.com/yrqs/DMNetFinal/blob/main/MMDet_README.md))
2. Prepare Data
- Refer to [MPSR](https://github.com/jiaxi-wu/MPSR.git). 
- The final dataset file structure is as follows:
```
  ...
  configs
  data
    | -- coco
            | -- annotations
                    | -- instances_train2014_base.json
                    | -- instances_valminusminival2014_base.json
                    | -- instances_minival2014.json
            | -- images
                    | -- trainval2014
    | -- VOCdevkit
            | -- VOC2007
                    ...
                    | -- ImageSets
                            | -- Main
                                    | -- trainval_split*_base.txt
                                    | -- trainval_*shot_novel_standard.txt
                                    | -- test.txt
            | -- VOC2012
                    ...
                    | -- ImageSets
                            | -- Main
                                    | -- trainval_split*_base.txt
                                    | -- trainval_*shot_novel_standard.txt
  ...
```
3. Config files
- Config files are shown below:
```
  configs
    | -- few_shot
            | -- coco
                    | -- dmnet
                            | -- base.py
                            | -- finetune.py
            | -- voc
                    | -- dmnet_split*
                            | -- base.py
                            | -- finetune.py
```
4. Training and Finetuning
- Training on base classes:
```
# remember to change work_dir in dist_train.sh
tools/dist_train.sh config_file num_gpus
```
- Finetuning on all classes:
```
# remember to change load_from in config_file
# remember to change work_dir in dist_finetuning.sh
tools/dist_finetuning.sh config_file num_gpus
```
5. Test
```
# if test on coco, change '--eval mAP' to '--eval bbox'
tools/dist_test.sh config_file checkpoint_file num_gpus
```

### Acknowledgement
This repo is developed based on [MMDetection v1.1.0](https://github.com/open-mmlab/mmdetection/tree/v1.1.0). Please check them for more details and features.