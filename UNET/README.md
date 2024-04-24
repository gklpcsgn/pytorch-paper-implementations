## U-NET Paper Implementation

This repository contains the implementation of the U-NET paper. "U-Net: Convolutional Networks for Biomedical Image Segmentation" paper was published in 2015 by Olaf Ronneberger, Philipp Fischer, and Thomas Brox. The paper can be found [here](https://arxiv.org/abs/1505.04597).

### Usage
All of the args has been set to default values. You can change the values by passing the arguments. For example:
```bash
python train.py --lr 0.1 --batch_size 16 --mask-paths "path/to/mask" --image-paths "path/to/image"
```

### Dataset

Dataset is not included in this repository. You can use any dataset you want. This repository makes a object detection and segmentation at the same time. So, you need to have both image and mask data. You can use mask_to_bbox.py to convert your masks to bounding boxes.