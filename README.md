Change Detection
========================================

Code for the Paper

**[ChangeNet-v2: Semantic Change detection with Convolutional Neural Networks][1]**
K. Ram Prabhakar, Akshaya Ramasamy, Suvaansh Bhambri, Jayavardhana Gubbi, R. Venkatesh Babu, Balamuralidhar Purushothaman
<br>Link to be added[]()

Introduction
------------

In this paper, a novel deep learning architecture is proposed for change detection that targets higher-level inferencing. 
The new network architecture involves extracting features using CNN and combining filter outputs at different levels to 
localize the change. Finally, detected changes are identified using the same network, and output is an object-level change
detection with the label. 

The proposed architecture is compared with the state-of-the-art using three different modern
change detection datasets: VL-CMU-CD (Alcantarilla et al. (2018)), TSUNAMI (Sakurada and Okatani(2015)), and GSV 
(Sakurada and Okatani (2015)) datasets.

<img src="https://github.com/suvaansh/CorrNet/blob/master/Images/ChangeNet_Img1.jpg"/>

Setup
-----

This repository has been tested for Python3.

1. Install PyTorch (Python3) by following instructions on [PyTorch Homepage][7].
2. Install [Torchvision][9] via pip3. It is used for incorporating feature extractor(VGG) pretrained on Imagenet
3. Install [tqdm][8] via pip3. It is used for generating pregress bars.

Dataset
------------------
The VL-CMU-CD dataset can be downloaded from the [project page][5] of the paper [Street-View Change Detection with Deconvolutional Networks (RSS'16)][4].
This dataset is availble on request.

#### Dataset Schema

    .
    ├── ...
    ├── VL_CMU_CD                    # Test files (alternatively `spec` or `tests`)
    │   ├── left          # Contains test images
    │   │   ├── 001_00.jpeg
    │   │   ├── 001_01.jpeg
    │   │   └── ...
    │   │
    │   ├── right          # Contains reference images
    │   │   ├── 001_00.jpeg
    │   │   ├── 001_01.jpeg
    │   │   └── ...
    │   │
    │   ├── GT_MULTICLASS          # Contains Groundtruth 3d maps with each channel(11) representing single class at every pixel
    │   │   ├── 001_00.npy
    │   │   ├── 001_01.npy
    │   │   └── ...
    │   │
    │   ├── mask          # Binary Mask for region of interest
    │   │   ├── 001_00.jpeg
    │   │   ├── 001_01.jpeg
    │   │   └── ...
    │   │
    │   └── ...
    │
    └── ...

Training
--------




Evaluation
----------

[4]: http://www.robesafe.com/personal/roberto.arroyo/docs/Alcantarilla16rss.pdf
[5]: https://ghsi.github.io/proj/RSS2016.html
[7]: https://pytorch.org
[8]: https://pypi.python.org/pypi/tqdm
[9]: https://pytorch.org/docs/stable/torchvision/index.html
