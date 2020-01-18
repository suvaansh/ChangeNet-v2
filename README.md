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

The **`main.py`** script is used for training. It trains the model iteratively over the entire dataset for the specified number of epochs. Use the following command for training the baseline model provided in this repository. The baseline experiment used Adam Optimiser with `1e-4` as initial learning rate. The model trains for `50 epochs` by default.

```
python main.py --data /path/to/dataset/VL_CMU_CD
```

We can resume training from a saved checkpoint by using the `resume` option and passing the checkpoint path as argument: 

```
python main.py --data /path/to/dataset/VL_CMU_CD --resume model/checkpoint.pth.tar
```

We can train our model on multiple GPUs using the `device_ids` option and passing the device ids as arguments as a `string`. 

```
python main.py --data /path/to/dataset/VL_CMU_CD --device_ids "gpu ids separated by commas (e.g. 0,1,2,...)"
```

Evaluation
----------
The **`main.py`** script along with `evaluate` flag is  used for the purpose of evaluation. It takes a pretrained model and evaluate the model on the image ids present in the csv file passed as an argument with `efile` option.

```
python main.py --data /path/to/dataset/VL_CMU_CD --resume /path/to/saved/model.pth.tar --evaluate --efile test 
```
The above command will test the trained model on `test.csv` file

### Metrics

The metrics used for evaluation are:

**Precision**: Precision tells us about how accurate our model is. Means, out of the predicted positive pixels, how many of are actually positive.

**Recall**: Recall calculates out of all the Actual Positives, how many can our model identify by labelling them as positive.

**F Measure**: F Measure is the harmonic mean of Precision and Recall. We need this metric when we need to maintain a balance between the both. F Measure's value goes down if either of the 2 have low value. Which makes it the perfect metric for class imbalanced datasets 




[4]: http://www.robesafe.com/personal/roberto.arroyo/docs/Alcantarilla16rss.pdf
[5]: https://ghsi.github.io/proj/RSS2016.html
[7]: https://pytorch.org
[8]: https://pypi.python.org/pypi/tqdm
[9]: https://pytorch.org/docs/stable/torchvision/index.html
