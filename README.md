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
This dataset is available on request.

#### Dataset Schema

    .
    ├── ...
    ├── VL_CMU_CD                    # Test files (alternatively `spec` or `tests`)
    │   ├── left                     # Contains test images
    │   │   ├── 001_00.jpeg
    │   │   ├── 001_01.jpeg
    │   │   └── ...
    │   │
    │   ├── right                    # Contains reference images
    │   │   ├── 001_00.jpeg
    │   │   ├── 001_01.jpeg
    │   │   └── ...
    │   │
    │   ├── GT_MULTICLASS            # Contains Groundtruth 3d maps with each channel(11) representing single class at every pixel
    │   │   ├── 001_00.npy
    │   │   ├── 001_01.npy
    │   │   └── ...
    │   │
    │   ├── mask                     # Binary Mask for region of interest
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

```sh
python3 main.py --data /path/to/dataset/VL_CMU_CD
```

We can resume training from a saved checkpoint by using the `resume` option and passing the checkpoint path as argument: 

```sh
python3 main.py --data /path/to/dataset/VL_CMU_CD --resume models/checkpoint.pth.tar
```

We can train our model on multiple GPUs using the `device_ids` option and passing the device ids as arguments as a `string`. 

```sh
python3 main.py --data /path/to/dataset/VL_CMU_CD --device_ids "gpu ids separated by commas (e.g. 0,1,2,...)"
```

Evaluation
----------
The **`main.py`** script along with `evaluate` flag is  used for the purpose of evaluation. It takes a pretrained model and evaluate the model on the image ids present in the csv file passed as an argument with `efile` option.

```sh
python3 main.py --data /path/to/dataset/VL_CMU_CD --resume /path/to/saved/model.pth.tar --evaluate --efile test 
```
The above command will test the trained model on `test.csv` file


### Metrics

The metrics used for evaluation are:

**Precision**: Precision tells us about how accurate our model is. Means, out of the predicted positive pixels, how many of are actually positive.<br><br>
<img src="https://latex.codecogs.com/svg.latex?\Large&space;Precision=\frac{TruePositives}{TruePositives+FalsePositives}" title="\Large Precision=\frac{TruePositives}{TruePositives+FalsePositives}" />
<br>

**Recall**: Recall calculates out of all the Actual Positives, how many can our model identify by labelling them as positive.<br><br>
<img src="https://latex.codecogs.com/svg.latex?\Large&space;Recall=\frac{TruePositives}{TruePositives+FalseNegatives}" title="\Large Recall=\frac{TruePositives}{TruePositives+FalseNegatives}" />
<br>

**F Measure**: F Measure is the harmonic mean of Precision and Recall. We need this metric when we need to maintain a balance between the both. F Measure's value goes down if either of the 2 have low value. Which makes it the perfect metric for class imbalanced datasets <br><br> 
<img src="https://latex.codecogs.com/svg.latex?\Large&space;FMeasure=\frac{2*Precision*Recall}{Precision+Recall}" title="\Large FMeasure=\frac{2*Precision*Recall}{Precision+Recall}" />
<br>

Best Checkpoint
---------------
Best performing checkpoint has been made available in this repository [here](https://github.com/suvaansh/CorrNet/tree/master/models)

TODO: Add inferencing code using trained checkpoint

Reported Results
----------------

<table>
    <caption> Analysis of ChangeNet-v2 results at class level on VL-CMU-CD data set. </caption>
    <thead>
        <tr>
            <th>Classification</th>
            <th>Class→  <br>----------<br>  Metric↓</th>
            <th>Barrier</th>
            <th>Bin</th>
            <th>Construction</th>
            <th>Other Objects</th>
            <th>Person Bicycle</th>
            <th>Rubbish Bin</th>
            <th>Sign Board</th>
            <th>Traffic Cone</th>
            <th>Vehicle</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <th rowspan=3>Pixel Based</th>
            <td>Precision</td>
            <td >0.74</td>
            <td>0.76</td>
            <td>0.90</td>
            <td>0.67</td>
            <td>0.84</td>
            <td>0.56</td>
            <td>0.78</td>
            <td>0.67</td>
            <td>0.92</td>            
        </tr>
        <tr>
            <td>Recall</td>
            <td>0.70</td>
            <td>0.72</td>
            <td>0.85</td>
            <td>0.65</td>
            <td>0.79</td>
            <td>0.50</td>
            <td>0.69</td>
            <td>0.60</td>
            <td>0.88</td>
        </tr>
        <tr>
            <td>F_Measure</td>
            <td>0.72</td>
            <td>0.74</td>
            <td>0.87</td>
            <td>0.66</td>
            <td>0.81</td>
            <td>0.53</td>
            <td>0.73</td>
            <td>0.63</td>
            <td>0.90</td>
        </tr>
        <tr>
            <th rowspan=3>Object Based</th>
            <td>Precision</td>
            <td>1.00</td>
            <td>0.97</td>
            <td>0.88</td>
            <td>1.00</td>
            <td>1.00</td>
            <td>0.96</td>
            <td>1.00</td>
            <td>1.00</td>
            <td>1.00</td>            
        </tr>
        <tr>
            <td>Recall</td>
            <td>0.78</td>
            <td>1.00</td>
            <td>1.00</td>
            <td>0.63</td>
            <td>1.00</td>
            <td>1.00</td>
            <td>0.87</td>
            <td>0.58</td>
            <td>0.97</td>
        </tr>
        <tr>
            <td>F_Measure</td>
            <td>0.87</td>
            <td>0.98</td>
            <td>0.94</td>
            <td>0.78</td>
            <td>1.00</td>
            <td>0.97</td>
            <td>0.93</td>
            <td>0.73</td>
            <td>0.98</td>
        </tr>    
    </tbody>
</table>

<br><br>

<table>
    <caption> Average results of 5-fold cross validation for binary and multi-class
categories in VL-CMU-CD dataset. </caption>
    <thead>
        <tr>
            <th></th>
            <th>Accuracy</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>f-score</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <th>Binary</th>
            <td>99.2</td>
            <td>93.7</td>
            <td>93.9</td>
            <td>93.8</td>
        </tr>
        <tr>
            <th>Multi-class</th>
            <td>78.5</td>
            <td>76.0</td>
            <td>71.3</td>
            <td>73.4</td>
        </tr>
    </tbody>
</table>

<br><br>

<table>
    <caption> The quantitative comparison of our method with other approaches for
FPR = 0.1 and FPR = 0.01. </caption>
    <thead>
        <tr>
            <th></th>
            <th colspan=3>FPR = 0.1</th>
            <th colspan=3>FPR = 0.01</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <th>Metric→  <br>----------<br>  Methods↓</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>f-score</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>f-score</th>
        </tr>
        <tr>
            <th>Super-pixel</th>
            <td>0.17</td>
            <td>0.35</td>
            <td>0.23</td>
            <td>0.23</td>
            <td>0.12</td>
            <td>0.15</td>
        </tr>
        <tr>
            <th>CDnet</th>
            <td>0.40</td>
            <td>0.85</td>
            <td>0.55</td>
            <td>0.79</td>
            <td>0.46</td>
            <td>0.58</td>
        </tr>
        <tr>
            <th>ChangeNet</th>
            <td>0.79</td>
            <td>0.80</td>
            <td>0.79</td>
            <td>0.80</td>
            <td>0.79</td>
            <td>0.79</td>
        </tr>
        <tr>
            <th>ChangeNet-v2</th>
            <td>0.93</td>
            <td>0.93</td>
            <td>0.93</td>
            <td>0.90</td>
            <td>0.94</td>
            <td>0.93</td>         
        </tr>
    </tbody>
</table>


[4]: http://www.robesafe.com/personal/roberto.arroyo/docs/Alcantarilla16rss.pdf
[5]: https://ghsi.github.io/proj/RSS2016.html
[7]: https://pytorch.org
[8]: https://pypi.python.org/pypi/tqdm
[9]: https://pytorch.org/docs/stable/torchvision/index.html
