# [Task Adaptive Parameter Sharing for Multi-Task Learning](https://arxiv.org/abs/2203.16708)

Unofficial Pytorch implementation of **Task Adaptive Parameter Sharing** (CVPR 2022). <br />


<p align="center">
<img src="./assets/teaser.jpg" width="512"/>
</p>

Task Adaptive Parameter Sharing (TAPS) is a general method for tuning a base model to a new task by adaptively modifying a small, task-specific subset of layers. This enables multi-task learning while minimizing resources used and competition between tasks. TAPS solves a joint optimization problem which determines which layers to share with the base model and the value of the task-specific weights.


## Installation

### Requirements

### Datasets

**ImageNet-to-Sketch**
The 5 datasets comprising ImagetNet-to-Sketch can be download from the [PiggyBack repository](https://github.com/arunmallya/piggyback) at this link: [https://uofi.box.com/s/ixncr3d85guosajywhf7yridszzg5zsq](https://uofi.box.com/s/ixncr3d85guosajywhf7yridszzg5zsq)

**DomainNet**
The 6 DomainNet datasets can be downloaded from the [original website](http://ai.bu.edu/M3SDA/). A formatted version can be downloaded [here](https://drive.google.com/file/d/1GYv-I7febM56xF7Jxdzyi1VZuwDbChd1/view?usp=sharing). The structure of the folder should be the following:
```
├── DomainNet
    ├── sketch
        ├── train
        ├── test
    ├── infograph
        ├── train
        ├── test
    ...
    ├── clipart
        ├── train
        ├── test
```

Place the datasets in the datasets folder. If you choose to place them elsewhere use the --dataset flag to point towards the dataset you would like to fine-tune on.





## Training

### Train Options
Command line arguments that you may want to adjust. For the full list of options see options.py. Arguments for a given experiment are logged in settings.txt of the respective folder.

```
--lam - The sparsity coefficient. Larger lam results in fewer layers being tuned (λ in the paper).
--lr - The learning rate.
--multi_gpu - Trains the model with data parallel if set to true.
--dataset - The relative path to the dataset.
--model_type - Specifies the network architecture. Currently supports ResNet34 and ResNet50. 
                Support for VIT and all convolutional networks coming soon. 
--model_path - Relative path to a pretrained model. Default option uses the pytorch pretrained models.
```

### Sequential TAPS Training
Fine-tune a pretrained ResNet34 with TAPS on the sketch dataset. 
```
python train_sequential.py --dataset ../datasets/DomainNet/sketch --experiment_name \
./results/DN_sketch --multi_gpu --model_type resnet34
```

Fine-tune a pretrained ResNet50 with TAPS on the CUBS dataset. 
```
python train_sequential.py --dataset ../datasets/CUBS_cropped --experiment_name \
./results/CUBS --multi_gpu --model_type resnet50 --lam .1
```



### Joint TAPS Training


## Evaluation

### Tensorboard
To view results run ```tensorboard --logdir=./results``` and navigate to http://localhost:6006/.

We log validation error/training loss/training error/percentage of layers tuned.


### Visualizing Modified Layers

We provide the [visualize_taps.ipynb]() for viewing which specific layers were adapted for a TAPS trained model. 
