# [Task Adaptive Parameter Sharing for Multi-Task Learning](https://arxiv.org/abs/2203.16708)

Unofficial Pytorch implementation of **Task Adaptive Parameter Sharing** (CVPR 2022). <br />


<p align="center">
<img src="./assets/teaser.jpg" width="512"/>
</p>

Task Adaptive Parameter Sharing (TAPS) is a general method for tuning a base model to a new task by adaptively modifying a small, task-specific subset of layers. This enables multi-task learning while minimizing resources used and competition between tasks. TAPS solves a joint optimization problem which finds which layers to share with the base model and the value of the task-specific weights. TAPS is agnostic to model architecture and requires minor changes to the training scheme.


## Installation

### Requirements
Run ```pip install -r requirements.txt```.
The main packages required are pytorch, torchvision, timm, tqdm, and tensorboard.
### Datasets

**ImageNet-to-Sketch**
The 5 datasets comprising ImagetNet-to-Sketch can be download from the [PiggyBack repository](https://github.com/arunmallya/piggyback) at this link: [https://uofi.box.com/s/ixncr3d85guosajywhf7yridszzg5zsq](https://uofi.box.com/s/ixncr3d85guosajywhf7yridszzg5zsq)

**DomainNet**
The 6 DomainNet datasets can be downloaded from the [original website](http://ai.bu.edu/M3SDA/). A formatted version can be downloaded [here](https://drive.google.com/file/d/1Eowq0kHzS0MKgo1oglqJIRAC_wDlqWP9/view?usp=sharing). The structure of the folder should be the following:
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
--cropped - Flag that changes the data augmentation for already cropped datasets (include for Stanford cars and CUBS).
--model_type - Specifies the network architecture. Currently supports ResNet34, ResNet50, and ResNet101. 
                Support for VIT and all convolutional networks coming soon. 
--model_path - Relative path to a pretrained model. Default option uses the pytorch pretrained models.
```

### Sequential TAPS Training
Fine-tune a pretrained ResNet34 with TAPS on the sketch dataset with multiple gpus. 
```
python train_sequential.py --dataset ./datasets/DomainNet/sketch --experiment_name \
./results/DN_sketch --multi_gpu --model_type resnet34
```

Fine-tune a pretrained ResNet50 with TAPS on the CUBS dataset with single gpu. 
```
python train_sequential.py --dataset ./datasets/cubs_cropped --experiment_name \
./results/CUBS --model_type resnet50 --lam .1 --cropped
```



### Joint TAPS Training
To run the joint version of TAPS, first train a shared network on the 6 DomainNet datasets:
```
python train_joint.py --dataset ./datasets/DomainNet/ --experiment_name \
./results/DN_joint --multi_gpu --model_type resnet34
```

Next, load the pretrained model from the previous step and run sequential TAPS. This is the efficient variant of joint TAPS which has constant memory requirements during training. To train on the different datasets of DomainNet, change out ```--dataset ./datasets/DomainNet/infograph``` for the path to the other datasets. 
```
python train_sequential.py --dataset ./datasets/DomainNet/infograph --experiment_name \
./results/DN_sketch --multi_gpu --model_type resnet34 --model_path ./results/DN_joint/model_best.pth
```

## Evaluation

### Tensorboard
To view results run ```tensorboard --logdir=./results``` and navigate to http://localhost:6006/.

We log validation error/training loss/training error/percentage of layers tuned.


### Visualizing Modified Layers

We provide the [VisualizeLayers.ipynb](./VisualizeLayers.ipynb) for viewing which layers of a TAP trained model were adapted and calculating percentage of parameters added.
<p align="center">
<img src="./assets/heatmap.png" width="1024"/>
</p>


## Coming Soon

* Support for VIT and arbitrary convolutional networks. 


## Citation
If you found this repository useful, consider giving a 🌟 and citation:
```
@inproceedings{wallingford2022task,
  title={Task adaptive parameter sharing for multi-task learning},
  author={Wallingford, Matthew and Li, Hao and Achille, Alessandro and Ravichandran, Avinash and Fowlkes, Charless and Bhotika, Rahul and Soatto, Stefano},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7561--7570},
  year={2022}
}
```

For questions regarding this repository email mcw244 at cs dot washington dot edu
