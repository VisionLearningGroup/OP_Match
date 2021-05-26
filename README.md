# OpenMatch
This is an PyTorch implementation of OpenMatch.
This implementation is based on Pytorch-FixMatch[here](https://github.com/kekmodel/FixMatch-pytorch).

## Usage
### Dataset Preparation
This repository needs CIFAR10, CIFAR100, or ImageNet-30 to train a model.

To fully reproduce the results in evaluation, we also need SVHN, LSUN, ImageNet
for CIFAR10, 100, and LSUN, DTD, CUB, Flowers, Caltech_256, Stanford Dogs for ImageNet-30.

To prepare the datasets above, follow [CSI](https://github.com/alinlab/CSI).

After downloading the datasets, change dataset path in line 21 in dataset/cifar.py.

The repository is currently only for CIFAR10 and CIFAR100.
We will provide the image list of ImageNet-30 after acceptance.
But, by making the image lists of labeled training, unlabeled training,
validation, and testing split, you can run the experiments on ImageNet-30.
Fill the blank of get_imagenet in that case.

### Train
Train the model by 50 labeled data per class of CIFAR-10 dataset:

```
sh run_cifar10.sh 50 save_directory
```


### Evaluation
Evaluate a model trained on cifar10

```
sh run_eval_cifar10.sh trained_model.pth
```

## Requirements
- python 3.6+
- torch 1.4
- torchvision 0.5
- tensorboard
- numpy
- tqdm
- apex (optional)

