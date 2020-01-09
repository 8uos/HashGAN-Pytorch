# HashGAN-Pytorch

## Introduction

This is a pytorch implementation of [***Unsupervised Deep Generative Adversarial Hashing Network, CVPR'18***](http://openaccess.thecvf.com/content_cvpr_2018/papers/Dizaji_Unsupervised_Deep_Generative_CVPR_2018_paper.pdf) for CIFAR10 and MNIST dataset.

* * *

## Prerequisites

* **Linux**

  This code was written to be run on Linux.
* **Python > 3.6**

  Using conda is recommended: [https://docs.anaconda.com/anaconda/install/linux/](https://docs.anaconda.com/anaconda/install/linux/)
* **pytorch**

  To install: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
* **numpy** 

  Run this in terminal: `conda install numpy`

* **tqdm**

  Run this in terminal: `conda install tqdm`

* * *

## Installation

Run this in terminal:

`git clone https://github.com/8uos/HashGAN-pytorch`

* * *

## Usage
### Train
The simplest way is just run this in terminal:

`python HashGAN.py --train`

There are some possible additional arguments:

  `--batch_size`: The size of minibatch. Default value is 100.
  
  `--z_dim`     : The length of a continuous part of input random variable. Default value is 128.
  
  `--b_dim`     : The length of a binary part of input random variable. Default value is 16.
  
  `--init_lr`   : Initial learning rate. Default value is 9e-04.
  
  `--final_lr`  : Final learning rate. Default value is 3e-04.
  
  `--epochs`    : The number of epochs to train. Default value is 100.
  
  `--gpu`       : The id of gpu to use. Default value is 0.
  
  `--dataset`   : The name of dataset to use. 'cifar10' and 'mnist' are possible, and default value is 'mnist'.
  
  `--G_dict`    : The path to generator model to load. If this is None, generator will be randomly initialized. Default value is None.
  
  `--D_dict`    : The path to discriminator model to load. If this is None, discriminator will be randomly initialized. Default value is None.
  
  `--save_dir`  : The name of the save directory. Everything will be saved in `results/save_dir`. Default value is 'temp'.
  
  `--log_step`  : Interval to print the losses.
  
  `--data_dir`  : The location of the dataset. If the dataset does not exists in data_dir, the dataset will be downloaded. Default value is './data'.
  
### Evaluate

```
python HashGAN.py --eval \
                  --G_dict=Path/to/generator/dict/to/evaluate \
                  --D_dict=Path/to/discriminator/dict/to/evaluate
```

The possible additional arguments are identical to the ones above.

* * *

## Structure
### HashGAN.py
* HashGAN ***(class)***

    * __init__ ***(method)***

        Defines the HashGAN network.
    
    * loss_D ***(method)***

        Computes the dicriminator and encoder losses and stores them as attributes of HashGAN class.

    * loss_G ***(method)***

        Computes the feature matching loss and stores it as an attribute of HashGAN class.

    * step_opt ***(method)***

        Computes gradient and step optimizer.

    * generate_code_label ***(method)***

        Generates codes using current encoder and labels of all datapoints in given dataloader.

    * eval ***(method)***

        Evaluates the hashgan network with given query set and database set.

    * train ***(method)***

        Train the network.

* Define the net, train, evaluate

### utils.py
* get_trmat ***(function)***

    Builds transform matrix to compute consistent bit loss.

* set_input ***(function)***

    Makes input random variable consisting of a continuous part and a binary part of input random variable.

* get_prec_topn ***(function)***

    Computes precision@topn with given query and database codes.

* bit_entropy ***(function)***

    Computes the entropy of each bit of the given code.

### models.py
* Generator ***(class)***

    The definition of the model of generator network.

* Discriminator ***(class)***

    The definition of the model of discriminator and encoder network.

  


