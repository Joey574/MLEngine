## MLEngine

MLEngine is a C++ command line tool for linux aimed to allow easy iteration on different neural network designs

#### In the works
* Dropout
* Adding/Removing layers through iterations

#### Future ideas
* GPU compatability
* weight decay
* CNN
* better install

### Install

To compile the project from source, simply download and extract the repo, then, in the project directory, run

```
bash build.sh
```

which will compile the program in release mode for you, currently does not add the program to a user's PATH, but I'll probably change that someday

### Description

Written from scratch, MLEngine makes use of its own forward and back prop algorithms, currently does not support GPUs but that might change. Highly optimized and customizable, a user can define what dataset to train on, the name of the model, model dimensions, activations, weight initilization, loss, scoring, and various other options that are used during training.
<br><br>
Much of the mathematical code has been lifted from a [previous project](https://github.com/Joey574/MachineLearningCpp) of mine, specifically, the core is very similair to the *SingleBlockNeuralNetwork* version. That proejct was much more focused on just getting the math right, it allowed me to form an understanding of how neural networks worked but was by no means easy to use. This project aims to change that, primarily by making it a command line tool and allowing easy iterations of different neural network designs.

### How to use

### How it works
