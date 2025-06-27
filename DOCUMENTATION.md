**This document provides explanations and examples of all YAML arguments that can be passed to the program**

## Metadata
**modelname** *(string):* defines modelname which is used for saving and loading over multiple runs
```
modelname: name
```

## Datasets
**dataset** *(string):* defines which dataset to train on *(mnist, mandlebrot)*
```
dataset: mnist
```
<br>

**dataset_args** *(object):* an array of configurations that defines how the dataset is loaded/modified
```
dataset_args:
    arg1: 1
    arg2: 2
```
<br>

#### MNIST Args
All following args must be passed in **dataset_args**

**rotation** *(int):* maximum roation (+-) to use when generating randomly rotated samples
```
rotation: 10
```
<br>

**rot_variants** *(int):* number of rotated samples to generate per original sample in the dataset
```
rot_variants: 5
```
<br>

**scale** *(float):* maximum scale (+-) centered around 1 to use when randomly generating scaled samples
```
scale: 0.2
```
<br>

**scale_variants** *(int):* number of scaled samples to generate per original sample in the dataset
```
scale_variants: 3
```

#### Mandlebrot Args
All following args must be passed in **dataset_args**

**samples** *(int):* number of sample points to generate
```
samples: 200000
```
<br>

**mandledepth** *(int):* maximum number of iterations when calculating if a point is in the mandlebrot or not
```
mandledepth: 500
```
<br>

**fourier_series** *(int):* number of fourier orders to compute as additional features for the dataset
```
fourier_series: 32
```
## Layers
**layers** *(object):* defines the networks layers structure
```
layers:
    - arg1: 1
    - arg2: 2
```
<br>

#### Layer Args
All following args must be passed in **layers**

**type** *(string):* defines what type of layer this is *(input, hidden, output, conv2d, conv3d)*
```
type: input
```
<br>

**nodes** *(int):* defines the number of nodes in that layer, input layer doesn't take nodes as an argument
```
nodes: 256
```
<br>

**activation** *(string):* defines what activation and derivative function to use *(linear, sigmoid, relu, leakyrelu, elu, softmax)*

```
activation: sigmoid
```
<br>

**loss** *(string):* defines what loss metric to use during backprop, only applies to output layer *(mae, mse, onehot)*
```
loss: onehot
```
<br>

**metric** *(string):* defines what scoring function to use when validating model, only applies to output layer *(mae, mse, accuracy)*
```
accuracy: mae
```
<br>

**momentum** *(float):* defines what momentum coef to use for the layer, if momentum is not defined, it's not used
```
momentum: 0.9
```
<br>

**dropout** *(float):* defines what dropout rate to use during training for the layer, if dropout is not defined, it's not used
```
dropout: 0.2
```
<br>

**skipconn** *(int):* defines what layer's output to append to the layer's input, ie skipconn: 0 would append the 0th layers output to this layers input
```
skipconn: 0
```
<br>

## Optimizer
**Optimizer** *(object):* defines what optimizer to use
```
optimizer:
  arg_1: 0
  arg_2: 1
```
<br>

#### Optimizer Args
All following args must be passed in **optimizer**

**learning_rate** *(float):* learning rate to use during training
```
learning_rate: 0.1
```
<br>

**type** *(string):* type of optimizer to use *(sgd, momuntumsgd, rmsprop, adam)*
```
type: rmsprop
```

#### SGD Args
All following args only used if **SGD** is defined

**regularization** *(string):* defines what regularization tehcnique to use, if not defined, a basic update rule is used *(l1, l2)*
```
regularization: l2
```
<br>

**reg_lambda** *(float):* defines the l1/l2 lambda value used if regularization is defined
```
reg_lambda: 0.0001
```
<br>

#### MomentumSGD Args
All following args only used if **MomentumSGD** is defined *(momentum can also use all args already described in SGD)*

**momentum** *(float):* defines the momentum to use during training
```
momentum: 0.9
```

#### RMSProp Args

#### Adam Args



## Training/Init
**weight** *(string):* defines the weight initialization technique to use in model is being created not loaded *(he, xavier, normalize)*
```
weight: normalize
```
<br>

**epochs** *(int):* number of epochs to train for
```
epochs: 100
```
<br>

**valid_freq** *(int):* how often to test the network against the validation set during training, in epochs, ie. 2 = test every 2 epochs
```
valid_freq: 5
```
<br>

**batch_size** *(int):* batch size to use during training
```
batch_size: 512
```

## Examples

```YAML
# name of the model, will be used as the folder to store the model in
name: modelname

# dataset to train on
dataset: mnist

# dataset args, we're defining additional ways we want to load mnist
# in this case we're defining what techniques we want to use to expand the training data
# rotation is the max value to rotate images by in either direction, which is randomly taken
# rot variant times and appended to the dataset
# scale is like roatation, as it defines the range +- of 1.0 scale to modify images with
# scale variants doing the same thing as rot variants
dataset_args:
  rotation: 15
  rot_variants: 5
  scale: 0.2
  scale_variants: 2

# weight initilization technique to use
weight: he

layers:
  # input layers are special and don't require the number of nodes to be
  # explicitly stated, can vary based on dataset args and will be set
  # at runtime, it does need to be stated however, as options like
  # batch reg, etc, would be applicable to the input layer
  - type: input

  # a hidden layer has many different options for configuration, some of which are shown below
  # most configs are optional, like dropout, and update
  # type, nodes, and activation are required however
  - type: hidden
    nodes: 128
    dropout: 0.3
    momentum: 0.1
    activation: leakyrelu
    regularization: l1
    l1_lambda: 0.0001

  - type: hidden
    nodes: 128
    dropout: 0.3
    activation: leakyrelu
    regularization: l2
    l2_lambda: 0.0002

  - type: output
    nodes: 10
    activation: sigmoid
    loss: onehot
    metric: accuracy
    momentum: 0.2

# number of epochs to train for
epochs: 1000

# how often to test the model against the validation/test dataset in epochs
valid_freq: 1

# learning rate to use during model training
learning_rate: 0.05

# batch size to use for training
batch_size: 512
```