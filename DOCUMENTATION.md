**This document provides explanations and examples of all YAML arguments that can be passed to the program**

## Metadata
**modelname** *(string):* defines modelname which is used for saving and loading over multiple runs

## Datasets
**dataset** *(string):* defines which dataset to train on *(mnist, mandlebrot)*
<br>

**dataset_args** *(object):* an array of configurations that defines how the dataset is loaded/modified
<br>

#### MNIST Args
All following args must be passed in **dataset_args**

**rotation** *(int):* maximum roation (+-) to use when generating randomly rotated samples
<br>

**rot_variants** *(int):* number of rotated samples to generate per original sample in the dataset
<br>

**scale** *(float):* maximum scale (+-) centered around 1 to use when randomly generating scaled samples
<br>

**scale_variants** *(int):* number of scaled samples to generate per original sample in the dataset

#### Mandlebrot Args
All following args must be passed in **dataset_args**

**samples** *(int):* number of sample points to generate
<br>

**mandledepth** *(int):* maximum number of iterations when calculating if a point is in the mandlebrot or not
<br>

**fourier_series** *(int):* number of fourier orders to compute as additional features for the dataset

## Layers
**layers** *(object array):* defines the networks layers structure
<br>

#### Layer Args
All following args must be passed in **layers**

**type** *(string):* defines what type of layer this is *(input, hidden, output, conv2d, conv3d)*
<br>

**nodes** *(int):* defines the number of nodes in that layer, input layer doesn't take nodes as an argument
<br>

**activation** *(string):* defines what activation and derivative function to use *(linear, sigmoid, relu, leakyrelu, elu, softmax)*
<br>

**loss** *(string):* defines what loss metric to use during backprop, only applies to output layer *(mae, mse, onehot)*
<br>

**metric** *(string):* defines what scoring function to use when validating model, only applies to output layer *(mae, mse, accuracy)*
<br>

**dropout** *(float):* defines what dropout rate to use during training for the layer, if dropout is not defined, it's not used
<br>

**skipconn** *(int):* defines what layer's output to append to the layer's input, ie skipconn: 0 would append the 0th layers output to this layers input
<br>

## Optimizer
**Optimizer** *(object):* defines what optimizer to use
<br>

#### Optimizer Args
All following args must be passed in **optimizer**

**learning_rate** *(float):* learning rate to use during training
<br>

**type** *(string):* type of optimizer to use *(sgd, momuntumsgd, rmsprop, adam)*

#### SGD Args
All following args only used if **sgd** is used

**regularization** *(string):* Defines what regularization tehcnique to use, if not defined, a basic update rule is used *(l1, l2)*
<br>

**reg_lambda** *(float):* Defines the l1/l2 lambda value used if regularization is defined
<br>

## MomentumSGD Args
All following args only used if **momentumsgd** is used *(momentum can also make use of the sgd args)*

**momentum** *(float):* Defines the momentum to use during training
<br>

#### RMSProp Args
All following args only used if **rmsprop** is used

**decay** *(float):* Defines the decay value to use during training
<br>

**epsl** *(float):* Defines the epsilon value to use during training
<br>

#### Adam Args
All following args only used if **adam** is used

**b1** *(float):* Defines the B1 value to use (first moment) <br>

**b2** *(float):* Defines the B2 value to use (second moment) <br>

**epsl** *(float):* Defines the epsilon value to use during training <br>

## Training/Init
**weight** *(string):* Defines the weight initialization technique to use in model is being created not loaded *(he, xavier, normalize)*
<br>

**epochs** *(int):* Number of epochs to train for
<br>

**valid_freq** *(int):* How often to test the network against the validation set during training, in epochs, ie. 2 = test every 2 epochs
<br>

**batch_size** *(int):* Batch size to use during training

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

# optimizer and everything else update related to use
optimizer:
  learning_rate: 0.05
  type: rmsprop
  decay: 0.7

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
    activation: leakyrelu

  - type: hidden
    nodes: 128
    dropout: 0.3
    activation: leakyrelu

  - type: output
    nodes: 10
    activation: sigmoid
    loss: onehot
    metric: accuracy

# number of epochs to train for
epochs: 1000

# how often to test the model against the validation/test dataset in epochs
valid_freq: 1

# batch size to use for training
batch_size: 512
```