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

**regularization** *(string):* defines what regularization technique to use during weight/bias update *(l1, l2)*
```
regularization: l2
```
<br>

**skipconn** *(int):* defines what layer's output to append to the layer's input, ie skipconn: 0 would append the 0th layers output to this layers input
```
skipconn: 0
```
<br>

**l1_lambda** *(float):* defines the l1_lambda used if l1 regularization is defined
```
l1_lambda: 0.0001
```
<br>

**l2_lambda** *(float):* defines the l2_lambda used if l2 regularization is defined
```
l2_lambda: 0.0001
```
<br>

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

**learning_rate** *(float):* learning rate to use during training
```
learning_rate: 0.1
```
<br>

**batch_size** *(int):* batch size to use during training
```
batch_size: 512
```