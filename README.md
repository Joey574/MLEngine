## MLEngine

MLEngine is a C++ command line tool for linux aimed to allow easy iteration on different neural network designs

#### In the works
* Dropout
* Adding/Removing layers through iterations

#### Future ideas
* GPU compatability
* weight decay 

### Description

Written from scratch, MLEngine makes use of its own forward and back prop algorithms, currently does not support GPUs but that might change. Highly optimized and customizable, a user can define what dataset to train on, the name of the model, model dimensions, activations, weight initilization, loss, scoring, and various other options that are used during training.
<br><br>
Much of the mathematical code has been lifted from a [previous project](https://github.com/Joey574/MachineLearningCpp) of mine.
