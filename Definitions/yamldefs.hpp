#pragma once

// model metadata
#define Y_MODELNAME "name"
#define Y_DATASET "dataset"
#define Y_DATASETARGS "dataset_args"
#define Y_WEIGHT "weight"
#define Y_LAYERS "layers"
#define Y_ENSEMBLE "ensemble"

// ds args
#define Y_ROTATION "rotation"
#define Y_ROT_VARIANTS "rot_variants"
#define Y_SCALE "scale"
#define Y_SCALE_VARIANTS "scale_variants"
#define Y_SAMPLES "samples"
#define Y_MANDLEDEPTH "mandledepth"
#define Y_FOURIERSERIES "fourier_series"

// training data
#define Y_EPOCHS "epochs"
#define Y_BATCHSIZE "batch_size"
#define Y_VALIDFREQ "valid_freq"
#define Y_LEARNINGRATE "learning_rate"

// layer data
#define Y_LAYERTYPE "type"
#define Y_NODES "nodes"
#define Y_DROPOUT "dropout"
#define Y_REGULARIZATION "regularization"
#define Y_ACTIVATION "activation"
#define Y_LOSS "loss"
#define Y_METRIC "metric"
#define Y_MOMENTUM "momentum"
#define Y_SKIPCONN "skipconn"
#define Y_L1_LAMBDA "l1_lambda"
#define Y_L2_LAMBDA "l2_lambda"

// optimizer data
#define Y_OPT_OPTIMIZER "optimizer"
#define Y_OPT_TYPE "type"
#define Y_OPT_REGULARIZATION "regularization"
#define Y_OPT_REGLAMBDA "reg_lambda"
#define Y_OPT_LEARNINGRATE "learning_rate"
#define Y_OPT_MOMENTUM "momentum"

// scheduler data
#define Y_SCH_LRSCHEDULE "lr_scheduler"
#define Y_SCH_SCHTYPE "type"