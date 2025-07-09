#pragma once

// model metadata
#define Y_MODELNAME "name"
#define Y_DATASET "dataset"
#define Y_DATASETARGS "dataset_args"
#define Y_WEIGHT "weight"
#define Y_LAYERS "layers"
#define Y_ENSEMBLE "ensemble"

// ds args
#define Y_AUGMENT_REF_INTERVAL "augment_refresh_interval"

#define Y_ROTATION "rotation"
#define Y_MIN_ROTATION "min_rotation"
#define Y_ROT_VARIANTS "rot_variants"

#define Y_SCALE "scale"
#define Y_MIN_SCALE "min_scale"
#define Y_SCALE_VARIANTS "scale_variants"

#define Y_SHEAR "shear"
#define Y_MIN_SHEAR "min_shear"
#define Y_SHEAR_VARIANTS "shear_variants"

#define Y_SAMPLES "samples"
#define Y_MANDLEDEPTH "mandledepth"
#define Y_FOURIERSERIES "fourier_series"

#define Y_ELASTIC_DEFORM "elastic_deform"
#define Y_ELASTIC_ALPHA "alpha"
#define Y_ELASTIC_SIGMA "sigma"
#define Y_ELASTIC_VARIANTS "variants"

// training data
#define Y_EPOCHS "epochs"
#define Y_BATCHSIZE "batch_size"
#define Y_VALIDFREQ "valid_freq"
#define Y_SEED "seed"

// layer data
#define Y_LAYERTYPE "type"
#define Y_NODES "nodes"
#define Y_DROPOUT "dropout"
#define Y_ACTIVATION "activation"
#define Y_LOSS "loss"
#define Y_METRIC "metric"
#define Y_SKIPCONN "skipconn"

// optimizer data
#define Y_OPT_OPTIMIZER "optimizer"
#define Y_OPT_TYPE "type"
#define Y_OPT_REGULARIZATION "regularization"
#define Y_OPT_REGLAMBDA "reg_lambda"
#define Y_OPT_LEARNINGRATE "learning_rate"
#define Y_OPT_MOMENTUM "momentum"
#define Y_OPT_DECAY "decay"
#define Y_OPT_EPSL "epsilon"
#define Y_OPT_B1 "b1"
#define Y_OPT_B2 "b2"

// scheduler data
#define Y_SCH_LRSCHEDULE "lr_scheduler"
#define Y_SCH_SCHTYPE "type"
