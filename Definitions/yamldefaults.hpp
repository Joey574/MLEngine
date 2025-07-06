#pragma once

// model defailts
#define Y_WEIGHT_DEFAULT "none"

// ds arg defaults
#define Y_RUNNING_AUGMENT_DEFAULT false
#define Y_ROT_VAR_DEFAULT 0
#define Y_MIN_ROTATION_DEFAULT 0

#define Y_SCALE_VAR_DEFAULT 0
#define Y_MIN_SCALE_DEFAULT 0

#define Y_SHEAR_VAR_DEFAULT 0
#define Y_MIN_SHEAR_DEFAULT 0

#define Y_SAMPLE_DEFAULT 0
#define Y_MANDLEDEPTH_DEFAULT 50
#define Y_FOURIER_DEFAULT 0

#define Y_ELASTIC_ALPHA_DEFAULT 34.0
#define Y_ELASTIC_SIGMA_DEFAULT 4.0
#define Y_ELASTIC_VAR_DEFAULT 0

// training defaults
#define Y_EPOCH_DEFAULT 0
#define Y_BATCH_DEFAULT 512
#define Y_VALID_DEFAULT 0

// layer defaults
#define Y_DROPOUT_DEFAULT 0
#define Y_ACTV_DEFAULT "none"
#define Y_LOSS_DEFAULT "none"
#define Y_METRIC_DEFAULT "none"

// optimizer defaults
#define Y_OPTIMIZER_DEFAULT "sgd"
#define Y_REGLAMBDA_DEFAULT 0.0001
#define Y_LEARNRATE_DEFAULT 0.1
#define Y_MOMENTUM_DEFAULT 0.9
#define Y_DECAY_DEFAULT 0.999
#define Y_EPSL_DEFAULT 0.000001
#define Y_B1_DEFAULT 0.9
#define Y_B2_DEFAULT 0.999
