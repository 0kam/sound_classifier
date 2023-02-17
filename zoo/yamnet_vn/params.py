SAMPLE_RATE = 16000
STFT_WINDOW_SECONDS = 0.025
STFT_HOP_SECONDS = 0.010
MEL_BANDS = 64
MEL_MIN_HZ = 500 # 125
MEL_MAX_HZ = 7500
LOG_OFFSET = 0.001
PATCH_WINDOW_SECONDS = 0.96
PATCH_HOP_SECONDS = 0.096

PATCH_FRAMES = int(round(PATCH_WINDOW_SECONDS / STFT_HOP_SECONDS))
PATCH_BANDS = MEL_BANDS
NUM_CLASSES = 2
CLASSES = ["coot", "mallard"]
CONV_PADDING = 'same'
BATCHNORM_CENTER = True
BATCHNORM_SCALE = False
BATCHNORM_EPSILON = 1e-4
CLASSIFIER_ACTIVATION = 'sigmoid'

FEATURES_LAYER_NAME = 'my_features'
EXAMPLE_PREDICTIONS_LAYER_NAME = 'my_predictions'
BATCH_SIZE = 20
QUANTIZE = True