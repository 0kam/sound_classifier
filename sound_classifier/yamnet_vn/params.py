SAMPLE_RATE = 16000
STFT_WINDOW_SECONDS = 0.025
STFT_HOP_SECONDS = 0.010
MEL_BANDS = 64
MEL_MIN_HZ = 125
MEL_MAX_HZ = 7500
LOG_OFFSET = 0.001
PATCH_WINDOW_SECONDS = 0.96
PATCH_HOP_SECONDS = 0.1

PATCH_FRAMES = int(round(PATCH_WINDOW_SECONDS / STFT_HOP_SECONDS))
PATCH_BANDS = MEL_BANDS
NUM_CLASSES = 3 # Mallard, Coot & Other Birds
CONV_PADDING = 'same'
BATCHNORM_CENTER = True
BATCHNORM_SCALE = False
BATCHNORM_EPSILON = 1e-4
CLASSIFIER_ACTIVATION = 'sigmoid'

FEATURES_LAYER_NAME = 'my_features'
EXAMPLE_PREDICTIONS_LAYER_NAME = 'my_predictions'

CLASSES = "sound_classifier/yamnet_vn/classes.csv"
PRETRAINED_WEIGHTS = "sound_classifier/yamnet_google/yamnet.h5"
BATCH_SIZE = 20