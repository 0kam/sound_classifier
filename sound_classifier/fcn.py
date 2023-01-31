from sound_classifier.sound_classifier import SoundClassifier
from sound_classifier.features import log_mel_spec
from keras import Model, Sequential, layers
import tensorflow as tf
import math

class FCN(SoundClassifier):
    def __init__(self, params_path) -> None:
        super().__init__(params_path)
        # Loading audio and converting it to Log-Mel Spectrogram
        waveform = layers.Input(shape = (math.floor(self.params.SAMPLE_RATE * self.params.PATCH_WINDOW_SECONDS)))
        spec = self.features(waveform)
        self.feature_extraction = Model(name = "feature_extraction", inputs = waveform, outputs = spec)
        # Extracting features
        self.model_base = Sequential([
            layers.Input(spec.shape[1:]),
            layers.Conv2D(32, (2,2), strides=(1,1), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2,2)),
            layers.Conv2D(64, (2,2), strides=(1,1), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2,2)),
            layers.Conv2D(128, (2,2), strides=(1,1), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2,2)),
            # Bottleneck
            layers.Conv2DTranspose(64, (2,1), strides=(2,1), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2DTranspose(32, (2,1), strides=(2,1), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2DTranspose(1, (2,1), strides=(2,1), padding="same", activation="relu"),
            layers.BatchNormalization()
        ], name="fcn_base")
        
        h = self.model_base(spec)

        self.model_top = Sequential([
            layers.Reshape(h.shape[1:3]),
            layers.Dense(units=self.params.NUM_CLASSES, use_bias=True),
            layers.Activation(
                name=self.params.EXAMPLE_PREDICTIONS_LAYER_NAME,
                activation=self.params.CLASSIFIER_ACTIVATION
            )
        ], name="fcn_top")

        self.model = Sequential([
            self.feature_extraction,
            self.model_base,
            self.model_top
        ])

        self.history = None
        
    def features(self, waveform):
        spec = log_mel_spec(waveform, self.params.SAMPLE_RATE, self.params.STFT_WINDOW_SECONDS, \
                self.params.STFT_HOP_SECONDS, self.params.MEL_BANDS, self.params.MEL_MIN_HZ, self.params.MEL_MAX_HZ, \
                self.params.LOG_OFFSET)
        spec = tf.expand_dims(spec, -1)
        return spec
    
    def train(self, epochs, train_ds, val_ds, optimizer, fine_tune, reduce_method=tf.reduce_max, workers=0):
        return super().train(epochs, train_ds, val_ds, optimizer=optimizer, \
            fine_tune = fine_tune, idx = 0, reduce_method = reduce_method, reduce_axis=1, workers=workers)
    
    def evaluate(self, dataset, threshold=0.5, reduce_method=tf.reduce_max):
        return super().evaluate(dataset, threshold, reduce_method=reduce_method, reduce_axis=1)