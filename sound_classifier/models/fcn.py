from sound_classifier.core.sound_classifier import SoundClassifier
from sound_classifier.core.features import LogMelSpectrogram
from keras import Model, Sequential, layers
import tensorflow as tf
import math

class FCN(SoundClassifier):
    def __init__(self, params_path) -> None:
        super().__init__(params_path)
        # Loading audio and converting it to Log-Mel Spectrogram
        input_shape = math.floor(self.params.SAMPLE_RATE * self.params.PATCH_WINDOW_SECONDS)
        waveform = layers.Input(shape = input_shape)
        self.feature_extraction = LogMelSpectrogram(
            rate=self.params.SAMPLE_RATE, stft_win_sec=self.params.STFT_WINDOW_SECONDS,
            stft_hop_sec=self.params.STFT_HOP_SECONDS, mel_bands=self.params.MEL_BANDS,
            mel_min_hz=self.params.MEL_MIN_HZ, mel_max_hz=self.params.MEL_MAX_HZ, \
            log_offset=self.params.LOG_OFFSET, pad_begin=False, pad_end=True
        )
        spec = self.feature_extraction(waveform)
        # Extracting features
        self.model_base = Sequential([
            layers.Input(spec.shape[1:]),
            layers.Reshape(
                (self.params.PATCH_FRAMES, self.params.PATCH_BANDS, 1),
                input_shape=(self.params.PATCH_FRAMES, self.params.PATCH_BANDS)
            ),
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
        self.model.build((None, input_shape))
        self.history = None

    def train(self, epochs, train_ds, val_ds, optimizer, fine_tune, reduce_method=tf.reduce_max, workers=0):
        return super().train(epochs, train_ds, val_ds, optimizer=optimizer, \
            fine_tune = fine_tune, idx = 0, reduce_method = reduce_method, reduce_axis=1, workers=workers)
    
    def evaluate(self, dataset, threshold=0.5, reduce_method=tf.reduce_max):
        return super().evaluate(dataset, threshold, reduce_method=reduce_method, reduce_axis=1)