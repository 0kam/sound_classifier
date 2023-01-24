from sound_classifier.sound_classifier import SoundClassifier
from sound_classifier.features import log_mel_spec
from tensorflow.keras import Model, Sequential, layers
import math

class YAMNet(SoundClassifier):
    def __init__(self, params_path) -> None:
        super().__init__(params_path)
        self._YAMNET_LAYER_DEFS = [
            # (layer_function, kernel, stride, num_filters)
            (self._conv,          [3, 3], 2,   32),
            (self._separable_conv, [3, 3], 1,   64),
            (self._separable_conv, [3, 3], 2,  128),
            (self._separable_conv, [3, 3], 1,  128),
            (self._separable_conv, [3, 3], 2,  256),
            (self._separable_conv, [3, 3], 1,  256),
            (self._separable_conv, [3, 3], 2,  512),
            (self._separable_conv, [3, 3], 1,  512),
            (self._separable_conv, [3, 3], 1,  512),
            (self._separable_conv, [3, 3], 1,  512),
            (self._separable_conv, [3, 3], 1,  512),
            (self._separable_conv, [3, 3], 1,  512),
            (self._separable_conv, [3, 3], 2, 1024),
            (self._separable_conv, [3, 3], 1, 1024)
        ]
        
        # Loading audio and converting it to Log-Mel Spectrogram
        waveform = layers.Input(shape = (math.floor(self.params.SAMPLE_RATE * self.params.PATCH_WINDOW_SECONDS)))
        spec = self.features(waveform)
        # Extracting features
        h = layers.Reshape(
            (self.params.PATCH_FRAMES, self.params.PATCH_BANDS, 1),
            input_shape=(self.params.PATCH_FRAMES, self.params.PATCH_BANDS)
        )(spec)
        for (i, (layer_fun, kernel, stride, filters)) in enumerate(self._YAMNET_LAYER_DEFS):
            h = layer_fun('layer{}'.format(i + 1), kernel, stride, filters)(h)
        #h = layers.GlobalAveragePooling2D()(h)
        h = layers.GlobalMaxPooling2D()(h)
        self.feature_extraction = Model(name = "feature_extraction", inputs = waveform, outputs = spec)
        self.model_base = Model(name = "yamnet_base", inputs = spec, outputs = h)
        self.model = Sequential([
            self.feature_extraction,
            self.model_base,
            layers.Dense(units=self.params.NUM_CLASSES, use_bias=True),
            layers.Activation(
                name=self.params.EXAMPLE_PREDICTIONS_LAYER_NAME,
                activation=self.params.CLASSIFIER_ACTIVATION
            )
        ])
        self.history = None
        
    def features(self, waveform):
        spec = log_mel_spec(waveform, self.params.SAMPLE_RATE, self.params.STFT_WINDOW_SECONDS, \
                self.params.STFT_HOP_SECONDS, self.params.MEL_BANDS, self.params.MEL_MIN_HZ, self.params.MEL_MAX_HZ, \
                self.params.LOG_OFFSET)
        return spec
    
    def _batch_norm(self, name):
        def _bn_layer(layer_input):
          return layers.BatchNormalization(
            name=name,
            center=self.params.BATCHNORM_CENTER,
            scale=self.params.BATCHNORM_SCALE,
            epsilon=self.params.BATCHNORM_EPSILON)(layer_input)
        return _bn_layer

    def _conv(self, name, kernel, stride, filters):
        def _conv_layer(layer_input):
            output = layers.Conv2D(name='{}/conv'.format(name),
                               filters=filters,
                               kernel_size=kernel,
                               strides=stride,
                               padding=self.params.CONV_PADDING,
                               use_bias=False,
                               activation=None)(layer_input)
            output = self._batch_norm(name='{}/conv/bn'.format(name))(output)
            output = layers.ReLU(name='{}/relu'.format(name))(output)
            return output
        return _conv_layer
    
    def _separable_conv(self, name, kernel, stride, filters):
        def _separable_conv_layer(layer_input):
            output = layers.DepthwiseConv2D(name='{}/depthwise_conv'.format(name),
                                    kernel_size=kernel,
                                    strides=stride,
                                    depth_multiplier=1,
                                    padding=self.params.CONV_PADDING,
                                    use_bias=False,
                                    activation=None)(layer_input)
            output = self._batch_norm(name='{}/depthwise_conv/bn'.format(name))(output)
            output = layers.ReLU(name='{}/depthwise_conv/relu'.format(name))(output)
            output = layers.Conv2D(name='{}/pointwise_conv'.format(name),
                           filters=filters,
                           kernel_size=(1, 1),
                           strides=1,
                           padding=self.params.CONV_PADDING,
                           use_bias=False,
                           activation=None)(output)
            output = self._batch_norm(name='{}/pointwise_conv/bn'.format(name))(output)
            output = layers.ReLU(name='{}/pointwise_conv/relu'.format(name))(output)
            return output
        return _separable_conv_layer
    
    def train(self, epochs, train_ds, val_ds, optimizer, fine_tune, n_layers=3, workers=0):
        return super().train(epochs, train_ds, val_ds, optimizer=optimizer, \
            fine_tune=fine_tune, idx = (n_layers * 6) + 1, reduce_method=None, reduce_axis=0, workers=workers)
    
    def evaluate(self, dataset, threshold=0.5):
        return super().evaluate(dataset, threshold, reduce_method=None, reduce_axis=0)