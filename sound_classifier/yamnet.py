from sound_classifier.sound_classifier import SoundClassifier
from sound_classifier.features import log_mel_spec, spectrogram_to_patches
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential, layers
from tensorflow_addons.optimizers import RectifiedAdam
from matplotlib import pyplot as plt
import pandas as pd
from importlib import import_module

class YAMNet(SoundClassifier):
    def __init__(self, params_path) -> None:
        self.params = import_module(params_path)
        self.classes = pd.read_csv(self.params.CLASSES, index_col="index").\
            sort_values("index")
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
        waveform = layers.Input(batch_shape=(1, None))
        spec = self.features(waveform)
        s = spec.shape
        spec = tf.reshape(spec, [-1, s[-2], s[-1]])
        # Extracting features
        h = layers.Reshape(
            (self.params.PATCH_FRAMES, self.params.PATCH_BANDS, 1),
            input_shape=(self.params.PATCH_FRAMES, self.params.PATCH_BANDS)
        )(spec)
        for (i, (layer_fun, kernel, stride, filters)) in enumerate(self._YAMNET_LAYER_DEFS):
            h = layer_fun('layer{}'.format(i + 1), kernel, stride, filters)(h)
        h = layers.GlobalAveragePooling2D()(h)
        h = tf.reshape(h, [-1, self.params.NUM_PATCH, h.shape[-1]])
        self.model_base = Model(name = "yamnet_base", inputs = waveform, outputs = h)
        self.model = Sequential([
            self.model_base,
            layers.Dense(units=self.params.NUM_CLASSES, use_bias=True),
            layers.Activation(
                name=self.params.EXAMPLE_PREDICTIONS_LAYER_NAME,
                activation=self.params.CLASSIFIER_ACTIVATION
            ),
            layers.GlobalMaxPooling1D()
        ])
        
    def features(self, waveform):
        spec = log_mel_spec(waveform, self.params.SAMPLE_RATE, self.params.STFT_WINDOW_SECONDS, \
                self.params.STFT_HOP_SECONDS, self.params.MEL_BANDS, self.params.MEL_MIN_HZ, self.params.MEL_MAX_HZ, \
                self.params.LOG_OFFSET)
        patch = spectrogram_to_patches(spec, self.params.SAMPLE_RATE, self.params.STFT_HOP_SECONDS, \
                self.params.PATCH_WINDOW_SECONDS, self.params.PATCH_HOP_SECONDS)
        return patch
    
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
    
    def train(self, epochs, train_ds, val_ds, fine_tune = False):
        if fine_tune:
            lr = 1e-3
            self.model_base.trainable = True
        else:
            lr = 1e-3
            self.model_base.trainable = False
        
        rmse = tf.keras.metrics.RootMeanSquaredError()
        self.model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=RectifiedAdam(lr),
            metrics=[rmse]
        )
        
        callback = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=3,
            restore_best_weights=True
        )
        
        history = self.model.fit(train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=callback
        )
        return history
    
    def plot(self, waveform, show = True):
        plt.figure(figsize=(10, 6))
        # Plot the waveform.
        plt.subplot(3, 1, 1)
        plt.plot(tf.squeeze(waveform))
        plt.xlim([0, len(tf.squeeze(waveform))])
        # Plot the log-mel spectrogram (returned by the model).
        plt.subplot(3, 1, 2)
        spec = log_mel_spec(waveform).numpy()
        plt.imshow(spec.T, aspect='auto', interpolation='nearest', origin='lower')
        # Plot and label the model output scores for the top-scoring classes.
        scores = self.predict(waveform).numpy()
        mean_scores = np.mean(scores, axis=0)
        plt.subplot(3, 1, 3)
        plt.imshow(scores.T, aspect='auto', interpolation='nearest', cmap='gray_r')
        # patch_padding = (PATCH_WINDOW_SECONDS / 2) / PATCH_HOP_SECONDS
        # values from the model documentation
        patch_padding = (0.025 / 2) / 0.01
        plt.xlim([-patch_padding-0.5, scores.shape[0] + patch_padding-0.5])
        # Label the top_N classes.
        yticks = range(0, self.params.NUM_CLASSES, 1)
        
        plt.yticks(yticks, [self.classes["class"][x] for x in yticks])
        _ = plt.ylim(-0.5 + np.array([self.params.NUM_CLASSES, 0]))
        
        if show:
            plt.show()