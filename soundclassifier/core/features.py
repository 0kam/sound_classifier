from kapre.time_frequency_tflite import STFTTflite, MagnitudeTflite
from kapre.time_frequency import STFT
from keras.models import Model
import tensorflow as tf
import numpy as np

class LogMelSpectrogram(Model):
    def __init__(self, rate:int=16000,
        stft_win_sec:float=0.025, stft_hop_sec:float=0.010,
        mel_bands:int=64, mel_min_hz:int=125, mel_max_hz:int=7500,
        log_offset:float=0.001, pad_begin:bool=False, pad_end:bool=True,
        tflite=False):
        """
        Keras model to calculate Log Mel Spectrogram
        Parameters
        ----------
        rate : int default 16000
        stft_win_sec : float default 0.025
        stft_hop_sec : float default 0.0010
        mel_bands : int default 64
        mel_min_hz : int default 125
        mel_max_hz : int default 7500
        log_offset : float default 0.001
        pad_begin : bool default False
        pad_end : bool default True
        tflite : bool default False
            If True, use TensorFlow Lite-compatible stft layer (kapre.time_frequency_tflite.STFTTflite).
            kapre's STFTTflite layer is restricted to a batch size of one and not available in training.
            DO NOT use this option exept when converting a trained model to a tflite model.
        """
        super().__init__(name = "mel_spectrogram")

        # Calculate parameters
        self.window_length_samples = int(round(rate * stft_win_sec))
        self.hop_length_samples = int(round(rate * stft_hop_sec))
        self.fft_length = 2 ** int(np.ceil(np.log(self.window_length_samples) / np.log(2.0)))
        self.num_spectrogram_bins = self.fft_length // 2 + 1
        self.log_offset = log_offset

        # Create layers
        if tflite:
            self.stft = STFTTflite(
                n_fft=self.fft_length, win_length=self.window_length_samples,
                hop_length=self.hop_length_samples, pad_begin=pad_begin, pad_end=pad_end
            )
            self.magnitude = lambda x: tf.norm(x, ord='euclidean', axis=-1)
        else:
            self.stft = STFT(
                n_fft=self.fft_length, win_length=self.window_length_samples,
                hop_length=self.hop_length_samples, pad_begin=pad_begin, pad_end=pad_end
            )
            self.magnitude = lambda x: tf.abs(x)

        # magnitude_spectrogram has shape [stft_frames, num_spectrogram_bins]
        # Convert spectrogram into log mel spectrogram.
        self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins = mel_bands,
            num_spectrogram_bins = self.num_spectrogram_bins,
            sample_rate = rate,
            lower_edge_hertz = mel_min_hz,
            upper_edge_hertz = mel_max_hz
        )
    
    def call(self, input_tensor, training=False):
        x = tf.expand_dims(input_tensor, 2)
        x = self.stft(x)
        x = self.magnitude(x)
        x = tf.squeeze(x, axis = -1) # Squeeze the channel dimension
        x = tf.matmul(x, self.linear_to_mel_weight_matrix)
        x = tf.math.log(x + self.log_offset)
        return x
