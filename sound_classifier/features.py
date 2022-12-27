import numpy as np
import tensorflow as tf



def log_mel_spec(waveform:tf.Tensor, rate:int=16000, \
        stft_win_sec:float=0.025, stft_hop_sec:float=0.010, \
        mel_bands:int=64, mel_min_hz:int=125, mel_max_hz:int=7500, \
        log_offset:float=0.001):
    """
    Compute log mel spectrogram of a 1-D waveform.
    The default values are of Google's Yamnet.
    See https://github.com/tensorflow/models/tree/master/research/audioset/yamnet
    Note that tf.signal.stft() uses a periodic Hann window by default.
    Parameters
    ----------
    waveform : tf.tensor
        An audio waveform that has a shape [samples].
    rate : int, default 16000
        A sampling rate of the audio
    stft_win_sec : float, default 0.025
        A window size of STFT in seconds 
    stft_hop_sec : float, default 0.010
        A hop length of STFT in seconds
    mel_bands : int, default 64
        A number of Mel bands
    mel_min_hz : int, default 125
        A minimum frequency of the mel bands
    mel_max_hz : int default 7500
        A maximum frequency of the mel bands
    log_offset : float, default 0.001
        An offset avoid zero-divisions in calculating log value
    
    Returns
    -------
    log_mel_spectrogram : tf.Tensor
        A Log-Mel Spectrogram with shape [stft_frames, mel_bands]
    """
    with tf.name_scope('log_mel_features'):
        # calculate
        window_length_samples = int(
            round(rate * stft_win_sec))
        hop_length_samples = int(
            round(rate * stft_hop_sec))
        fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))
        num_spectrogram_bins = fft_length // 2 + 1
        magnitude_spectrogram = tf.abs(tf.signal.stft(
            signals=waveform,
            frame_length=window_length_samples,
            frame_step=hop_length_samples,
            fft_length=fft_length
        ))
        # magnitude_spectrogram has shape [stft_frames, num_spectrogram_bins]
        # Convert spectrogram into log mel spectrogram.
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins = mel_bands,
            num_spectrogram_bins = num_spectrogram_bins,
            sample_rate = rate,
            lower_edge_hertz = mel_min_hz,
            upper_edge_hertz = mel_max_hz
        )
        mel_spectrogram = tf.matmul(
            magnitude_spectrogram, linear_to_mel_weight_matrix)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + log_offset)
        return log_mel_spectrogram


def spectrogram_to_patches(spectrogram, rate:int = 16000, stft_hop_sec:float = 0.010, \
        patch_win_sec:float = 0.96, patch_hop_sec:float=0.48):
    """
    Break up a spectrogram into a stack of fixed-size patches.
    The default values are of Google's Yamnet.
    See https://github.com/tensorflow/models/tree/master/research/audioset/yamnet
    Only complete frames are emitted, so if there is less than patch_win_sec of 
    waveform then nothing is emitted (to avoid this, zero-pad before processing).
    Parameters
    ----------
    spectrogram : tf.tensor
        Spectrograms with shape [batch_size, stft_frames, mel_bands]
    rate : int, default 16000
        A sampling rate of the audio
    stft_hop_sec : float, default 0.010
        A hop length of STFT in seconds
    patch_win_sec : float, default 0.96
        A patch size in seconds.
    patch_hop_sec : float default 0.48
        A hopping length of each patch in seconds 
    
    Returns
    -------
    features : tf.Tensor
        The splitted spectrograms
    """
    with tf.name_scope('feature_patches'):
        hop_length_samples = int(
            round(rate * stft_hop_sec))
        spectrogram_sr = rate / hop_length_samples
        patch_window_length_samples = int(
            round(spectrogram_sr * patch_win_sec))
        patch_hop_length_samples = int(
            round(spectrogram_sr * patch_hop_sec))
        features = tf.signal.frame(
            signal=spectrogram,
            frame_length=patch_window_length_samples,
            frame_step=patch_hop_length_samples,
            axis=1) # axis=0 for yamnet_old.py
        
        # features has shape [batch_size, <# patches>, <# STFT frames in an patch>, MEL_BANDS]
        return features