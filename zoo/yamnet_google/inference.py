import tensorflow as tf
import tensorflow_io as tfio
from sound_classifier.models.yamnet import YAMNet
from zoo.yamnet_google import params
from matplotlib import pyplot as plt
import numpy as np

yamnet = YAMNet("zoo.yamnet_google.params")
yamnet.load_weights("zoo/yamnet_google/yamnet.h5")

from sound_classifier.core.audio_device import USBMic
mic = USBMic(5)

a = mic.q.get()
while True:
    plt.ion()
    waveform = mic.q.get()
    waveform = tf.convert_to_tensor(waveform.astype(np.float32) / tf.int16.max)
    waveform = tfio.audio.resample(waveform, mic.sampling_rate, params.SAMPLE_RATE)
    yamnet.plot(waveform, False)
    plt.pause(4)
    plt.close()