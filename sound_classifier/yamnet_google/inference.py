import tensorflow as tf
import tensorflow_io as tfio
from sound_classifier.data import load_audio
from sound_classifier.yamnet_old import YAMNet
from sound_classifier.yamnet_google import params
from matplotlib import pyplot as plt
import numpy as np

yamnet = YAMNet("sound_classifier.yamnet_google.params")
yamnet.load_weights("./sound_classifier/yamnet_google/yamnet.h5")
waveform = load_audio("data/cicada/Tsukutsukuboushi_06.wav", params.SAMPLE_RATE)
yamnet.plot(waveform)
waveform = load_audio("Niiniizemi_02.wav", params.SAMPLE_RATE)
yamnet.plot(waveform)
waveform = load_audio("Aburazemi_02.wav", params.SAMPLE_RATE)
yamnet.plot(waveform)
waveform = load_audio("Higurashi_02.wav", params.SAMPLE_RATE)
yamnet.plot(waveform)
waveform = load_audio("Minminzemi_02.wav", params.SAMPLE_RATE)
yamnet.plot(waveform)
waveform = load_audio("Kumazemi_03.wav", params.SAMPLE_RATE)
yamnet.plot(waveform)

from sound_classifier.audio_device import USBMic
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