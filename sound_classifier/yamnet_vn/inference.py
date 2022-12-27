import tensorflow as tf
import tensorflow_io as tfio
from sound_classifier.data import load_audio
from sound_classifier.yamnet import YAMNet
from sound_classifier.yamnet_vn import params
from matplotlib import pyplot as plt
import numpy as np

yamnet = YAMNet("sound_classifier.yamnet_vn.params")
yamnet.load_weights("./sound_classifier/yamnet_vn/finetune.h5")

n_sample = params.SAMPLE_RATE*params.AUDIO_SEC
waveform = load_audio("data/virtual_net/source/signals/mallard/JAtamura1_220226_vn_IMAG0498_s0_e5_p1_magamo.wav", params.SAMPLE_RATE)
n_patch = int(np.floor(waveform.shape[0] / n_sample))
waveform = waveform[:n_patch * n_sample]
waveform = tf.reshape(waveform, [n_patch, n_sample])
yamnet.plot(waveform)

from sound_classifier.audio_device import USBMic
mic = USBMic(3)

while True:
    #plt.ion()
    waveform = mic.q.get()
    waveform = tf.convert_to_tensor(waveform.astype(np.float32) / tf.int16.max)
    waveform = tfio.audio.resample(waveform, mic.sampling_rate, params.SAMPLE_RATE)
    waveform = tf.expand_dims(waveform, 0)
    print(yamnet.predict(waveform))
    #yamnet.plot(waveform, False)
    #plt.pause(2)
    #plt.close()