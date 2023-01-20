from sound_classifier.yamnet import YAMNet
from sound_classifier.data import StrongAudioSequence
from sound_classifier.yamnet_vn import params

yamnet = YAMNet("sound_classifier.yamnet_vn.params")
yamnet.load_weights(
    "sound_classifier/yamnet_google/yamnet_base.h5",
    model_base=True
)

ds = yamnet.dataset(
    "data/virtual_net_strong/source", "data/virtual_net_strong/labels", 
    ["coot", "mallard", "otherbirds"], 0.96, 0.48,
    val_ratio=0.2,
    batch_size = params.BATCH_SIZE,
    patch_hop = params.PATCH_HOP_SECONDS,
    patch_sec=params.PATCH_WINDOW_SECONDS
)

train_ds = ds.set_mode("train")
val_ds = ds.set_mode("val")

yamnet.train(20, train_ds, val_ds, fine_tune = False)
yamnet.evaluate(val_ds)
yamnet.train(5, train_ds, val_ds, fine_tune = True)
yamnet.evaluate(val_ds)

from sound_classifier.audio_device import CustomMic
import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
mic = CustomMic(0.96, "Analog")

yamnet.save("sound_classifier/yamnet_vn/finetune.h5", model_base = False)
yamnet.save("sound_classifier/yamnet_vn/finetune_base.h5", model_base = True)

while True:
    waveform = mic.q.get()
    waveform = tf.convert_to_tensor(waveform.astype(np.float32) / tf.int16.max)
    waveform = tfio.audio.resample(waveform, mic.sampling_rate, params.SAMPLE_RATE)
    waveform = tf.expand_dims(waveform, 0)
    print(yamnet.predict(waveform))