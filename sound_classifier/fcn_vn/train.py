from sound_classifier.fcn import FCN
from sound_classifier.fcn_vn import params
from tensorflow_addons.optimizers import RectifiedAdam
from tensorflow import optimizers as optim
from audiomentations import Compose, AirAbsorption, AddBackgroundNoise, TanhDistortion, PitchShift, AddGaussianNoise

fcn = FCN("sound_classifier.fcn_vn.params")

augs = Compose([
    AirAbsorption(p = 0.75),
    TanhDistortion(),
    PitchShift(),
    AddGaussianNoise(p = 1.0)
])

ds = fcn.dataset(
    "data/virtual_net_strong/source", "data/virtual_net_strong/labels", 
    labels = ["coot", "mallard"], 
    pred_patch_sec = params.STFT_WINDOW_SECONDS, pred_hop_sec = params.STFT_HOP_SECONDS,
    val_ratio=0.2,
    batch_size = params.BATCH_SIZE,
    patch_hop = params.PATCH_HOP_SECONDS,
    patch_sec=params.PATCH_WINDOW_SECONDS,
    threshold = 0.1
)

train_ds = ds.set_mode("train")
val_ds = ds.set_mode("val")

optimizer = RectifiedAdam(1e-3)
fcn.train(20, train_ds, val_ds, fine_tune=True, optimizer=optimizer, workers=4)
fcn.evaluate(val_ds, 0.75)

from sound_classifier.audio_device import CustomMic
import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
mic = CustomMic(params.PATCH_WINDOW_SECONDS, "Analog")

fcn.save("sound_classifier/fcn_vn/fcn.h5", model_base = False)
fcn.save("sound_classifier/fcn_vn/fcn_base.h5", model_base = True)

while True:
    waveform = mic.q.get()
    waveform = tf.convert_to_tensor(waveform.astype(np.float32) / tf.int16.max)
    waveform = tfio.audio.resample(waveform, mic.sampling_rate, params.SAMPLE_RATE)
    waveform = tf.expand_dims(waveform, 0)
    print(tf.reduce_mean(fcn.predict(waveform), 1))
    print("----------")