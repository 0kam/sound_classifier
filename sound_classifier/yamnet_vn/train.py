from sound_classifier.yamnet import YAMNet
from sound_classifier.yamnet_vn import params
from tensorflow_addons.optimizers import RectifiedAdam
from tensorflow import optimizers as optim
from audiomentations import Compose, AirAbsorption, AddBackgroundNoise, TanhDistortion, PitchShift, AddGaussianNoise

yamnet = YAMNet("sound_classifier.yamnet_vn.params")
yamnet.load_weights(
    "sound_classifier/yamnet_google/yamnet_base.h5",
    model_base=True
)

augs = Compose([
    AirAbsorption(p = 0.75),
    TanhDistortion(),
    PitchShift(),
    AddGaussianNoise(p = 1.0)
])

ds = yamnet.dataset(
    "data/virtual_net_strong/source", "data/virtual_net_strong/labels", 
    ["coot", "mallard"], 0.96, 0.96,
    val_ratio=0.2,
    batch_size = params.BATCH_SIZE,
    patch_hop = params.PATCH_HOP_SECONDS,
    patch_sec=params.PATCH_WINDOW_SECONDS,
    threshold = 0,
    augmentations = augs
)

train_ds = ds.set_mode("train")
val_ds = ds.set_mode("val")

optim_top = RectifiedAdam(learning_rate = 5e-4)
yamnet.train(10, train_ds, val_ds, optim_top, fine_tune=False, workers=18)
res1 = yamnet.evaluate(val_ds, 0.5)
optim_finetune = RectifiedAdam(learning_rate = 5e-5)
yamnet.train(10, train_ds, val_ds, optim_finetune, fine_tune = True, n_layers = 3, workers=0)
res2 = yamnet.evaluate(val_ds, 0.5)

yamnet.save("sound_classifier/yamnet_vn/finetune.h5", model_base = False)
yamnet.save("sound_classifier/yamnet_vn/finetune_base.h5", model_base = True)

from sound_classifier.audio_device import CustomMic
import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
mic = CustomMic(0.96, "Analog")

while True:
    waveform = mic.q.get()
    waveform = tf.convert_to_tensor(waveform.astype(np.float32) / tf.int16.max)
    waveform = tfio.audio.resample(waveform, mic.sampling_rate, params.SAMPLE_RATE)
    waveform = tf.expand_dims(waveform, 0)
    print(yamnet.predict(waveform))