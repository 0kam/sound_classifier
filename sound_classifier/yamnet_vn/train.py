from sound_classifier.yamnet import YAMNet
import tensorflow as tf
import numpy as np
from sound_classifier.features import log_mel_spec, spectrogram_to_patches

yamnet = YAMNet("sound_classifier.yamnet_vn.params")
yamnet.load_weights(
    "sound_classifier/yamnet_google/yamnet_base.h5",
    model_base=True
)

ds = yamnet.dataset("./data/virtual_net/fragment/train", 20, val_ratio = 0.2)
train_ds = ds.set_mode("train")
val_ds = ds.set_mode("val")

test_ds = yamnet.dataset("./data/virtual_net/fragment/test", 20, shuffle=False)

a, l = train_ds.__getitem__(1)
yamnet.model(a).shape

yamnet.model.summary()

yamnet.train(20, train_ds, val_ds, fine_tune = True)
yamnet.evaluate(test_ds)

yamnet.save("sound_classifier/yamnet_vn/finetune.h5", model_base = False)
yamnet.save("sound_classifier/yamnet_vn/finetune_base.h5", model_base = True)