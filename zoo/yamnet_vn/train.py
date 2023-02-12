from sound_classifier.models.yamnet import YAMNet
from zoo.yamnet_vn import params
from tensorflow_addons.optimizers import RectifiedAdam
from tensorflow import optimizers as optim
from audiomentations import Compose, AirAbsorption, AddBackgroundNoise, TanhDistortion, PitchShift, AddGaussianNoise, Gain

yamnet = YAMNet("zoo.yamnet_vn.params")
path = "zoo/yamnet_google/yamnet_base.h5"
yamnet.load_weights(
    path,
    model_base=True
)

augs = None

augs = Compose([
    Gain(min_gain_in_db = -10, max_gain_in_db=5, p = 0.5),
    AirAbsorption(),
    TanhDistortion(),
    PitchShift(),
    AddGaussianNoise()
])

ds = yamnet.dataset(
    "data/virtual_net_strong/source", "data/virtual_net_strong/labels", 
    ["coot", "mallard"], 0.96, 0.96,
    val_ratio=0.2,
    batch_size = params.BATCH_SIZE,
    patch_hop = params.PATCH_HOP_SECONDS,
    patch_sec=params.PATCH_WINDOW_SECONDS,
    threshold = 0,
    augmentations = augs,
)

train_ds = ds.set_mode("train")
val_ds = ds.set_mode("val")

#yamnet.model = yamnet.quantize_model(yamnet.model)

optim_top = RectifiedAdam(learning_rate = 5e-4)
yamnet.train(20, train_ds, val_ds, optim_top, fine_tune=False, workers=6)
res1 = yamnet.evaluate(val_ds, 0.5)
yamnet.save_weights("zoo/yamnet_vn/transfer.h5", model_base = False)

# 徐々に深い層まで学習
optim_top = RectifiedAdam(learning_rate = 1e-3)
yamnet.train(10, train_ds, val_ds, optim_top, fine_tune=False, workers=12)

for i in range(6):
    optim_finetune = RectifiedAdam(learning_rate = 5e-5)
    if i == 6:
        epoch = 10
    else:
        epoch = 5
    yamnet.train(5, train_ds, val_ds, optim_finetune, fine_tune=True, n_layers=(i + 1) * 2, workers=6)
    yamnet.evaluate(val_ds, 0.5)

yamnet.save_weights("zoo/yamnet_vn/finetune.h5", model_base = False)

import tensorflow_model_optimization as tfmot
with tfmot.quantization.keras.quantize_scope():
    yamnet.save_weights("zoo/yamnet_vn/transfer_quantized.h5", model_base = False)

yamnet2 = YAMNet("zoo.yamnet_vn.params")
yamnet2.model = yamnet2.quantize_model(yamnet2.model)
yamnet2.model.build(input_shape=(None, 15360))
with tfmot.quantization.keras.quantize_scope():
    yamnet.load_weights("zoo/yamnet_vn/transfer_quantized.h5")

optim_finetune = RectifiedAdam(learning_rate = 5e-5)
yamnet.train(30, train_ds, val_ds, optim_finetune, fine_tune = True, n_layers = 3, workers=6)
res2 = yamnet.evaluate(val_ds, 0.5)
yamnet.save_weights("zoo/yamnet_vn/finetune.h5", model_base = False)
yamnet.save_weights("zoo/yamnet_vn/finetune_base.h5", model_base = True)

yamnet2 = YAMNet("zoo.yamnet_vn.params")
optim_scratch = RectifiedAdam(learning_rate = 1e-3)
yamnet2.train(30, train_ds, val_ds, optim_scratch, fine_tune = True, n_layers = 14, workers=18)
res3 = yamnet2.evaluate(val_ds, 0.5)
yamnet.save_weights("zoo/yamnet_vn/scratch.h5", model_base = False)
yamnet.save_weights("zoo/yamnet_vn/scratch.h5", model_base = True)


optim_finetune = RectifiedAdam(learning_rate = 5e-5)
yamnet.train(30, train_ds, val_ds, optim_finetune, fine_tune = True, n_layers = 14, workers=6)
res2 = yamnet.evaluate(val_ds, 0.5)
yamnet.save_weights("zoo/yamnet_vn/finetune_all.h5", model_base = False)
yamnet.save_weights("zoo/yamnet_vn/finetune_all_base.h5", model_base = True)