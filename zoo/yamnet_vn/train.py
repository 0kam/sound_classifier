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
    normalize=True
)

train_ds = ds.set_mode("train")
val_ds = ds.set_mode("val")

#yamnet.model = yamnet.quantize_model(yamnet.model)

optim_top = RectifiedAdam(learning_rate = 5e-4)
yamnet.train(10, train_ds, val_ds, optim_top, fine_tune=False, workers=6)
res1 = yamnet.evaluate(val_ds, 0.5)
yamnet.save_weights("zoo/yamnet_vn/transfer_normalized.h5", model_base = False)

# 徐々に深い層まで学習
optim_top = RectifiedAdam(learning_rate = 1e-3)
yamnet.train(10, train_ds, val_ds, optim_top, fine_tune=False, workers=12)

for i in range(3):
    optim_finetune = RectifiedAdam(learning_rate = 5e-5)
    yamnet.train(5, train_ds, val_ds, optim_finetune, fine_tune=True, n_layers=(i + 1) * 2, workers=6)
    yamnet.evaluate(val_ds, 0.5)

yamnet.save_weights("zoo/yamnet_vn/finetune_normalized.h5", model_base = False)