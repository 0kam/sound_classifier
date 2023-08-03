import mlflow
from soundclassifier.models.yamnet import YAMNet
from zoo.yamnet_vn import params
from tensorflow_addons.optimizers import RectifiedAdam
from tensorflow import optimizers as optim
from audiomentations import Compose, AirAbsorption, AddBackgroundNoise, TanhDistortion, PitchShift, AddGaussianNoise, Gain

#mlflow.set_experiment("yamnet_vn_multiclass_v1")

yamnet = YAMNet("zoo.yamnet_vn.params")
path = "zoo/yamnet_google/yamnet_base.h5"
yamnet.load_weights(
    path,
    model_base=True
)

augs = None

augs = Compose([
    Gain(min_gain_in_db = -15, max_gain_in_db=0, p = 0.5),
    AirAbsorption(p=0.5),
    #TanhDistortion(),
    #PitchShift(),
    AddGaussianNoise()
])

ds = yamnet.dataset(
    "data/virtual_net_strong/sources", "data/virtual_net_strong/labels2", 
    params.CLASSES, 0.96, 0.96,
    val_ratio=0.2,
    batch_size = params.BATCH_SIZE,
    patch_hop = params.PATCH_HOP_SECONDS,
    patch_sec=params.PATCH_WINDOW_SECONDS,
    threshold = 0,
    augmentations = augs,
    normalize=False
)

train_ds = ds.set_mode("train")
val_ds = ds.set_mode("val")

# 徐々に深い層まで学習
optim_top = RectifiedAdam(learning_rate = 1e-3)
yamnet.train(5, train_ds, val_ds, optim_top, fine_tune=False, workers=12)
yamnet.evaluate(val_ds, 0.5, fig_path="zoo/yamnet_vn/figs/transfer_multiclass.png")

for i in range(7):
    optim_finetune = RectifiedAdam(learning_rate = 5e-5)
    yamnet.train(30, train_ds, val_ds, optim_finetune, fine_tune=True, n_layers=(i + 1) * 2, workers=12, early_stopping=6, reduce_lr=2)
    yamnet.evaluate(val_ds, 0.5, fig_path="zoo/yamnet_vn/figs/finetune_multiclass_{}.png".format((i + 1) * 2))
    out = "zoo/yamnet_vn/h5s/finetune_multiclass_{}.h5".format((i + 1) * 2)
    print(out)
    yamnet.save_weights(out, model_base = False)

