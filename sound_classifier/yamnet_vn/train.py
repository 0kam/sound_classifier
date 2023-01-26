from sound_classifier.yamnet import YAMNet
from sound_classifier.yamnet_vn import params
from tensorflow_addons.optimizers import RectifiedAdam
from tensorflow import optimizers as optim
from audiomentations import Compose, AirAbsorption, AddBackgroundNoise, TanhDistortion, PitchShift, AddGaussianNoise, Gain

yamnet = YAMNet("sound_classifier.yamnet_vn.params")
yamnet.load_weights(
    "sound_classifier/yamnet_google/yamnet_base.h5",
    model_base=True
)

augs = Compose([
    Gain(min_gain_in_db = -20, max_gain_in_db=0, p = 0.5),
    AirAbsorption(),
    #TanhDistortion(),
    #PitchShift(),
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
    augmentations = augs
)

train_ds = ds.set_mode("train")
val_ds = ds.set_mode("val")

train_ds.normalize = False
val_ds.normalize = False

optim_top = RectifiedAdam(learning_rate = 5e-4)
yamnet.train(30, train_ds, val_ds, optim_top, fine_tune=False, workers=18)
res1 = yamnet.evaluate(val_ds, 0.5)
optim_finetune = RectifiedAdam(learning_rate = 5e-5)
yamnet.train(30, train_ds, val_ds, optim_finetune, fine_tune = True, n_layers = 3, workers=18)
res2 = yamnet.evaluate(val_ds, 0.5)

yamnet.save_weights("sound_classifier/yamnet_vn/finetune.h5", model_base = False)
yamnet.save_weights("sound_classifier/yamnet_vn/finetune_base.h5", model_base = True)

yamnet2 = YAMNet("sound_classifier.yamnet_vn.params")
optim_scratch = RectifiedAdam(learning_rate = 1e-3)
yamnet2.train(30, train_ds, val_ds, optim_scratch, fine_tune = True, n_layers = 14, workers=18)
res3 = yamnet2.evaluate(val_ds, 0.5)