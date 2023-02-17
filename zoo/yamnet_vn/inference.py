from sound_classifier.models.yamnet import YAMNet
from sound_classifier.core.audio_device import CustomMic, USBMic
import numpy as np
from zoo.yamnet_vn import params
np.set_printoptions(precision=2, suppress=True)

yamnet = YAMNet("zoo.yamnet_vn.params")
yamnet.load_weights("zoo/yamnet_vn/finetune.h5")

path = "/home/okamoto/Projects/VirtualNet/swavs_14FEB2023/merged/kijo1.wav"
yamnet.predict_file(path, normalize=True)

mic = CustomMic(0.96, "Analog")

th = 0.5
labels = ["coot", "mallard"]
while True:
    res = yamnet.mic_inference(mic, normalize=False).numpy()
    positives = {}
    for i in range(params.NUM_CLASSES):
        if res[:,i] >= th:
            positives[labels[i]] = res[:,i]
    print(res)
    print(positives)