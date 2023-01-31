from sound_classifier.yamnet import YAMNet
from sound_classifier.audio_device import CustomMic, USBMic
import numpy as np
from sound_classifier.yamnet_vn import params
np.set_printoptions(precision=2, suppress=True)

yamnet = YAMNet("sound_classifier.yamnet_vn.params")
yamnet.load_weights("sound_classifier/yamnet_vn/finetune.h5")

mic = CustomMic(0.96, "USB")

th = 0.75
labels = ["coot", "mallard"]
while True:
    res = yamnet.mic_inference(mic, normalize=False).numpy()
    positives = {}
    for i in range(params.NUM_CLASSES):
        if res[:,i] >= th:
            positives[labels[i]] = res[:,i]
    #print(res)
    print(positives)