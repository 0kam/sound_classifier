from sound_classifier.yamnet import YAMNet
from sound_classifier.audio_device import CustomMic
import numpy as np
np.set_printoptions(precision=2, suppress=True)

yamnet = YAMNet("sound_classifier.yamnet_vn.params")
yamnet.load_weights("sound_classifier/yamnet_vn/finetune.h5")

mic = CustomMic(0.96, "Analog")

while True:
    res = yamnet.mic_inference(mic, normalize=False).numpy()
    print(res)