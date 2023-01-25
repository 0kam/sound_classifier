from sound_classifier.yamnet import YAMNet
import numpy as np
from sound_classifier.audio_device import CustomMic

yamnet = YAMNet("sound_classifier.yamnet_vn.params")
yamnet.load_weights("sound_classifier/yamnet_vn/finetune.h5")

mic = CustomMic(0.96, "Analog")

while True:
    yamnet.mic_inference(mic)