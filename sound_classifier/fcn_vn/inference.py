from sound_classifier.fcn import FCN
from sound_classifier.audio_device import CustomMic

fcn = FCN("sound_classifier.fcn_vn.params")
fcn.load_weights("./sound_classifier/fcn_vn/finetune.h5")

mic = CustomMic(0.96, "Analog")

while True:
    fcn.mic_inference(mic)