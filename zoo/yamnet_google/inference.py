from sound_classifier.models.yamnet import YAMNet
from sound_classifier.core.audio_device import CustomMic
import pandas as pd

yamnet = YAMNet("zoo.yamnet_google.params")
yamnet.load_weights("zoo/yamnet_google/yamnet.h5")

classes = pd.read_csv("zoo/yamnet_google/classes.csv")
classes = classes.sort_values("index")

mic = CustomMic(0.96, "Analog")

while True:
    y = yamnet.mic_inference(mic).numpy().squeeze()
    classes["y"] = y
    print(classes.query("y > 0.3"))
    