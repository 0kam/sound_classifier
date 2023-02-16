from sound_classifier.models.yamnet import YAMNet

yamnet = YAMNet("zoo.yamnet_google.params")
yamnet.load_weights("zoo/yamnet_google/yamnet.h5")

tflite_model = yamnet.convert_to_tflite()
with open("zoo/yamnet_google/yamnet_google.tflite", "wb") as f:
    f.write(tflite_model)