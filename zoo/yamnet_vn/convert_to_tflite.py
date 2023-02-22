from sound_classifier.models.yamnet import YAMNet

yamnet = YAMNet("zoo.yamnet_vn.params")
yamnet.load_weights("zoo/yamnet_vn/finetune_normalized.h5")

tflite_model = yamnet.convert_to_tflite()
with open("zoo/yamnet_vn/yamnet_vn.tflite", "wb") as f:
    f.write(tflite_model)

# Quantized model
yamnet = YAMNet("zoo.yamnet_vn.params")
yamnet.model = yamnet.quantize_model(yamnet.model)
yamnet.load_weights("zoo/yamnet_vn/transfer_quantized.h5")

import tensorflow as tf
tflite_model = yamnet.convert_to_tflite()
with open("zoo/yamnet_vn/finetune.tflite", "wb") as f:
    f.write(tflite_model)