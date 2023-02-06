from sound_classifier.models.yamnet import YAMNet

yamnet = YAMNet("zoo.yamnet_vn.params")
yamnet.load_weights("zoo/yamnet_vn/finetune.h5")

import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_keras_model(yamnet.model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
with open("zoo/yamnet_vn/finetune.tflite", "wb") as f:
    f.write(tflite_model)