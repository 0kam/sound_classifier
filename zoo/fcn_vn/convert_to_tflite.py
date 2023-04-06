from soundclassifier.models.fcn import FCN

fcn = FCN("zoo.fcn_vn.params")
fcn.load_weights("zoo/fcn_vn/fcn.h5")

import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_keras_model(fcn.model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
with open("zoo/fcn_vn/fcn.tflite", "wb") as f:
    f.write(tflite_model)