from sound_classifier.core.audio_device import CustomMic
import numpy as np
from zoo.yamnet_vn import params
from scipy.signal import resample
import numpy as np
import math
# from tensorflow import lite as tflite # Python 3.10 has not support tflite_runtime yet.
import tflite_runtime.interpreter as tflite
np.set_printoptions(precision=2, suppress=True)

th = 0.75

# Set interpreter
interpreter = tflite.Interpreter(model_path = "zoo/yamnet_vn/finetune.tflite")
interpreter.allocate_tensors()

# Get model I/O
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Run inference
mic = CustomMic(0.96, "USB")
labels = ["coot", "mallard"]
while True:
    waveform = mic.q.get()
    waveform = resample(waveform, math.floor(
            waveform.shape[0] / mic.sampling_rate * params.SAMPLE_RATE
    ))
    waveform = waveform.astype(np.float32) / 32767 # tf.int16.max
    waveform = np.expand_dims(waveform, 0)
    interpreter.set_tensor(input_details[0]['index'], waveform)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    positives = {}
    for i in range(params.NUM_CLASSES):
        if output_data[:,i] >= th:
            positives[labels[i]] = output_data[:,i]
    print(positives)