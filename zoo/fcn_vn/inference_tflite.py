from sound_classifier.core.audio_device import CustomMic
import numpy as np
from zoo.fcn_vn import params
from scipy.signal import resample
import numpy as np
import math
# from tensorflow import lite as tflite # Python 3.10 has not support tflite_runtime yet.
import tflite_runtime.interpreter as tflite
np.set_printoptions(precision=2, suppress=True)

mic = CustomMic(0.96, "Analog")

th = 0.5
n_th = int(0.96 * 240 * 0.1) # 0.1秒以上Positiveなら反応する

# Set interpreter
interpreter = tflite.Interpreter(model_path = "zoo/fcn_vn/fcn.tflite")
interpreter.allocate_tensors()

# Get model I/O
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Run inference

while True:
    waveform = mic.q.get()
    waveform = resample(waveform, math.floor(
            waveform.shape[0] / mic.sampling_rate * params.SAMPLE_RATE
    ))
    waveform = waveform.astype(np.float32) / 32767
    waveform = np.expand_dims(waveform, 0)
    interpreter.set_tensor(input_details[0]['index'], waveform)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    positives = []
    for i in range(params.NUM_CLASSES):
        r = output_data[:,:,i]
        n_pos = r[r >= th].shape[0]
        if n_pos >= n_th:
            positives.append(i)
    print(positives)