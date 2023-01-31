from sound_classifier.models.fcn import FCN
from sound_classifier.core.audio_device import CustomMic
import numpy as np
from sound_classifier.zoo.fcn_vn import params
np.set_printoptions(precision=2, suppress=True)

fcn = FCN("sound_classifier.fcn_vn.params")
fcn.load_weights("./sound_classifier/fcn_vn/fcn.h5")

mic = CustomMic(0.96, "Analog")

th = 0.5
n_th = int(0.96 * 240 * 0.1) # 0.1秒以上Positiveなら反応する
while True:
    res = fcn.mic_inference(mic, normalize=False)
    positives = []
    for i in range(params.NUM_CLASSES):
        r = res[:,:,i]
        n_pos = r[r >= th].numpy().shape[0]
        if n_pos >= n_th:
            positives.append(i)
    print(positives)