# sound_classifier
Training and deploying deep sound classifiers, especially for bio-acoustic monitorings.

## Installation
```
python3 -m pip install git+https://github.com/0kam/sound_classifier
```

## Requirements
```
# For inference
tflite-runtime
or
tensorflow (tested on 2.10.0) # tflite-runtime has not support Python 3.10 yet, so use tensorflow.lite instead

# For a real-time inference using microphones
pyaudio
numpy
scipy # for audio resampling

# For training
tensorflow (tested on 2.10.0)
tensorflow-io
kapre
audiomentations
scikit-learn
tqdm
pandas
numpy
scypi
```

## Modules
- sound_classifier  
  The main component of this package
  - core  
    The core scripts for all models
    - audio_device.py  
      For sampling audio from microphones
    - data.py  
      For preparing audio datasets to train models
    - features.py  
      For extracting features (such as Log-Mel spectrograms) from audios, using `kapre` package
    - sound_classifier.py  
      An abstract based class for all classifier models
  - models  
    Audio classifiers
    - yamnet.py  
      A Google's YAMNet based audio classification model
    - fcn.py  
      A small Fully Convolutional Neural network model
- zoo  
  Implementation examples of models with sound_classifier  
  Each directory contains Python scripts for training and inferencing the model. Also, some models supports TensorFlowLite inferencing for edge and mobile applications.
  - yamnet_google (!!under construction)  
    The original YAMNet