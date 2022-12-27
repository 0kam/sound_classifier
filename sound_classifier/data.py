import tensorflow_io as tfio
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.utils import Sequence
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
import copy

def load_audio(path:str, rate:int = 16000):
        """
        Loading and resampling an audio file.
        Parameters
        ----------
        path : str
            A path to the audio file.
        rate : int, default 16000
            The output audio will be resampled to this rate.
        
        Returns
        -------
        waveform : tf.Tensor
            A tf.float32 tensor of the output waveform.
        """
        audio = tfio.audio.AudioIOTensor(path)
        waveform = audio.to_tensor()
        if audio.dtype == tf.int16:
            waveform = waveform / tf.int16.max
        waveform = tfio.audio.resample(waveform, audio.rate.numpy(), rate)
        return waveform[:,0]

class AudioSequence(Sequence):
    def __init__(self, data_dir:str, batch_size:int, rate:int = 16000, shuffle:bool = True, val_ratio=0):
        self.data_dir = Path(data_dir)
        self.rate = rate
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._check_dir()
        self.files = np.array(list(self.data_dir.glob("**/*.*")))
        if shuffle:
            random.shuffle(self.files)
        self.multi_hots = []
        self.labels = []
        for f in self.files:
            o, l = self.__getlabel__(f)
            self.multi_hots.append(o)
            self.labels.append(l)
        self.multi_hots = np.stack(self.multi_hots)
        self.labels = np.stack(self.labels)
        self.val_ratio = val_ratio
        
        index = range(len(self.files))
        if val_ratio > 0:
            self.train_index, self.val_index = train_test_split(index, test_size = val_ratio, stratify = self.labels)
        else:
            self.train_index = index
            self.val_index = []
        self.mode = "train"
        self.index = self.train_index
        
    def set_mode(self, mode:str):
        """
        Changing training/validation mode.
        Parameters
        ----------
        mode : str
            "train" or "val"
        """
        if mode == "train":
            self.mode = "train"
            self.index = self.train_index
        elif mode == "val":
            self.mode = "val"
            self.index = self.val_index
        else:
            raise ValueError("'mode' can only take 'train' or 'val'.")
        return copy.deepcopy(self)
    
    def _check_dir(self):
        if not (os.path.exists(str(self.data_dir) + "/others") & \
            os.path.exists(str(self.data_dir) + "/signals")):
            raise ValueError("The data_dir must have 'others' and 'signals' directory!")
        signals = list(Path(str(self.data_dir) + "/signals").iterdir())
        self.class_to_idx = {}
        self.classes = []
        for i, s in enumerate(signals):
            self.class_to_idx[s.name] = i
            self.classes.append(s.name)
        self.class_to_idx["others"] = None
        print("Classes: " + str(self.class_to_idx))

    def __len__(self):
        return int(np.floor(len(self.index) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_file = self.files[self.index][idx * self.batch_size:(idx + 1) * self.batch_size]
        audio = tf.stack([load_audio(str(f), self.rate) for f in batch_file])
        label = tf.convert_to_tensor(self.multi_hots[self.index][idx * self.batch_size:(idx + 1) * self.batch_size])
        return audio, label
    
    def __getlabel__(self, f):
        multi_hot = np.zeros(len(self.classes))
        idx = self.class_to_idx[f.parent.name]
        if idx is not None:
            multi_hot[idx] = 1
        else:
            idx = len(self.classes)
        return multi_hot, idx
