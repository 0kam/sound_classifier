import tensorflow_io as tfio
import tensorflow as tf
from scipy.io import wavfile
from pathlib import Path
from glob import glob
from keras.utils import Sequence
import numpy as np
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
import copy
import pandas as pd
import math

def load_audio(path:str, rate:int = 16000, normalize = False):
        """
        Loading and resampling an audio file.
        Parameters
        ----------
        path : str
            A path to the audio file.
        rate : int, default 16000
            The output audio will be resampled to this rate.
        normalize : boolean, default False
            If True, normalizes the audio by deviding its STD.
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
        if normalize:
            mu = waveform.numpy().mean()
            sigma = abs(waveform.numpy()).std()
            waveform = (waveform / sigma) - mu
        return waveform[:,0]

def save_audio(waveform, path, rate):
    if waveform.dtype == tf.int16:
        waveform = tf.cast(waveform, tf.int16) /  tf.int16.max
    waveform = waveform.numpy()
    wavfile.write(path, rate, waveform)

class WeakAudioSequence(Sequence):
    def __init__(self, label_path:str, batch_size:int, rate:int = 16000, shuffle:bool = True, val_ratio=0):
        self.rate = rate
        self.batch_size = batch_size
        self.shuffle = shuffle
        df = pd.read_csv(label_path)
        self.files = np.stack(df["path"].to_list())
        if shuffle:
            random.shuffle(self.files)
        self.multi_hots = df[df.columns[df.columns != "path"]].to_numpy()
        self.multi_hots = np.stack(self.multi_hots)
        self.val_ratio = val_ratio
        
        index = range(len(self.files))
        if val_ratio > 0:
            self.train_index, self.val_index = train_test_split(index, test_size = val_ratio)
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

    def __len__(self):
        return int(np.floor(len(self.index) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_file = self.files[self.index][idx * self.batch_size:(idx + 1) * self.batch_size]
        audio = tf.stack([load_audio(str(f), self.rate) for f in batch_file])
        label = tf.convert_to_tensor(self.multi_hots[self.index][idx * self.batch_size:(idx + 1) * self.batch_size])
        return audio, label


class StrongAudioSequence(Sequence):
    def __init__(
        self, source_dir:str, label_dir:str, labels:list, pred_patch_sec:float, pred_hop_sec:float, \
        patch_sec:float, patch_hop:float = 1.0, rate:int = 16000, \
        batch_size:int = 10, shuffle:bool = True, val_ratio=0.8,
        stratify_by_dir:bool = True,
        normalize:bool = False,
        under_sample:bool = False,
        threshold:float = 0.1,
        augmentations = None
        ):
        """
        Parameters
        ----------
        stratify_by_dir : bool
            Stratify the train and tvalidation data by the subdirectory of the  source_dir
        normalize : bool
            If true, normalize the audio when loading
        under_sample : bool
            Not implemented yet
        threshold : float
            thresholdより小さい割合でしか含まれていないラベルは無視する
        augmentations : audiomentations.Compose default audiomentations.Compose
            audiomentationsパッケージで作ったデータオーギュメンテーションのワークフロー。
            labelデータは変更されないため、Shiftのような時間方向の処理は入れないこと。
        """
        self.rate = rate
        self.batch_size = batch_size
        self.labels = labels
        self.normalize = normalize
        self.under_sample = under_sample
        self.threshold = threshold
        self.augmentations = augmentations
        self.label_to_idx = {}
        for i, c in enumerate(self.labels):
            self.label_to_idx[c] = i
        self.shuffle = shuffle
        self.patch_hop = patch_hop
        self.pred_hop = math.floor(self.rate * pred_hop_sec)
        self.pred_len = math.floor(self.rate * pred_patch_sec)
        self.patch_len = int(patch_sec * self.rate)
        self.label_files = glob(label_dir + "/**/*.txt", recursive=True)
        if shuffle:
            random.shuffle(self.label_files)
        self.source_files = []
        for l in self.label_files:
            f = glob(source_dir + "/**/" + Path(l).stem + ".*", recursive=True)[0]
            self.source_files.append(f)
        index = range(len(self.label_files))
        self.label_files
        if val_ratio > 0:
            if stratify_by_dir:
                dirnames = [Path(f).parent.name for f in self.label_files]
                self.train_index, self.val_index = train_test_split(index, test_size = val_ratio, stratify=dirnames)
            else:
                self.train_index, self.val_index = train_test_split(index, test_size = val_ratio)
        else:
            self.train_index = index
            self.val_index = []
        self.audio_data = []
        self.label_data = []
    
    def load_data(self, source_path, label_path):
        assert(Path(source_path).stem == Path(label_path).stem)
        # loading an audio file
        a = load_audio(source_path, self.rate, self.normalize)
        # loading a label file
        label = pd.read_table(label_path, header=None, names=["start", "end", "label"])
        label["start"] = (label["start"] * self.rate).astype(int)
        label["end"] = (label["end"] * self.rate).astype(int)
        l = np.zeros((len(a), len(self.labels)))
        for _, s, e, c in label.itertuples():
            if c in self.labels:
                c = self.label_to_idx[c]
                l[s:e, c] = 1
        audios = []
        labels = []
        i = 0
        while True:
            start = int(i * self.patch_hop * self.rate)
            end = start + self.patch_len
            if end > len(a):
                break
            audios.append(a[start:end])
            ll = tf.signal.frame(l[start:end], frame_length=self.pred_len,\
                frame_step=self.pred_hop, axis=0, pad_end = True)
            ll = tf.reduce_mean(ll, axis = 1)
            ll = tf.cast(tf.greater(ll, self.threshold), tf.int32)
            if len(ll.shape) < 3:
                ll = tf.expand_dims(ll, 0)
            labels.append(ll)
            i += 1
        if len(audios) == 0:
            return (None, None)
        audios = tf.stack(audios)
        labels = tf.concat(labels, 0)
        return audios, labels

    def set_data(self):
        del(self.audio_data)
        del(self.label_data)
        source_files = np.array(self.source_files)[self.index]
        label_files = np.array(self.label_files)[self.index]
        self.audio_data = []
        self.label_data = []
        print("Loading data...")
        for s, l in tqdm(zip(source_files, label_files), total = len(source_files)):
            audio, label = self.load_data(s, l)
            if audio is not None:
                self.audio_data.append(audio)
                self.label_data.append(label)
        self.audio_data = tf.squeeze(tf.concat(self.audio_data, 0))
        self.label_data = tf.squeeze(tf.concat(self.label_data, 0))
        if self.shuffle:
            idx = tf.range(start=0, limit=tf.shape(self.audio_data)[0], dtype=tf.int32)
            idx = tf.random.shuffle(idx)
            self.audio_data = tf.gather(self.audio_data, idx)
            self.label_data = tf.gather(self.label_data, idx)
            
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
        self.set_data()
        return copy.deepcopy(self)

    def __len__(self):
        return math.ceil(len(self.audio_data) / float(self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        audio = self.audio_data[start:end]
        label = self.label_data[start:end]
        if (self.augmentations is not None) & (self.mode == "train"):
            audio = self.augmentations(audio.numpy(), self.rate)
            audio = tf.convert_to_tensor(audio)
        return audio, label
    
    @classmethod
    def others_label(cls, in_dir, out_dir):
        for f in glob(in_dir + "/**/*.wav", recursive=True):
            suf = Path(f).suffix
            out = f.replace(in_dir, out_dir).replace(suf, ".txt")
            Path(out).touch()

# self = StrongAudioSequence("data/virtual_net_strong/source", "data/virtual_net_strong/labels", ["coot", "mallard", "otherbirds"], 0.96, 0.48)