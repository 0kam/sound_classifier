from abc import ABC, abstractmethod
import tensorflow as tf
from keras.models import Sequential 
from importlib import import_module
from sound_classifier.core.audio_device import AudioDevice
from scipy.signal import resample
import numpy as np
import math
from tensorflow_addons.metrics import MultiLabelConfusionMatrix
from sound_classifier.core.data import StrongAudioSequence, load_audio
import tensorflow_model_optimization as tfmot
from pathlib import Path
from tqdm import tqdm
import os

class ReducedAUC(tf.keras.metrics.Metric):
    def __init__(self, curve="PR", multi_label=True, reduce_method=tf.reduce_max, reduce_axis=1, **kwargs):
        """
        セグメンテーションデータにも対応できるAUC。
        例えば[batch, length, classes] のデータを、[batch, classes]の形にsummationしてからAUCを計算する。
        Parameters
        ----------
        curve : str default "PR"
            "ROC" か "PR"。 See https://www.tensorflow.org/api_docs/python/tf/keras/metrics/AUC
        multi_label
            See https://www.tensorflow.org/api_docs/python/tf/keras/metrics/AUC
        reduce_method : function or None default tf.reduce_max
            次元数を減らすための関数。第一引数にtf.Tensor、第二引数に減らす次元を指定するもの。
            Noneの場合はkeras.metrics.AUCと同じになる。
        reduce_axis : int default 1
            reduce_method に渡される次元のインデックス。
        """
        super().__init__(**kwargs)
        self.auc = tf.keras.metrics.AUC(curve=curve, multi_label=multi_label)
        self.reduce_method = reduce_method
        self.reduce_axis = reduce_axis

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.reduce_method is not None:
            y_true = self.reduce_method(y_true, self.reduce_axis)
            y_pred = self.reduce_method(y_pred, self.reduce_axis)
        self.auc.update_state(y_true, y_pred, sample_weight=sample_weight)

    def result(self):
        return self.auc.result()

    def reset_state(self):
        self.auc.reset_state()

class ReducedMultiLabelConfusionMatrix(tf.keras.metrics.Metric):
    def __init__(self, num_classes, reduce_method=tf.reduce_max, reduce_axis=1, **kwargs):
        """
        セグメンテーションデータにも対応できるMultilabelConfusionMatric。
        例えば[batch, length, classes] のデータを、[batch, classes]の形にsummationしてから計算する。
        Parameters
        ----------
        num_classes : int
            分類のクラス数
        reduce_method : function or None default tf.reduce_max
            次元数を減らすための関数。第一引数にtf.Tensor、第二引数に減らす次元を指定するもの。
            Noneの場合はkeras.metrics.MultiLabelConfusionMatrixと同じになる。
        reduce_axis : int default 1
            reduce_method に渡される次元のインデックス。
        """
        super().__init__(**kwargs)
        self.conf = MultiLabelConfusionMatrix(num_classes, dtype = tf.int32)
        self.reduce_method = reduce_method
        self.reduce_axis = reduce_axis

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.reduce_method is not None:
            y_true = self.reduce_method(y_true, self.reduce_axis)
            y_pred = self.reduce_method(y_pred, self.reduce_axis)
        self.conf.update_state(y_true, y_pred, sample_weight)

    def result(self):
        return self.conf.result()

    def reset_state(self):
        self.conf.reset_state()

class SoundClassifier(ABC):
    def __init__(self, params_path) -> None:
        self.params = import_module(params_path)
        self.model = self._get_model_instance(tflite=False)
        self.model.quantized = False 
        self.feature_extraction = self.model.layers[0]
        self.model_base = self.model.layers[1]
        self.model_top = self.model.layers[2]
        self.history = None
    
    @abstractmethod
    def _get_model_instance(self, tflite=False):
        pass

    def dataset(self, source_dir:str, label_dir:str, labels:list, pred_patch_sec:float, pred_hop_sec:float, \
        patch_sec:float = 3.0, patch_hop:float = 1.0, rate:int = 16000,
        batch_size:int = 10, shuffle:bool = True, val_ratio:float = 0, 
        stratify_by_dir:bool = True, normalize:bool = False,
        under_sample:bool = False,threshold:float = 0, 
        augmentations = None):
        return StrongAudioSequence(source_dir=source_dir, label_dir=label_dir, labels=labels,\
            pred_patch_sec=pred_patch_sec, pred_hop_sec=pred_hop_sec,
            patch_sec=patch_sec, patch_hop=patch_hop, rate=rate,
            batch_size=batch_size, shuffle=shuffle, val_ratio=val_ratio,
            stratify_by_dir=stratify_by_dir, normalize=normalize,
            under_sample=under_sample,
            threshold=threshold, augmentations=augmentations)
    
    def train(self, epochs, train_ds, val_ds, optimizer, fine_tune, idx, reduce_method, reduce_axis, workers=0):
        self.model.trainable = True
        if fine_tune:
            for layer in self.model_base.layers[:-idx]:
                layer.trainable = False
        else:
            self.model_base.trainable = False

        rmse = tf.keras.metrics.RootMeanSquaredError(name="metric_rmse")
        auc = ReducedAUC(reduce_method = reduce_method, reduce_axis = reduce_axis, name="metric_reduced_auc")
        self.model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(name="BCEloss"),
            optimizer=optimizer,
            metrics=[rmse, auc]
        )
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.1,
                patience=3
            )
        ]

        initial_epoch = 0
        if (self.history is not None):
            initial_epoch = self.history.epoch[-1] + 1
            epochs = initial_epoch + epochs
        
        self.history = self.model.fit(train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=callbacks,
            initial_epoch = initial_epoch,
            workers = workers
        )
    
    def evaluate(self, dataset, threshold, reduce_method, reduce_axis):
        conf = ReducedMultiLabelConfusionMatrix(num_classes=self.params.NUM_CLASSES, reduce_method=reduce_method, reduce_axis=reduce_axis, name="reduced_confusion_matrix")
        y_pred = []
        y_true = []
        for a, l in dataset:
            y_pred.append(self.predict(a))
            y_true.append(l)
        y_pred = tf.concat(y_pred, 0).numpy()
        y_true = tf.concat(y_true, 0).numpy()
        y_pred[y_pred > threshold] = 1
        y_true[y_true > threshold] = 1
        y_pred[y_pred != 1] = 0
        y_true[y_true != 1] = 0
        conf.reset_state()
        conf.update_state(y_true.astype(int), y_pred.astype(int))
        return conf.result().numpy()

    def predict(self, waveform):
        return self.model(waveform)

    def predict_from_file(self, path, rate):
        waveform = load_audio(path, rate)
        return self.model(waveform)

    def load_weights(self, path, model_base=False):
        self.model.trainable = False
        if model_base:
            self.model_base.load_weights(path)
        else:
            self.model.load_weights(path)

    def save_weights(self, path, model_base=False):
        self.model.trainable = False
        if model_base:
            self.model_base.save_weights(path)
        else:
            self.model.save_weights(path)
        
    def mic_inference(self, mic: AudioDevice, normalize:bool=False):
        """
        Predict from raw waveform
        Parameters:
        ----------
        waveform : np.array
            np.array with shape [num_samples] and dtype float32 (which means waveform takes values with in -1.0 ~ 1.0)
        """
        waveform = mic.q.get()
        waveform = resample(waveform, math.floor(waveform.shape[0] / mic.sampling_rate * self.params.SAMPLE_RATE))
        waveform = tf.convert_to_tensor(waveform.astype(np.float32) / tf.int16.max)
        #waveform = tfio.audio.resample(waveform, mic.sampling_rate, self.params.SAMPLE_RATE)
        if normalize:
            sigma = abs(waveform.numpy()).std()
            waveform = waveform / sigma
        waveform = tf.expand_dims(waveform, 0)
        res = self.predict(waveform)
        return res
    
    def quantize_model(self, model):
        if model.quantized:
            raise ValueError("The model has already been quantized!")
        feature_extraction = model.layers[0]
        model_base = tfmot.quantization.keras.quantize_model(model.layers[1])
        model_top = tfmot.quantization.keras.quantize_model(model.layers[2])
        model_q = Sequential([
            feature_extraction,
            model_base,
            model_top
        ], name = model.name + "_quantized")
        model_q.quantized = True
        input_shape = math.floor(self.params.SAMPLE_RATE * self.params.PATCH_WINDOW_SECONDS)
        model_q.build(input_shape=(None, input_shape))
        return model_q
    
    def convert_to_tflite(self):
        model = self._get_model_instance(tflite=True)
        if self.model.quantized:
            self.quantize_model()
        model.set_weights(self.model.get_weights())
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
        return tflite_model
    
    def predict_file(self, path, th = 0.5, overwrite = True, reduce_method=None, reduce_axis=1, normalize=False):
        sr = self.params.SAMPLE_RATE
        step = int(sr * self.params.PATCH_WINDOW_SECONDS)
        waveform = load_audio(path, rate=sr, normalize=normalize)
        n = math.floor(waveform.shape[0] / step)
        out_path = path.replace(Path(path).suffix, ".txt")
        if overwrite & os.path.exists(out_path):
            os.remove(out_path)
        for i in tqdm(range(n)):
            start = i * step
            end = start + step
            a = tf.expand_dims(waveform[start:end], 0)
            y = self.predict(a).numpy().squeeze()
            if reduce_method is not None:
                y = reduce_method(y, reduce_axis)
            for j in range(self.params.NUM_CLASSES):
                if y[j] >= th:
                    c = self.params.CLASSES[j]
                    s = start / sr
                    e = end / sr
                    with open(out_path, mode='a') as f:
                        f.write("{}\t{}\t{}\n".format(s, e, c))

            
        