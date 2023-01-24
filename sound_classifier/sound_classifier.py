from abc import ABC, abstractmethod
from sound_classifier.data import StrongAudioSequence, load_audio
import tensorflow as tf
from tensorflow_addons.metrics import MultiLabelConfusionMatrix
from importlib import import_module
from audiomentations import Compose

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
    
    @abstractmethod
    def features(self, waveform):
        """
        Converting a waveform to audio features, such as spectrogram.
        """
        pass

    def dataset(self, source_dir:str, label_dir:str, labels:list, pred_patch_sec:float, pred_hop_sec:float, \
        patch_sec:float = 3.0, patch_hop:float = 1.0, rate:int = 16000, \
        batch_size:int = 10, shuffle:bool = True, val_ratio:float = 0, threshold:float = 0, 
        augmentations: Compose = None):
        return StrongAudioSequence(source_dir=source_dir, label_dir=label_dir, labels=labels,\
            pred_patch_sec=pred_patch_sec, pred_hop_sec=pred_hop_sec, \
            patch_sec=patch_sec, patch_hop=patch_hop, rate=rate, \
            batch_size=batch_size, shuffle=shuffle, val_ratio=val_ratio, \
            threshold=threshold, augmentations=augmentations)
    
    def train(self, epochs, train_ds, val_ds, optimizer, fine_tune, idx, reduce_method, reduce_axis, workers):
        if fine_tune:
            self.model.trainable = True
            for layer in self.model_base.layers[:-idx]:
                layer.trainable = False
        else:
            self.model_base.trainable = False
        
        rmse = tf.keras.metrics.RootMeanSquaredError()
        auc = ReducedAUC(reduce_method = reduce_method, reduce_axis = reduce_axis)
        #auc = tf.keras.metrics.AUC(curve = "PR", multi_label=True)
        self.model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=optimizer,
            metrics=[rmse, auc]
        )
        
        callback = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=3,
            restore_best_weights=True
        )

        initial_epoch = 0
        if (self.history is not None):
            initial_epoch = self.history.epoch[-1] + 1
            epochs = initial_epoch + epochs
        
        self.history = self.model.fit(train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=callback,
            initial_epoch = initial_epoch,
            workers = workers
        )
    
    def evaluate(self, dataset, threshold, reduce_method, reduce_axis):
        conf = ReducedMultiLabelConfusionMatrix(self.params.NUM_CLASSES, reduce_method, reduce_axis)
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
        if model_base:
            self.model_base.load_weights(path)
        else:
            self.model.load_weights(path)

    def save(self, path, model_base=False):
        if model_base:
            self.model_base.save(path)
        else:
            self.model.save(path)