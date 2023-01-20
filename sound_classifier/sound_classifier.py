from abc import ABC, abstractmethod
from sound_classifier.data import StrongAudioSequence, load_audio

class SoundClassifier(ABC):
    def __init__(self) -> None:
        """
        Construct an instance. Including model building.
        """
        pass
    
    @abstractmethod
    def features(self, waveform):
        """
        Converting a waveform to audio features, such as spectrogram.
        """
        pass

    def dataset(self, source_dir:str, label_dir:str, labels:list, pred_patch_sec:float, pred_hop_sec:float, \
        patch_sec:float = 3.0, patch_hop:float = 1.0, rate:int = 16000, \
        batch_size:int = 10, shuffle:bool = True, val_ratio=0):
        return StrongAudioSequence(source_dir, label_dir, labels, pred_patch_sec, pred_hop_sec, \
            patch_sec, patch_hop, rate, \
            batch_size, shuffle, val_ratio)

    def train(self):
        pass

    def evaluate(self, dataset):
        self.model.evaluate(dataset)

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