from abc import ABC, abstractmethod
from sound_classifier.data import AudioSequence, load_audio

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

    def dataset(self, data_dir:str, batch_size:int, rate=16000, shuffle=True, val_ratio=0):
        return AudioSequence(data_dir, batch_size, rate=rate, shuffle=shuffle, val_ratio=val_ratio)

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