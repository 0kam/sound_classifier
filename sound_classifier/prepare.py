import torchaudio
import torchaudio.functional as F
from glob import glob
import random
import math
from pathlib import Path
import time
from tqdm import tqdm
import torch
import os
import random
from pathlib import Path

class AudioSplitter():
    """
    音声データを特定の長さの断片に分割し、音声イベント検出用の訓練データを作る。
    """
    def __init__(self, signals:str, others:str, out:str) -> None:
        """
        インスタンスを作成する。
        Parameters
        ----------
        signals : str
            検出対象カテゴリーの音が入ったディレクトリーへのパス。
            このディレクトリーには、カテゴリーごとに別れたサブディレクトリを作成し、その下に音源を入れる。
        others : str
            検出対象以外の音（バックグラウンド等）が入ったディレクトリ。
            このディレクトリーの直下に音源を入れる。
        out : str
            出力先のディレクトリー名。
        """
        self.signals = glob(signals + "/*")
        self.others = glob(others + "/*")
        self.out = out
        train_d = out + "/train"
        test_d = out + "/test"
        for d in [train_d, test_d]:
            if not os.path.exists(d):
                os.makedirs(d)

    def split(self, duration:float, test_ratio:float=0.2):
        """
        音声データをtrain/testに分割し、さらに特定の長さに分割する。
        Parameters
        ----------
        duration : float
            出力する音声断片の長さ（秒）
        test_ratio : float, default 0.2
            テストデータの割合。
        """
        # for signals
        print("Splitting signals...")
        for signal in self.signals:
            cat = Path(signal).name
            for mode in ["/train/", "/test/"]:
                o = self.out + mode + "signals/" + cat
                if not os.path.exists(o):
                    os.makedirs(o)
            signal = glob(signal + "/*")
            random.shuffle(signal)
            t = math.floor(len(signal) * test_ratio)
            for i, p in tqdm(enumerate(signal)):
                name = Path(p).name
                suf = Path(p).suffix
                if (i <= t):
                    mode = "/test/"
                else:
                    mode = "/train/"
                try:
                    fragments, sr = self._split(p, duration)
                except ValueError:
                    continue
                for i in range(fragments.shape[0]):
                    o = self.out + mode + "/signals/" + cat + "/" + name
                    o = o.replace(suf, "_" + str(i) + suf)
                    torchaudio.save(o, fragments[i].unsqueeze(0), sr)
        # for others
        print("Splitting others...")
        for mode in ["/train/", "/test/"]:
                o = self.out + mode + "others/"
                if not os.path.exists(o):
                    os.makedirs(o)
        random.shuffle(self.others)
        t = math.floor(len(self.others) * test_ratio)
        for i, p in tqdm(enumerate(self.others)):
            name = Path(p).name
            suf = Path(p).suffix
            if (i <= t):
                mode = "/test/"
            else:
                mode = "/train/"
            try:
                fragments, sr = self._split(p, duration)
            except ValueError:
                continue
            for i in range(fragments.shape[0]):
                o = self.out + mode + "/others/" + "/" + name
                o = o.replace(suf, "_" + str(i) + suf)
                torchaudio.save(o, fragments[i].unsqueeze(0), sr)

    def _split(self, p, duration):
        a, sr = torchaudio.load(p)
        a = a[0]
        l = duration * sr
        if a.shape[0] < l:
            raise ValueError("The audio file is shorter than the duration!")
        t = math.floor(a.shape[0] / l)
        fragments = []
        for i in range(t):
            start = i * l
            end = start + l
            fragments.append(a[start:end])
        if a.shape[0] > (t * l):
            start = a.shape[0] - l
            end = a.shape[0]
            fragments.append(a[start:end])
        return torch.stack(fragments), sr

    def mix(self, n:int=4, snr_range:list=[20,40]):
        """
        signalの音声断片とothersの音声断片を任意のS/N比で混ぜ、訓練データに追加する。
        先にsplit()を実行すること。
        Parameters
        ----------
        n : int
            1つのsignal音源に対して、いくつの音声データを作成するか。
        snr_range : list of int
            Signal/Noise比の幅。この中からランダムにサンプリングしたS/N比が使われる。
        """
        return NotImplementedError

asp = AudioSplitter("data/source/signals", "data/source/others", "data/fragment")
asp.split(3, 0.2)
