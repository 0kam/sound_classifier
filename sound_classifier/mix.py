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

def mix(signal, other, out, seq_len=1, snr_range=[20,40]):
    """
    Mix signal audio and background audio (other) in a SNR (dB)
    see https://engineering.linecorp.com/ja/blog/voice-waveform-arbitrary-signal-to-noise-ratio-python/
    """
    # loading data
    signal, sr_signal = torchaudio.load(signal)
    other, sr_other = torchaudio.load(other)
    # selecting the first channel
    signal = signal[0]
    other = other[0]
    # ensure that the sampling rate is same
    if sr_signal != sr_other:
        other = F.resample(other, sr_other, sr_signal)
    # clipping signal if needed
    if signal.shape[0] > (sr_signal * seq_len):
        signal = signal[0:(sr_signal * seq_len)]
    # clipping other 
    start = random.randint(0, len(other) - sr_signal * seq_len)
    other = other[start:(start + sr_signal * seq_len)]
    start = random.randint(0, len(other) - len(signal))
    end = start + len(signal)
    # Setting SNR
    snr = random.uniform(snr_range[0], snr_range[1])
    rms_s = torch.sqrt(torch.mean(torch.square(signal)))
    rms_o = torch.sqrt(torch.mean(torch.square(other)))
    a = snr / 20
    adjusted_rms = rms_s / (10**a) # Modify other to this rms
    other = other  * adjusted_rms / rms_o
    # Adding signal to other in a random position and random SNR
    other[start:end] = (signal + other[start:end]) / 2
    torchaudio.save(out, other.unsqueeze(0), sr_signal)


# Split data in train and validation
val_ratio = 0.2
signal_dirs = glob("/home/okamoto/Projects/VirtualNet/vnle/data/source/signals/*")
bgs = glob("/home/okamoto/Projects/VirtualNet/vnle/data/source/backgrounds/*.wav")
num_train = 400

for d in signal_dirs:
    sp = Path(d).stem
    files = glob(d + "/*.wav")
    random.shuffle(files)
    i = math.floor(val_ratio * len(files))
    val = files[0:i]
    train = files[i:len(files)]
    for ds in ["train", "validation"]:
        out_dir = "/home/okamoto/Projects/VirtualNet/vnle/data/fragment/" + ds + "/" + sp
        if os.path.exists(out_dir) == False:
            os.makedirs(out_dir)
        if ds == "train":
            signals = train
            n = num_train
        else:
            signals = val
            n = math.floor(num_train * val_ratio)
        for i in tqdm(range(n)):
            signal = random.choice(signals)
            bg = random.choice(bgs)
            id = random.randint(0, 100000)
            out = "/home/okamoto/Projects/VirtualNet/vnle/data/fragment/" + ds + "/" + sp + "/" + Path(signal).stem + "_" + Path(bg).stem + str(id) + ".wav"
            mix(signal, bg, out, seq_len=1, snr_range=[10, 40])


# Split others data in n segment
others = glob("/home/okamoto/Projects/VirtualNet/vnle/data/source/others/*")
n = 10
if os.path.exists("/home/okamoto/Projects/VirtualNet/vnle/data/fragment/train/others") == False:
    os.makedirs("/home/okamoto/Projects/VirtualNet/vnle/data/fragment/train/others")
if os.path.exists("/home/okamoto/Projects/VirtualNet/vnle/data/fragment/validation/others") == False:
    os.makedirs("/home/okamoto/Projects/VirtualNet/vnle/data/fragment/validation/others")

for o in others:
    other, sr = torchaudio.load(o)
    other = other[0]
    l = math.floor(other.shape[0] / sr) * sr
    other = other[:l]
    l = math.floor(l / n)
    other = [other[(i * l):(i * l + l)] for i in range(n)]
    random.shuffle(other)
    p = int(n * val_ratio)
    for i in range(n):
        if i < p:
            mode = "/validation/"
        else:
            mode = "/train/"
        out = o.replace("source", "fragment" + mode).replace(".wav", "_" + str(i) + ".wav") 
        torchaudio.save(out, other[i].unsqueeze(0), sr)
        



# Split others data in n segment
coot_rs = glob("/home/okamoto/Projects/VirtualNet/trainingWavs/resampled/ooban/*.wav")
mallard_rs = glob("/home/okamoto/Projects/VirtualNet/trainingWavs/resampled/magamo/*.wav")
n = 1
if os.path.exists("/home/okamoto/Projects/VirtualNet/vnle/data/fragment/resampled/coot") == False:
    os.makedirs("/home/okamoto/Projects/VirtualNet/vnle/data/fragment/resampled/coot")
if os.path.exists("/home/okamoto/Projects/VirtualNet/vnle/data/fragment/resampled/mallard") == False:
    os.makedirs("/home/okamoto/Projects/VirtualNet/vnle/data/fragment/resampled/mallard")
bgs = glob("/home/okamoto/Projects/VirtualNet/vnle/data/source/backgrounds/*.wav")

for d, s in zip([coot_rs, mallard_rs], ["coot/", "mallard/"]):
    for f in d:
        f = Path(f)
        for i in range(n):
            bg = random.choice(bgs)
            out = "/home/okamoto/Projects/VirtualNet/vnle/data/fragment/resampled/" + s +  f.name
            mix(f, bg, out, seq_len=1, snr_range=[20, 40])
        
