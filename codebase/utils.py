import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.io.wavfile as wav
import librosa
from scipy import signal
from torch.nn import functional as F


def set_config(obj, config_dic):
    for k, v in config_dic.items():
        setattr(obj, k, v)


# a base class that can be used to set attributes through keyword parameters
class ConfigBase:
    def __init__(self, **kwargs):
        set_config(self, kwargs)

def normalize(img, mean_vals, std_vals):
    return (img - mean_vals)  / std_vals


def unnormalize(img, mean_vals, std_vals, clip_min=0, clip_max=1):
    img = img * std_vals + mean_vals

    if isinstance(img, np.ndarray):
        img = np.clip(img, a_min=clip_min, a_max=clip_max)
    elif isinstance(img, torch.Tensor):
        img = torch.clamp(img, min=clip_min, max=clip_max)
    else:
        assert False, "Only support np.ndarray or torch.Tensor."
    return img


class CallbackList(list):
    def fire(self, *args, **kwargs):
        rlt = True
        for listener in self:
            listener_rlt = listener(*args, **kwargs)
            if listener_rlt is False:
                rlt = False

        return rlt


def show_imgs_tensor(nrows, ncols, imgs_arr:torch.Tensor, labels_arr:torch.Tensor, class_names):
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols)

    # draw original images first
    total_num = nrows * ncols
    for img_idx in range(total_num):
        img = imgs_arr[img_idx]
        label = labels_arr[img_idx].item()

        # let the channel be the last dimension
        img = (img * 255).detach().cpu().numpy().astype(np.uint8).transpose([1, 2, 0])

        row_idx = img_idx // ncols
        col_idx = img_idx % ncols

        axs[row_idx, col_idx].set_xlabel(f"{class_names[label]}")
        axs[row_idx, col_idx].imshow(img)

        axs[row_idx, col_idx].get_xaxis().set_ticks([])
        axs[row_idx, col_idx].get_yaxis().set_ticks([])

    plt.tight_layout()
    plt.show()
    plt.close()


def read_audio(audio_path, expected_sr=None, allow_stereo=True):
    audio_path = Path(audio_path)
    if audio_path.suffix == ".wav":
        sr, audio_np = wav.read(audio_path)

    elif audio_path.suffix == ".mp3" or audio_path.suffix == ".m4a":
        assert False, "Unsupported audio format."

    else:
        audio_np, sr = librosa.load(audio_path, sr=None)

    if audio_np.dtype == np.int16:
        audio_np = audio_np.astype(np.float32) / (2 ** 15)

    if len(audio_np.shape) == 2:
        assert allow_stereo is True
        audio_np = np.mean(audio_np, axis=1)

    assert len(audio_np.shape) == 1, "mono channel"
    assert audio_np.dtype == np.float32
    if expected_sr:
        assert sr == expected_sr, "sample rates must match"

    return audio_np, sr


def resample_wav(wav_in, sr, dest_sr):
    if isinstance(wav_in, Path):
        wav_in, sr = read_audio(wav_in, sr)

    resample_size = int(len(wav_in) / sr * dest_sr)
    resample = signal.resample(wav_in, resample_size)
    return resample


def cal_spec(raw_data, n_fft=2048, return_complex=False, win_length=None, hop_length=None):
    """
    return normalized log-scaled spectrogram, can be used by model
    if return_unnormalized==True, also return unnormalized log-scaled spectrogram
    """
    if win_length is None:
        win_length = n_fft

    if hop_length is None:
        hop_length = win_length // 4

    # STFT
    tgt_device = raw_data.get_device()

    D = torch.stft(raw_data, n_fft=n_fft, hop_length=hop_length,
                   win_length=win_length,
                   window=torch.hamming_window(win_length).to(tgt_device),
                   return_complex=True)

    if return_complex is True:
        return D

    # calculate the magnitude
    spec = torch.abs(D)
    spec = torch.log1p(spec)

    return spec








