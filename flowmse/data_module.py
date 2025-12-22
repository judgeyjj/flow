"""
FlowMSE Speech Super-Resolution (SR) data pipeline (SSR-only).

You provide:
- train_dir: folder containing 48kHz high-resolution wav files (can be nested in subfolders)
- valid_dir: folder containing 48kHz high-resolution wav files (can be nested)
- (optional) test_dir

For each sample, we ALWAYS construct the conditional input by:
  HR (48k) -> downsample to one of {8k,16k,24k,32k} -> upsample back to 48k
using FFT-based scipy.signal.resample (aligned with ClearerVoice-Studio SSR behavior).
"""

import os
from glob import glob
from os.path import join
from typing import List, Sequence, Optional

import numpy as np
import pytorch_lightning as pl
from scipy.signal import resample as scipy_resample
import torch
import torch.nn.functional as F
from torchaudio import load
from torch.utils.data import Dataset, DataLoader


def get_window(window_type: str, window_length: int) -> torch.Tensor:
    if window_type == "sqrthann":
        return torch.sqrt(torch.hann_window(window_length, periodic=True))
    if window_type == "hann":
        return torch.hann_window(window_length, periodic=True)
    raise NotImplementedError(f"Window type {window_type} not implemented!")


def _list_wavs_recursive(root_dir: str) -> List[str]:
    if root_dir is None or not os.path.isdir(root_dir):
        raise ValueError(f"Invalid directory: {root_dir}")
    files = glob(join(root_dir, "**", "*.wav"), recursive=True)
    files += glob(join(root_dir, "**", "*.WAV"), recursive=True)
    files = sorted(set(files))
    if len(files) == 0:
        raise ValueError(f"No wav files found under: {root_dir}")
    return files


def _scipy_resample_torch(wav: torch.Tensor, num_samples: int) -> torch.Tensor:
    """
    Resample waveform tensor using scipy.signal.resample (FFT-based).
    wav: Tensor (..., T) on CPU.
    """
    num_samples = int(max(int(num_samples), 1))
    wav_np = wav.detach().cpu().numpy()
    out_np = scipy_resample(wav_np, num_samples, axis=-1)
    return torch.from_numpy(out_np).to(dtype=wav.dtype)


class Specs(Dataset):
    """
    Returns a pair of complex STFTs (X, Y):
    - X: target / high-resolution speech (48kHz)
    - Y: conditional input constructed by downsample->upsample (always applied per sample)
    """

    def __init__(
        self,
        wav_dir: str,
        dummy: bool,
        shuffle_spec: bool,
        num_frames: int,
        spec_transform,
        stft_kwargs: dict,
        sampling_rate: int = 48000,
        supported_sampling_rates: Sequence[int] = (8000, 16000, 24000, 32000),
        normalize: str = "cond",
        **ignored_kwargs,
    ):
        self.wav_files = _list_wavs_recursive(wav_dir)
        self.dummy = dummy
        self.shuffle_spec = shuffle_spec
        self.num_frames = int(num_frames)

        assert all(k in stft_kwargs.keys() for k in ["n_fft", "hop_length", "center", "window"]), "misconfigured STFT kwargs"
        assert stft_kwargs.get("center", None) is True, "'center' must be True for current implementation"
        self.stft_kwargs = stft_kwargs
        self.hop_length = int(self.stft_kwargs["hop_length"])

        self.spec_transform = spec_transform

        self.sampling_rate = int(sampling_rate)
        rates = [int(s) for s in supported_sampling_rates]
        rates = [s for s in rates if s > 0 and s < self.sampling_rate]
        if len(rates) == 0:
            raise ValueError(
                f"supported_sampling_rates must contain at least one value < sampling_rate. "
                f"Got sampling_rate={self.sampling_rate}, supported_sampling_rates={supported_sampling_rates}"
            )
        self.supported_sampling_rates = rates

        if normalize not in ("cond", "target", "none"):
            raise ValueError("normalize must be one of {'cond','target','none'}")
        self.normalize = normalize

    def __len__(self):
        if self.dummy:
            return max(int(len(self.wav_files) / 200), 1)
        return len(self.wav_files)

    def __getitem__(self, i):
        x, sr_x = load(self.wav_files[i])

        # mono
        if x.dim() == 2 and x.size(0) > 1:
            x = x[:1]

        # Resample HR to target sampling_rate (if needed)
        if int(sr_x) != self.sampling_rate:
            new_len = int(round(x.size(-1) * (self.sampling_rate / float(sr_x))))
            x = _scipy_resample_torch(x, new_len)

        # Crop/pad to fixed length
        target_len = (self.num_frames - 1) * self.hop_length
        current_len = x.size(-1)
        pad = max(target_len - current_len, 0)
        if pad == 0:
            if self.shuffle_spec:
                start = int(np.random.uniform(0, current_len - target_len))
            else:
                start = int((current_len - target_len) / 2)
            x = x[..., start : start + target_len]
        else:
            x = F.pad(x, (pad // 2, pad // 2 + (pad % 2)), mode="constant")

        # ALWAYS construct conditional by downsample->upsample
        sr_out = int(np.random.choice(self.supported_sampling_rates))
        down_len = int(round(x.size(-1) * (sr_out / float(self.sampling_rate))))
        x_down = _scipy_resample_torch(x, down_len)
        y = _scipy_resample_torch(x_down, x.size(-1))

        # Normalize
        if self.normalize == "cond":
            normfac = y.abs().max()
        elif self.normalize == "target":
            normfac = x.abs().max()
        else:
            normfac = torch.tensor(1.0, dtype=x.dtype)
        normfac = torch.clamp(normfac, min=1e-8)
        x = x / normfac
        y = y / normfac

        X = torch.stft(x, **self.stft_kwargs)
        Y = torch.stft(y, **self.stft_kwargs)
        X, Y = self.spec_transform(X), self.spec_transform(Y)
        return X, Y, sr_out


class SpecsDataModule(pl.LightningDataModule):
    @staticmethod
    def add_argparse_args(parser):
        # Dataset dirs (recursive wav search)
        parser.add_argument("--train_dir", type=str, required=True, help="Training folder path (recursively search *.wav).")
        parser.add_argument("--valid_dir", type=str, required=True, help="Validation folder path (recursively search *.wav).")
        parser.add_argument("--test_dir", type=str, default=None, help="Optional test folder path (recursively search *.wav).")

        # SR settings
        parser.add_argument("--sampling_rate", type=int, default=48000, help="Target sampling rate (Hz).")
        parser.add_argument(
            "--supported_sampling_rates",
            type=int,
            nargs="+",
            default=[8000, 16000, 24000, 32000],
            help="Discrete sampling rates (Hz) for per-sample downsample (then upsample back). Must all be < sampling_rate.",
        )

        # STFT / batching
        parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
        parser.add_argument("--n_fft", type=int, default=2048, help="FFT size. For 48kHz, 2048 is recommended.")
        parser.add_argument("--hop_length", type=int, default=512, help="Hop length.")
        parser.add_argument("--num_frames", type=int, default=256, help="Number of STFT frames.")
        parser.add_argument("--window", type=str, choices=("sqrthann", "hann"), default="hann", help="STFT window.")
        parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")
        parser.add_argument("--dummy", action="store_true", help="Use reduced dummy dataset for prototyping.")

        # Spec transform
        parser.add_argument("--spec_factor", type=float, default=0.15)
        parser.add_argument("--spec_abs_exponent", type=float, default=0.5)
        parser.add_argument("--normalize", type=str, choices=("cond", "target", "none"), default="cond")
        parser.add_argument("--transform_type", type=str, choices=("exponent", "log", "none"), default="exponent")
        return parser

    def __init__(
        self,
        train_dir: str,
        valid_dir: str,
        test_dir: Optional[str] = None,
        sampling_rate: int = 48000,
        supported_sampling_rates: Sequence[int] = (8000, 16000, 24000, 32000),
        batch_size: int = 8,
        n_fft: int = 2048,
        hop_length: int = 512,
        num_frames: int = 256,
        window: str = "hann",
        num_workers: int = 4,
        dummy: bool = False,
        spec_factor: float = 0.15,
        spec_abs_exponent: float = 0.5,
        gpu: bool = True,
        normalize: str = "cond",
        transform_type: str = "exponent",
        **kwargs,
    ):
        super().__init__()
        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.test_dir = test_dir

        self.sampling_rate = int(sampling_rate)
        self.supported_sampling_rates = [int(s) for s in supported_sampling_rates]

        self.batch_size = int(batch_size)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.num_frames = int(num_frames)
        self.window = get_window(window, self.n_fft)
        self.windows = {}

        self.num_workers = int(num_workers)
        self.dummy = bool(dummy)

        self.spec_factor = float(spec_factor)
        self.spec_abs_exponent = float(spec_abs_exponent)
        self.gpu = bool(gpu)
        self.normalize = normalize
        self.transform_type = transform_type

        self.kwargs = kwargs

    def setup(self, stage=None):
        specs_kwargs = dict(
            stft_kwargs=self.stft_kwargs,
            num_frames=self.num_frames,
            spec_transform=self.spec_fwd,
            sampling_rate=self.sampling_rate,
            supported_sampling_rates=self.supported_sampling_rates,
            normalize=self.normalize,
            **self.kwargs,
        )

        if stage == "fit" or stage is None:
            self.train_set = Specs(
                wav_dir=self.train_dir,
                dummy=self.dummy,
                shuffle_spec=True,
                **specs_kwargs,
            )
            self.valid_set = Specs(
                wav_dir=self.valid_dir,
                dummy=self.dummy,
                shuffle_spec=False,
                **specs_kwargs,
            )
        if stage == "test" or stage is None:
            self.test_set = None
            if self.test_dir is not None and os.path.isdir(self.test_dir):
                self.test_set = Specs(
                    wav_dir=self.test_dir,
                    dummy=self.dummy,
                    shuffle_spec=False,
                    **specs_kwargs,
                )

    def spec_fwd(self, spec):
        if self.transform_type == "exponent":
            if self.spec_abs_exponent != 1:
                e = self.spec_abs_exponent
                spec = spec.abs() ** e * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.transform_type == "log":
            spec = torch.log(1 + spec.abs()) * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.transform_type == "none":
            spec = spec
        return spec

    def spec_back(self, spec):
        if self.transform_type == "exponent":
            spec = spec / self.spec_factor
            if self.spec_abs_exponent != 1:
                e = self.spec_abs_exponent
                spec = spec.abs() ** (1 / e) * torch.exp(1j * spec.angle())
        elif self.transform_type == "log":
            spec = spec / self.spec_factor
            spec = (torch.exp(spec.abs()) - 1) * torch.exp(1j * spec.angle())
        elif self.transform_type == "none":
            spec = spec
        return spec

    @property
    def stft_kwargs(self):
        return {**self.istft_kwargs, "return_complex": True}

    @property
    def istft_kwargs(self):
        return dict(n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, center=True)

    def _get_window(self, x):
        window = self.windows.get(x.device, None)
        if window is None:
            window = self.window.to(x.device)
            self.windows[x.device] = window
        return window

    def stft(self, sig):
        window = self._get_window(sig)
        return torch.stft(sig, **{**self.stft_kwargs, "window": window})

    def istft(self, spec, length=None):
        window = self._get_window(spec)
        return torch.istft(spec, **{**self.istft_kwargs, "window": window, "length": length})

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.gpu,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.gpu,
            shuffle=False,
        )

    def test_dataloader(self):
        if getattr(self, "test_set", None) is None:
            return None
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.gpu,
            shuffle=False,
        )

 
