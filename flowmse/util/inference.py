from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.signal import resample as scipy_resample
from torchaudio import load

from pesq import pesq
from pystoi import stoi

from .other import pad_spec
from .. import sampling


def _resample_torch_fft(wav: torch.Tensor, num_samples: int) -> torch.Tensor:
    """scipy.signal.resample wrapper for torch tensor. CPU only."""
    num_samples = int(max(int(num_samples), 1))
    x = wav.detach().cpu().numpy()
    y = scipy_resample(x, num_samples, axis=-1)
    return torch.from_numpy(y).to(dtype=wav.dtype)


def _resample_1d_np(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if sr_in == sr_out:
        return x
    new_len = int(round(x.shape[-1] * (float(sr_out) / float(sr_in))))
    new_len = max(new_len, 1)
    return scipy_resample(x, new_len)


def _nanmean(xs: List[float]) -> float:
    arr = np.asarray(xs, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def _si_sdr_np(s: np.ndarray, s_hat: np.ndarray, eps: float = 1e-8) -> float:
    s = np.asarray(s, dtype=np.float64)
    s_hat = np.asarray(s_hat, dtype=np.float64)
    s = s - s.mean()
    s_hat = s_hat - s_hat.mean()
    denom = np.sum(s ** 2) + eps
    alpha = float(np.dot(s_hat, s) / denom)
    s_target = alpha * s
    e_noise = s_hat - s_target
    return float(10.0 * np.log10((np.sum(s_target ** 2) + eps) / (np.sum(e_noise ** 2) + eps)))


@dataclass
class BucketMetrics:
    lsd_cond: float
    lsd_gen: float
    sc_cond: float
    sc_gen: float
    pesq_cond_16k: float
    pesq_gen_16k: float
    estoi_cond_16k: float
    estoi_gen_16k: float
    si_sdr_cond: float
    si_sdr_gen: float
    n: int


def evaluate_sr_buckets(
    model,
    wav_paths: Sequence[str],
    *,
    sr_target: int = 48000,
    sr_buckets: Sequence[int] = (8000, 16000, 24000, 32000),
    num_frames: Optional[int] = None,
    hop_length: Optional[int] = None,
    N: int = 5,
    device: str = "cuda",
    full_utt: bool = False,
    save_dir: Optional[str] = None,
) -> Dict[int, BucketMetrics]:
    """
    SSR offline evaluation (hard-coded buckets):
    For each HR wav: downsample to bucket sr, upsample back to sr_target as condition, then run sampling.
    Metrics are aggregated per bucket.
    """
    model = model.to(device)
    model.eval()  # uses EMA weights (if loaded) by default

    # segment length aligned with training/validation (optional)
    if num_frames is None:
        num_frames = int(getattr(model.data_module, "num_frames", 256))
    if hop_length is None:
        hop_length = int(getattr(model.data_module, "hop_length", 512))
    seg_len = int((num_frames - 1) * hop_length)

    # init accumulators
    acc: Dict[int, Dict[str, List[float]]] = {}
    for r in sr_buckets:
        acc[int(r)] = {
            "lsd_cond": [],
            "lsd_gen": [],
            "sc_cond": [],
            "sc_gen": [],
            "pesq_cond_16k": [],
            "pesq_gen_16k": [],
            "estoi_cond_16k": [],
            "estoi_gen_16k": [],
            "si_sdr_cond": [],
            "si_sdr_gen": [],
        }

    for sr_out in sr_buckets:
        sr_out = int(sr_out)
        for p in wav_paths:
            x, sr_x = load(p)
            # mono
            if x.dim() == 2 and x.size(0) > 1:
                x = x[:1]

            # resample HR to target
            if int(sr_x) != int(sr_target):
                new_len = int(round(x.size(-1) * (float(sr_target) / float(sr_x))))
                x = _resample_torch_fft(x, new_len)

            if full_utt:
                # full utterance evaluation (inference-side only)
                x_seg = x
            else:
                # deterministic crop/pad (center) aligned with training/validation segment length
                cur_len = x.size(-1)
                if cur_len >= seg_len:
                    start = int((cur_len - seg_len) / 2)
                    x_seg = x[..., start : start + seg_len]
                else:
                    pad = seg_len - cur_len
                    x_seg = torch.nn.functional.pad(x, (pad // 2, pad // 2 + (pad % 2)), mode="constant")

            target_len = int(x_seg.size(-1))

            # construct condition
            down_len = int(round(x_seg.size(-1) * (float(sr_out) / float(sr_target))))
            x_down = _resample_torch_fft(x_seg, down_len)
            y_seg = _resample_torch_fft(x_down, x_seg.size(-1))

            # normalize like dataset (cond max)
            normfac = torch.clamp(y_seg.abs().max(), min=1e-8)
            x_in = (x_seg / normfac).to(device)
            y_in = (y_seg / normfac).to(device)

            # STFT -> transformed spec
            X = model._stft(x_in)
            Y = model._stft(y_in)
            X0 = model._forward_transform(X).unsqueeze(0)  # (1, 1, F, T)
            Y0 = model._forward_transform(Y).unsqueeze(0)

            # sampling
            T_frames = Y0.size(-1)
            Yp = pad_spec(Y0)
            sampler = sampling.get_white_box_solver(
                "euler",
                model.ode,
                model,
                Y=Yp,
                T_rev=float(model.T_rev),
                t_eps=float(model.t_eps),
                N=int(N),
            )
            xhat_p, _ = sampler()
            xhat = xhat_p[..., :T_frames]

            # metrics in STFT domain (original)
            X_ref = model._backward_transform(X0)  # (1,1,F,T)
            Y_ref = model._backward_transform(Y0)
            Xhat = model._backward_transform(xhat)

            mag_X = X_ref.abs()
            mag_Y = Y_ref.abs()
            mag_Xhat = Xhat.abs()

            eps = 1e-8
            diff_cond = torch.log10((mag_X.pow(2) + eps) / (mag_Y.pow(2) + eps))
            diff_gen = torch.log10((mag_X.pow(2) + eps) / (mag_Xhat.pow(2) + eps))
            if diff_cond.size(2) > 1:
                diff_cond = diff_cond[:, :, 1:, :]
                diff_gen = diff_gen[:, :, 1:, :]
            lsd_cond = torch.sqrt(torch.mean(diff_cond.pow(2), dim=2)).mean(dim=2).mean().item()
            lsd_gen = torch.sqrt(torch.mean(diff_gen.pow(2), dim=2)).mean(dim=2).mean().item()

            flat_X = mag_X.reshape(1, -1)
            sc_cond = (
                torch.linalg.norm((mag_Y - mag_X).reshape(1, -1), dim=1)
                / torch.clamp(torch.linalg.norm(flat_X, dim=1), min=eps)
            ).mean().item()
            sc_gen = (
                torch.linalg.norm((mag_Xhat - mag_X).reshape(1, -1), dim=1)
                / torch.clamp(torch.linalg.norm(flat_X, dim=1), min=eps)
            ).mean().item()

            # time-domain metrics (restore amplitude)
            x_wav = (x_seg.squeeze(0).cpu().numpy()).astype(np.float64)
            y_wav = (y_seg.squeeze(0).cpu().numpy()).astype(np.float64)
            xhat_wav = (model.to_audio(xhat, length=target_len).squeeze(0).squeeze(0).detach().cpu().numpy() * float(normfac)).astype(np.float64)

            x_wav = np.clip(x_wav, -1.0, 1.0)
            y_wav = np.clip(y_wav, -1.0, 1.0)
            xhat_wav = np.clip(xhat_wav, -1.0, 1.0)

            # SI-SDR at HR
            try:
                sisdr_cond = _si_sdr_np(x_wav, y_wav)
            except Exception:
                sisdr_cond = float("nan")
            try:
                sisdr_gen = _si_sdr_np(x_wav, xhat_wav)
            except Exception:
                sisdr_gen = float("nan")

            # PESQ/ESTOI at 16k
            sr_eval = 16000
            ref_16k = np.clip(_resample_1d_np(x_wav, sr_target, sr_eval), -1.0, 1.0)
            cond_16k = np.clip(_resample_1d_np(y_wav, sr_target, sr_eval), -1.0, 1.0)
            gen_16k = np.clip(_resample_1d_np(xhat_wav, sr_target, sr_eval), -1.0, 1.0)

            try:
                pesq_cond = float(pesq(sr_eval, ref_16k, cond_16k, "wb"))
            except Exception:
                pesq_cond = float("nan")
            try:
                pesq_gen = float(pesq(sr_eval, ref_16k, gen_16k, "wb"))
            except Exception:
                pesq_gen = float("nan")

            try:
                estoi_cond = float(stoi(ref_16k, cond_16k, sr_eval, extended=True))
            except Exception:
                estoi_cond = float("nan")
            try:
                estoi_gen = float(stoi(ref_16k, gen_16k, sr_eval, extended=True))
            except Exception:
                estoi_gen = float("nan")

            acc[sr_out]["lsd_cond"].append(lsd_cond)
            acc[sr_out]["lsd_gen"].append(lsd_gen)
            acc[sr_out]["sc_cond"].append(sc_cond)
            acc[sr_out]["sc_gen"].append(sc_gen)
            acc[sr_out]["pesq_cond_16k"].append(pesq_cond)
            acc[sr_out]["pesq_gen_16k"].append(pesq_gen)
            acc[sr_out]["estoi_cond_16k"].append(estoi_cond)
            acc[sr_out]["estoi_gen_16k"].append(estoi_gen)
            acc[sr_out]["si_sdr_cond"].append(sisdr_cond)
            acc[sr_out]["si_sdr_gen"].append(sisdr_gen)

    out: Dict[int, BucketMetrics] = {}
    for r in sr_buckets:
        r = int(r)
        out[r] = BucketMetrics(
            lsd_cond=_nanmean(acc[r]["lsd_cond"]),
            lsd_gen=_nanmean(acc[r]["lsd_gen"]),
            sc_cond=_nanmean(acc[r]["sc_cond"]),
            sc_gen=_nanmean(acc[r]["sc_gen"]),
            pesq_cond_16k=_nanmean(acc[r]["pesq_cond_16k"]),
            pesq_gen_16k=_nanmean(acc[r]["pesq_gen_16k"]),
            estoi_cond_16k=_nanmean(acc[r]["estoi_cond_16k"]),
            estoi_gen_16k=_nanmean(acc[r]["estoi_gen_16k"]),
            si_sdr_cond=_nanmean(acc[r]["si_sdr_cond"]),
            si_sdr_gen=_nanmean(acc[r]["si_sdr_gen"]),
            n=len(acc[r]["lsd_gen"]),
        )
    return out

