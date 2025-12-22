import time
from math import ceil
import warnings
import numpy as np
import torch
import pytorch_lightning as pl
from torch_ema import ExponentialMovingAverage
import torch.nn.functional as F
from scipy.signal import resample as scipy_resample
from pesq import pesq
from pystoi import stoi
from flowmse import sampling
from flowmse.odes import ODERegistry
from flowmse.backbones import BackboneRegistry
from flowmse.util.other import pad_spec
import numpy as np
import matplotlib.pyplot as plt
import random


def _resample_1d_np(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """FFT-based resample, aligned with our dataset degradation behavior."""
    x = np.asarray(x, dtype=np.float64)
    if sr_in == sr_out:
        return x
    new_len = int(round(x.shape[-1] * (float(sr_out) / float(sr_in))))
    new_len = max(new_len, 1)
    return scipy_resample(x, new_len)


def _nanmean(xs) -> float:
    arr = np.array(xs, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def _si_sdr_np(s: np.ndarray, s_hat: np.ndarray, eps: float = 1e-8) -> float:
    """Scale-Invariant SDR."""
    s = np.asarray(s, dtype=np.float64)
    s_hat = np.asarray(s_hat, dtype=np.float64)
    s = s - s.mean()
    s_hat = s_hat - s_hat.mean()
    denom = np.sum(s ** 2) + eps
    alpha = float(np.dot(s_hat, s) / denom)
    s_target = alpha * s
    e_noise = s_hat - s_target
    return float(10.0 * np.log10((np.sum(s_target ** 2) + eps) / (np.sum(e_noise ** 2) + eps)))


class VFModel(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate (1e-4 by default)")
        parser.add_argument("--ema_decay", type=float, default=0.999, help="The parameter EMA decay constant (0.999 by default)")
        parser.add_argument("--t_eps", type=float, default=0.03, help="t_delta in the paper")
        parser.add_argument("--T_rev",type=float, default=1.0, help="Starting point t_N in the paper")
        
        parser.add_argument("--num_eval_files", type=int, default=0, help="Number of validation samples to run SR sampling and report SR metrics (LSD/SC). 0 disables SR metric evaluation.")
        parser.add_argument("--sr_eval_steps", type=int, default=5, help="Number of ODE steps (N) used for SR sampling during validation.")
        parser.add_argument("--loss_type", type=str, default="mse", help="The type of loss function to use.")
        parser.add_argument("--loss_abs_exponent", type=float, default= 0.5,  help="magnitude transformation in the loss term")
        return parser

    def __init__(
        self, backbone, ode, lr=1e-4, ema_decay=0.999, t_eps=0.03, T_rev = 1.0,  loss_abs_exponent=0.5, 
        num_eval_files=0, sr_eval_steps=5, loss_type='mse', data_module_cls=None, **kwargs
    ):
        """
        Create a new ScoreModel.

        Args:
            backbone: Backbone DNN that serves as a vector field model.
            ode: The ode used.
            lr: The learning rate of the optimizer. (1e-4 by default).
            ema_decay: The decay constant of the parameter EMA (0.999 by default).
            t_eps: The minimum time to practically run for to avoid issues very close to zero (1e-5 by default).
            loss_type: The type of loss to use (wrt. noise z/std). Options are 'mse' (default), 'mae'
        """
        super().__init__()
        # Initialize Backbone DNN
        dnn_cls = BackboneRegistry.get_by_name(backbone)
        self.dnn = dnn_cls(**kwargs)
        
        
        ode_cls = ODERegistry.get_by_name(ode)
        self.ode = ode_cls(**kwargs)
        # Store hyperparams and save them
        self.lr = lr
        self.ema_decay = ema_decay
        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)
        self._error_loading_ema = False
        self.t_eps = t_eps
        self.T_rev = T_rev
        self.ode.T_rev = T_rev
        self.loss_type = loss_type
        self.num_eval_files = num_eval_files
        self.sr_eval_steps = sr_eval_steps
        self.loss_abs_exponent = loss_abs_exponent
        # do not serialize class objects into hparams (breaks checkpoint portability)
        self.save_hyperparameters(ignore=['no_wandb', 'data_module_cls'])
        self.data_module = data_module_cls(**kwargs, gpu=kwargs.get('gpus', 0) > 0)



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.parameters())

    # on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint.get('ema', None)
        if ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")

    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema'] = self.ema.state_dict()

    def train(self, mode, no_ema=False):
        res = super().train(mode)  # call the standard `train` method with the given mode
        if not self._error_loading_ema:
            if mode == False and not no_ema:
                # eval
                self.ema.store(self.parameters())        # store current params in EMA
                self.ema.copy_to(self.parameters())      # copy EMA parameters over current params for evaluation
            else:
                # train
                if self.ema.collected_params is not None:
                    self.ema.restore(self.parameters())  # restore the EMA weights (if stored)
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)

    def _mse_loss(self, x, x_hat):    
        err = x-x_hat
        losses = torch.square(err.abs())

        # taken from reduce_op function: sum over channels and position and mean over batch dim
        # presumably only important for absolute loss number, not for gradients
        loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
        return loss
    
    
    def _loss(self, vectorfield, condVF):    
        if self.loss_type == 'mse':
            err = vectorfield-condVF
            losses = torch.square(err.abs())
        elif self.loss_type == 'mae':
            err = vectorfield-condVF
            losses = err.abs()
        # taken from reduce_op function: sum over channels and position and mean over batch dim
        # presumably only important for absolute loss number, not for gradients
        loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
        return loss

    def _step(self, batch, batch_idx):
        # batch may include sr_out for SR bucketed evaluation: (x0, y, sr_out)
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x0, y = batch[0], batch[1]
        else:
            raise ValueError("Unexpected batch format. Expected (x0, y) or (x0, y, sr_out).")
        rdm = (1-torch.rand(x0.shape[0], device=x0.device)) * (self.T_rev - self.t_eps) + self.t_eps
        # keep on the same device/dtype
        t = torch.clamp(rdm, max=float(self.T_rev))
        mean, std = self.ode.marginal_prob(x0, t, y)
        z = torch.randn_like(x0)  #
        sigmas = std[:, None, None, None]
        xt = mean + sigmas * z
        der_std = self.ode.der_std(t)
        der_mean = self.ode.der_mean(x0,t,y)
        condVF = der_std * z + der_mean
        vectorfield = self(xt, t, y)
        loss = self._loss(vectorfield, condVF)
        return loss
    

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('valid_loss', loss, on_step=False, on_epoch=True, sync_dist=True)

        # SR validation metrics (requires sampling; computed on a small subset)
        # Only run heavy sampling metrics on global rank 0 under DDP.
        if (
            batch_idx == 0
            and self.num_eval_files
            and self.num_eval_files > 0
            and self.sr_eval_steps > 0
            and (getattr(getattr(self, "trainer", None), "is_global_zero", True))
        ):
            with torch.no_grad():
                x0, y = batch[0], batch[1]
                sr_out = batch[2] if isinstance(batch, (list, tuple)) and len(batch) >= 3 else None
                n = min(int(self.num_eval_files), int(x0.shape[0]))
                x0 = x0[:n]
                y = y[:n]
                if sr_out is not None:
                    sr_out = sr_out[:n]

                # Pad time dim if needed (no-op if already divisible by 64)
                T = y.size(3)
                y_p = pad_spec(y)

                sampler = sampling.get_white_box_solver(
                    "euler",
                    self.ode,
                    self,
                    Y=y_p,
                    T_rev=self.T_rev,
                    t_eps=self.t_eps,
                    N=int(self.sr_eval_steps),
                )
                xhat_p, _ = sampler()
                xhat = xhat_p[..., :T]

                # Convert to original STFT domain for metrics
                X = self._backward_transform(x0)
                Yb = self._backward_transform(y)
                Xhat = self._backward_transform(xhat)
                mag_X = X.abs()
                mag_Y = Yb.abs()
                mag_Xhat = Xhat.abs()

                eps = 1e-8
                # LSD (as in the provided formula):
                #   LSD = (1/T) * sum_t sqrt( (1/F) * sum_f ( log10( S(t,f)^2 / Ŝ(t,f)^2 ) )^2 )
                # Here we use magnitude spectra |X| as S and |X_hat| as Ŝ.
                diff_cond = torch.log10((mag_X.pow(2) + eps) / (mag_Y.pow(2) + eps))      # (B, C, F, T)
                diff_gen = torch.log10((mag_X.pow(2) + eps) / (mag_Xhat.pow(2) + eps))    # (B, C, F, T)

                # Exclude DC bin by default for SSR-style LSD (f=1..F in common formulas)
                if diff_cond.size(2) > 1:
                    diff_cond = diff_cond[:, :, 1:, :]
                    diff_gen = diff_gen[:, :, 1:, :]

                # per-frame RMS over frequency -> (B, C, T)
                lsd_cond_bt = torch.sqrt(torch.mean(diff_cond.pow(2), dim=2))
                lsd_gen_bt = torch.sqrt(torch.mean(diff_gen.pow(2), dim=2))

                # per-sample (for sr bucket logging)
                lsd_cond_b = lsd_cond_bt.mean(dim=2).mean(dim=1)  # (B,)
                lsd_gen_b = lsd_gen_bt.mean(dim=2).mean(dim=1)    # (B,)

                # mean over batch
                lsd_cond = lsd_cond_b.mean()
                lsd_gen = lsd_gen_b.mean()

                flat_X = mag_X.reshape(n, -1)
                sc_cond_b = (
                    torch.linalg.norm((mag_Y - mag_X).reshape(n, -1), dim=1)
                    / torch.clamp(torch.linalg.norm(flat_X, dim=1), min=eps)
                )
                sc_gen_b = (
                    torch.linalg.norm((mag_Xhat - mag_X).reshape(n, -1), dim=1)
                    / torch.clamp(torch.linalg.norm(flat_X, dim=1), min=eps)
                )

                sc_cond = sc_cond_b.mean()
                sc_gen = sc_gen_b.mean()

                self.log("val_lsd_cond", lsd_cond, on_step=False, on_epoch=True, sync_dist=True)
                self.log("val_lsd_gen", lsd_gen, on_step=False, on_epoch=True, sync_dist=True)
                self.log("val_sc_cond", sc_cond, on_step=False, on_epoch=True, sync_dist=True)
                self.log("val_sc_gen", sc_gen, on_step=False, on_epoch=True, sync_dist=True)
                self.log("val_lsd_gain", lsd_cond - lsd_gen, on_step=False, on_epoch=True, sync_dist=True)

                # Bucketed logging by dynamic sr_out (only for the sampled subset)
                if sr_out is not None:
                    try:
                        bucket_rates = list(getattr(self.data_module, "supported_sampling_rates", []))
                    except Exception:
                        bucket_rates = []
                    for rate in bucket_rates:
                        mask = (sr_out == int(rate))
                        if bool(mask.any()):
                            self.log(f"val_n_sr{int(rate)}", float(mask.sum()), on_step=False, on_epoch=True, sync_dist=True)
                            self.log(f"val_lsd_cond_sr{int(rate)}", lsd_cond_b[mask].mean(), on_step=False, on_epoch=True, sync_dist=True)
                            self.log(f"val_lsd_gen_sr{int(rate)}", lsd_gen_b[mask].mean(), on_step=False, on_epoch=True, sync_dist=True)
                            self.log(f"val_sc_cond_sr{int(rate)}", sc_cond_b[mask].mean(), on_step=False, on_epoch=True, sync_dist=True)
                            self.log(f"val_sc_gen_sr{int(rate)}", sc_gen_b[mask].mean(), on_step=False, on_epoch=True, sync_dist=True)
                            self.log(f"val_lsd_gain_sr{int(rate)}", (lsd_cond_b[mask].mean() - lsd_gen_b[mask].mean()), on_step=False, on_epoch=True, sync_dist=True)

                # Time-domain metrics (not part of loss): PESQ/ESTOI are computed on 16kHz downsampled audio.
                target_len = int((self.data_module.num_frames - 1) * self.data_module.hop_length)
                sr_hr = int(getattr(self.data_module, "sampling_rate", 48000))
                sr_eval = 16000

                x_wav = self.to_audio(x0, length=target_len).squeeze(1).detach().cpu().numpy()
                y_wav = self.to_audio(y, length=target_len).squeeze(1).detach().cpu().numpy()
                xhat_wav = self.to_audio(xhat, length=target_len).squeeze(1).detach().cpu().numpy()

                pesq_cond_list, pesq_gen_list = [], []
                estoi_cond_list, estoi_gen_list = [], []
                sisdr_cond_list, sisdr_gen_list = [], []

                for i in range(n):
                    ref = np.clip(x_wav[i], -1.0, 1.0)
                    cond = np.clip(y_wav[i], -1.0, 1.0)
                    gen = np.clip(xhat_wav[i], -1.0, 1.0)

                    # SI-SDR at HR rate (scale-invariant)
                    try:
                        sisdr_cond_list.append(_si_sdr_np(ref, cond))
                    except Exception:
                        sisdr_cond_list.append(float("nan"))
                    try:
                        sisdr_gen_list.append(_si_sdr_np(ref, gen))
                    except Exception:
                        sisdr_gen_list.append(float("nan"))

                    # PESQ/ESTOI at 16kHz
                    ref_16k = _resample_1d_np(ref, sr_hr, sr_eval)
                    cond_16k = _resample_1d_np(cond, sr_hr, sr_eval)
                    gen_16k = _resample_1d_np(gen, sr_hr, sr_eval)
                    ref_16k = np.clip(ref_16k, -1.0, 1.0)
                    cond_16k = np.clip(cond_16k, -1.0, 1.0)
                    gen_16k = np.clip(gen_16k, -1.0, 1.0)

                    try:
                        pesq_cond_list.append(pesq(sr_eval, ref_16k, cond_16k, "wb"))
                    except Exception:
                        pesq_cond_list.append(float("nan"))
                    try:
                        pesq_gen_list.append(pesq(sr_eval, ref_16k, gen_16k, "wb"))
                    except Exception:
                        pesq_gen_list.append(float("nan"))

                    try:
                        estoi_cond_list.append(stoi(ref_16k, cond_16k, sr_eval, extended=True))
                    except Exception:
                        estoi_cond_list.append(float("nan"))
                    try:
                        estoi_gen_list.append(stoi(ref_16k, gen_16k, sr_eval, extended=True))
                    except Exception:
                        estoi_gen_list.append(float("nan"))

                pesq_cond = _nanmean(pesq_cond_list)
                pesq_gen = _nanmean(pesq_gen_list)
                estoi_cond = _nanmean(estoi_cond_list)
                estoi_gen = _nanmean(estoi_gen_list)
                sisdr_cond = _nanmean(sisdr_cond_list)
                sisdr_gen = _nanmean(sisdr_gen_list)

                self.log("val_pesq_cond_16k", pesq_cond, on_step=False, on_epoch=True, sync_dist=True)
                self.log("val_pesq_gen_16k", pesq_gen, on_step=False, on_epoch=True, sync_dist=True)
                self.log("val_pesq_gain_16k", pesq_gen - pesq_cond, on_step=False, on_epoch=True, sync_dist=True)
                self.log("val_estoi_cond_16k", estoi_cond, on_step=False, on_epoch=True, sync_dist=True)
                self.log("val_estoi_gen_16k", estoi_gen, on_step=False, on_epoch=True, sync_dist=True)
                self.log("val_estoi_gain_16k", estoi_gen - estoi_cond, on_step=False, on_epoch=True, sync_dist=True)
                self.log("val_si_sdr_cond", sisdr_cond, on_step=False, on_epoch=True, sync_dist=True)
                self.log("val_si_sdr_gen", sisdr_gen, on_step=False, on_epoch=True, sync_dist=True)
                self.log("val_si_sdr_gain", sisdr_gen - sisdr_cond, on_step=False, on_epoch=True, sync_dist=True)

                # Bucketed time-domain metrics (same subset)
                if sr_out is not None:
                    try:
                        bucket_rates = list(getattr(self.data_module, "supported_sampling_rates", []))
                    except Exception:
                        bucket_rates = []
                    sr_np = sr_out.detach().cpu().numpy().astype(int)
                    for rate in bucket_rates:
                        idx = np.where(sr_np == int(rate))[0]
                        if idx.size == 0:
                            continue
                        self.log(f"val_pesq_cond_16k_sr{int(rate)}", _nanmean([pesq_cond_list[i] for i in idx]), on_step=False, on_epoch=True, sync_dist=True)
                        self.log(f"val_pesq_gen_16k_sr{int(rate)}", _nanmean([pesq_gen_list[i] for i in idx]), on_step=False, on_epoch=True, sync_dist=True)
                        self.log(f"val_estoi_cond_16k_sr{int(rate)}", _nanmean([estoi_cond_list[i] for i in idx]), on_step=False, on_epoch=True, sync_dist=True)
                        self.log(f"val_estoi_gen_16k_sr{int(rate)}", _nanmean([estoi_gen_list[i] for i in idx]), on_step=False, on_epoch=True, sync_dist=True)
                        self.log(f"val_si_sdr_cond_sr{int(rate)}", _nanmean([sisdr_cond_list[i] for i in idx]), on_step=False, on_epoch=True, sync_dist=True)
                        self.log(f"val_si_sdr_gen_sr{int(rate)}", _nanmean([sisdr_gen_list[i] for i in idx]), on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def forward(self, x, t, y):
        # Concatenate y as an extra channel
        dnn_input = torch.cat([x, y], dim=1)
        
        # the minus is most likely unimportant here - taken from Song's repo
        score = -self.dnn(dnn_input, t)
        return score

    def to(self, *args, **kwargs):
        """Override PyTorch .to() to also transfer the EMA of the model weights"""
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)


    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def setup(self, stage=None):
        return self.data_module.setup(stage=stage)

    def to_audio(self, spec, length=None):
        return self._istft(self._backward_transform(spec), length)

    def _forward_transform(self, spec):
        return self.data_module.spec_fwd(spec)

    def _backward_transform(self, spec):
        return self.data_module.spec_back(spec)

    def _stft(self, sig):
        return self.data_module.stft(sig)

    def _istft(self, spec, length=None):
        return self.data_module.istft(spec, length)



