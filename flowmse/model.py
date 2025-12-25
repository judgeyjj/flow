import os
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

        # Frequency-weighted loss for addressing spectral bias
        parser.add_argument("--freq_weighted_loss", action="store_true", help="Enable frequency-weighted loss to emphasize high-frequency generation.")
        parser.add_argument("--lf_weight", type=float, default=0.1, help="Weight for low-frequency loss (below cutoff). Default: 0.1")
        parser.add_argument("--hf_weight", type=float, default=10.0, help="Weight for high-frequency loss (above cutoff). Default: 10.0")

        # Bandwidth conditioning (NU-Wave2 style) for spectral bias fix
        parser.add_argument("--bandwidth_conditioning", action="store_true", help="Enable bandwidth conditioning: inject cutoff frequency into backbone.")

        # Optimizer settings (adam/adamw/muon)
        parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw", "muon"], help="Optimizer: adam, adamw (default), or muon")
        parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam/AdamW beta1")
        parser.add_argument("--adam_beta2", type=float, default=0.999, help="Adam/AdamW beta2")
        parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW/Muon")
        parser.add_argument("--muon_lr", type=float, default=0.02, help="Muon learning rate for matrix params (default: 0.02)")
        parser.add_argument("--muon_momentum", type=float, default=0.95, help="Muon momentum (default: 0.95)")

        # Optional GAN fine-tuning (perceptual / high-frequency detail). Disabled by default.
        parser.add_argument("--gan_enabled", action="store_true", help="Enable GAN losses + discriminators (manual optimization).")
        parser.add_argument("--gan_warmup_epochs", type=int, default=0, help="Epochs to train with only flow loss before enabling GAN losses.")
        parser.add_argument("--gan_disc_lr", type=float, default=2e-4, help="Discriminator learning rate.")
        parser.add_argument("--gan_disc_beta1", type=float, default=0.8, help="Discriminator Adam beta1.")
        parser.add_argument("--gan_disc_beta2", type=float, default=0.99, help="Discriminator Adam beta2.")
        parser.add_argument("--gan_lambda_adv", type=float, default=1.0, help="Weight for adversarial generator loss.")
        parser.add_argument("--gan_lambda_fm", type=float, default=2.0, help="Weight for discriminator feature matching loss.")
        parser.add_argument("--gan_lambda_mel", type=float, default=45.0, help="Weight for log-mel L1 loss.")
        parser.add_argument("--gan_lambda_wav", type=float, default=0.0, help="Weight for waveform L1 loss (optional).")
        parser.add_argument("--gan_mbd_fft_sizes", type=int, nargs="+", default=[2048, 1024, 512], help="FFT sizes for MultiBandDiscriminator (MBD).")
        parser.add_argument("--gan_mel_n_mels", type=int, default=80, help="Number of mel bins for mel loss.")
        parser.add_argument("--gan_mel_fmin", type=float, default=0.0, help="Mel fmin for mel loss.")
        parser.add_argument("--gan_mel_fmax", type=float, default=-1.0, help="Mel fmax for mel loss (-1 => sr/2).")
        parser.add_argument("--gan_use_mbd", action="store_true", help="Include MultiBandDiscriminator (MBD) in GAN losses.")
        return parser

    def __init__(
        self,
        backbone,
        ode,
        lr=1e-4,
        ema_decay=0.999,
        t_eps=0.03,
        T_rev=1.0,
        loss_abs_exponent=0.5,
        num_eval_files=0,
        sr_eval_steps=5,
        loss_type='mse',
        freq_weighted_loss: bool = False,
        lf_weight: float = 0.1,
        hf_weight: float = 10.0,
        bandwidth_conditioning: bool = False,
        optimizer: str = "adamw",
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        weight_decay: float = 0.01,
        muon_lr: float = 0.02,
        muon_momentum: float = 0.95,
        gan_enabled: bool = False,
        gan_warmup_epochs: int = 0,
        gan_disc_lr: float = 2e-4,
        gan_disc_beta1: float = 0.8,
        gan_disc_beta2: float = 0.99,
        gan_lambda_adv: float = 1.0,
        gan_lambda_fm: float = 2.0,
        gan_lambda_mel: float = 45.0,
        gan_lambda_wav: float = 0.0,
        gan_mbd_fft_sizes=(2048, 1024, 512),
        gan_mel_n_mels: int = 80,
        gan_mel_fmin: float = 0.0,
        gan_mel_fmax: float = -1.0,
        gan_use_mbd: bool = False,
        data_module_cls=None,
        **kwargs,
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
        # Track EMA only for the generator (vector field model), not optional GAN discriminators.
        self.ema = ExponentialMovingAverage(self.dnn.parameters(), decay=self.ema_decay)
        self._error_loading_ema = False
        self.t_eps = t_eps
        self.T_rev = T_rev
        self.ode.T_rev = T_rev
        self.loss_type = loss_type
        self.num_eval_files = num_eval_files
        self.sr_eval_steps = sr_eval_steps
        self.loss_abs_exponent = loss_abs_exponent

        # Frequency-weighted loss (spectral bias fix)
        self.freq_weighted_loss = bool(freq_weighted_loss)
        self.lf_weight = float(lf_weight)
        self.hf_weight = float(hf_weight)

        # Bandwidth conditioning (spectral bias fix - NU-Wave2 style)
        self.bandwidth_conditioning = bool(bandwidth_conditioning)

        # Optimizer settings
        self.optimizer_type = optimizer
        self.adam_beta1 = float(adam_beta1)
        self.adam_beta2 = float(adam_beta2)
        self.weight_decay = float(weight_decay)
        self.muon_lr = float(muon_lr)
        self.muon_momentum = float(muon_momentum)

        # GAN (optional)
        self.gan_enabled = bool(gan_enabled)
        self.gan_warmup_epochs = int(gan_warmup_epochs)
        self.gan_disc_lr = float(gan_disc_lr)
        self.gan_disc_beta1 = float(gan_disc_beta1)
        self.gan_disc_beta2 = float(gan_disc_beta2)
        self.gan_lambda_adv = float(gan_lambda_adv)
        self.gan_lambda_fm = float(gan_lambda_fm)
        self.gan_lambda_mel = float(gan_lambda_mel)
        self.gan_lambda_wav = float(gan_lambda_wav)
        self.gan_mbd_fft_sizes = [int(x) for x in gan_mbd_fft_sizes]
        self.gan_mel_n_mels = int(gan_mel_n_mels)
        self.gan_mel_fmin = float(gan_mel_fmin)
        self.gan_mel_fmax = float(gan_mel_fmax)
        self.gan_use_mbd = bool(gan_use_mbd)

        # PL2+: multiple optimizers require manual optimization; enable it only when GAN is on.
        self.automatic_optimization = not self.gan_enabled

        # do not serialize class objects into hparams (breaks checkpoint portability)
        self.save_hyperparameters(ignore=['no_wandb', 'data_module_cls'])
        self.data_module = data_module_cls(**kwargs, gpu=kwargs.get('gpus', 0) > 0)

        # Lazily import torchaudio + discriminators only when GAN mode is enabled, to keep baseline runnable.
        if self.gan_enabled:
            try:
                from torchaudio.transforms import MelSpectrogram
                from flowmse.gan.discriminators import MultiBandDiscriminator, MultiPeriodDiscriminator, MultiScaleDiscriminator
                from flowmse.gan.losses import discriminator_loss, feature_loss, generator_loss
            except Exception as e:
                raise ImportError(
                    "GAN mode requires torchaudio. Please ensure torchaudio is installed and importable."
                ) from e

            self.mpd = MultiPeriodDiscriminator()
            self.msd = MultiScaleDiscriminator()
            self.mbd = MultiBandDiscriminator(fft_sizes=self.gan_mbd_fft_sizes)
            self._gan_discriminator_loss = discriminator_loss
            self._gan_generator_loss = generator_loss
            self._gan_feature_loss = feature_loss

            sr = int(getattr(self.data_module, "sampling_rate", 48000))
            fmax = (sr / 2.0) if float(self.gan_mel_fmax) <= 0 else float(self.gan_mel_fmax)
            n_fft = int(getattr(self.data_module, "n_fft", 2048))
            hop = int(getattr(self.data_module, "hop_length", 512))
            mel_kwargs = dict(
                sample_rate=sr,
                n_fft=n_fft,
                hop_length=hop,
                win_length=n_fft,
                f_min=float(self.gan_mel_fmin),
                f_max=float(fmax),
                n_mels=int(self.gan_mel_n_mels),
                power=1.0,
                center=True,
                pad_mode="reflect",
                mel_scale="htk",
            )
            # torchaudio version compatibility: some versions may not support mel_scale kwarg.
            try:
                self.mel_fn = MelSpectrogram(**mel_kwargs)
            except TypeError:
                mel_kwargs.pop("mel_scale", None)
                self.mel_fn = MelSpectrogram(**mel_kwargs)



    def configure_optimizers(self):
        # Create optimizer based on optimizer_type setting
        opt_type = getattr(self, 'optimizer_type', 'adamw').lower()
        betas = (self.adam_beta1, self.adam_beta2)
        weight_decay = getattr(self, 'weight_decay', 0.01)
        
        if opt_type == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=betas)
        elif opt_type == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=betas, weight_decay=weight_decay)
        elif opt_type == 'muon':
            # MuON optimizer with MuonWithAuxAdam: matrix params use Muon, others use AdamW
            # Requires: pip install git+https://github.com/KellerJordan/Muon
            try:
                from muon import MuonWithAuxAdam
                # Separate matrix params (for Muon) vs others (for AdamW)
                matrix_params = [p for p in self.dnn.parameters() if p.ndim >= 2]
                other_params = [p for p in self.dnn.parameters() if p.ndim < 2]
                
                muon_lr = getattr(self, 'muon_lr', 0.02)  # Muon default is higher
                muon_momentum = getattr(self, 'muon_momentum', 0.95)
                
                param_groups = [
                    dict(params=matrix_params, use_muon=True, lr=muon_lr, momentum=muon_momentum, weight_decay=weight_decay),
                    dict(params=other_params, use_muon=False, lr=self.lr, betas=betas, weight_decay=weight_decay),
                ]
                optimizer = MuonWithAuxAdam(param_groups)
            except ImportError:
                warnings.warn("MuON not available. Install with: pip install git+https://github.com/KellerJordan/Muon. Falling back to AdamW.")
                optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=betas, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {opt_type}")
        
        if not getattr(self, "gan_enabled", False):
            # Add LR scheduler (ReduceLROnPlateau like SSR)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6, verbose=True
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "valid_loss"}
            }

        # GAN mode: two optimizers (manual optimization in training_step)
        opt_g = optimizer  # Use the configured optimizer for generator
        disc_params = list(self.mpd.parameters()) + list(self.msd.parameters())
        if bool(getattr(self, "gan_use_mbd", False)):
            disc_params += list(self.mbd.parameters())
        opt_d = torch.optim.AdamW(
            disc_params,
            lr=float(self.gan_disc_lr),
            betas=(float(self.gan_disc_beta1), float(self.gan_disc_beta2)),
            weight_decay=weight_decay,
        )
        return [opt_g, opt_d]

    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        # Only track EMA for the generator (vector field network)
        self.ema.update(self.dnn.parameters())

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
                self.ema.store(self.dnn.parameters())        # store current params in EMA
                self.ema.copy_to(self.dnn.parameters())      # copy EMA parameters over current params for evaluation
            else:
                # train
                if self.ema.collected_params is not None:
                    self.ema.restore(self.dnn.parameters())  # restore the EMA weights (if stored)
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)

    def _mse_loss(self, x, x_hat):    
        err = x - x_hat
        losses = torch.square(err.abs())
        # Use mean over all elements (C, F, T) for proper MSE normalization
        loss = 0.5 * torch.mean(losses)
        return loss
    
    
    def _loss(self, vectorfield, condVF):    
        if self.loss_type == 'mse':
            err = vectorfield - condVF
            losses = torch.square(err.abs())
        elif self.loss_type == 'mae':
            err = vectorfield - condVF
            losses = err.abs()
        # Use mean over all elements for proper loss normalization
        loss = 0.5 * torch.mean(losses)
        return loss

    def _freq_weighted_loss(self, vectorfield, condVF, sr_out):
        """
        Frequency-weighted loss to address spectral bias.
        
        Applies different weights to low-frequency (LF) and high-frequency (HF) bins:
        - LF (below cutoff): weight = self.lf_weight (default 0.1)
        - HF (above cutoff): weight = self.hf_weight (default 10.0)
        
        Returns:
            tuple: (total_loss, lf_loss_raw, hf_loss_raw) for logging
        """
        B, C, F, T = vectorfield.shape
        
        sr_target = int(getattr(self.data_module, "sampling_rate", 48000))
        n_fft = int(getattr(self.data_module, "n_fft", 2048))
        freq_per_bin = float(sr_target) / float(n_fft)  # Hz per frequency bin
        
        # Compute per-sample cutoff bins: cutoff = sr_out / 2 (Nyquist)
        cutoff_bins = (sr_out.float() / 2.0 / freq_per_bin).long()  # (B,)
        cutoff_bins = torch.clamp(cutoff_bins, min=1, max=F)
        
        # Create frequency weight mask: (B, 1, F, 1)
        freq_idx = torch.arange(F, device=vectorfield.device)  # (F,)
        # Broadcast: (B, F) where True = HF, False = LF
        is_hf = freq_idx.unsqueeze(0) >= cutoff_bins.unsqueeze(1)  # (B, F)
        is_lf = ~is_hf
        
        weights = torch.where(is_hf, self.hf_weight, self.lf_weight)  # (B, F)
        weights = weights.unsqueeze(1).unsqueeze(-1)  # (B, 1, F, 1)
        
        # Compute error
        err = vectorfield - condVF
        if self.loss_type == 'mse':
            raw_losses = torch.square(err.abs())
        elif self.loss_type == 'mae':
            raw_losses = err.abs()
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        
        # Weighted total loss - use mean for proper normalization
        weighted_losses = weights * raw_losses
        total_loss = 0.5 * torch.mean(weighted_losses)
        
        # Compute separate LF/HF losses for logging (unweighted, for monitoring)
        # Expand mask to full shape (B, C, F, T) for proper masked mean
        B, C, F, T = raw_losses.shape
        is_lf_full = is_lf.unsqueeze(1).unsqueeze(-1).expand(B, C, F, T)  # (B, C, F, T)
        
        lf_loss = raw_losses[is_lf_full].mean() if is_lf_full.any() else torch.tensor(0.0, device=raw_losses.device)
        hf_loss = raw_losses[~is_lf_full].mean() if (~is_lf_full).any() else torch.tensor(0.0, device=raw_losses.device)
        
        return total_loss, lf_loss, hf_loss

    def _step(self, batch, batch_idx):
        # batch may include sr_out for SR bucketed evaluation: (x0, y, sr_out)
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x0, y = batch[0], batch[1]
            sr_out = batch[2] if len(batch) >= 3 else None
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
        
        # Compute cutoff_ratio for bandwidth conditioning
        cutoff_ratio = None
        if self.bandwidth_conditioning and sr_out is not None:
            sr_target = float(getattr(self.data_module, "sampling_rate", 48000))
            cutoff_ratio = sr_out.float() / sr_target  # (B,) normalized to [0, 1]
        
        vectorfield = self(xt, t, y, cutoff_ratio=cutoff_ratio)
        
        # Use frequency-weighted loss if enabled and sr_out is available
        if self.freq_weighted_loss and sr_out is not None:
            loss, lf_loss, hf_loss = self._freq_weighted_loss(vectorfield, condVF, sr_out)
            return loss, lf_loss, hf_loss
        else:
            loss = self._loss(vectorfield, condVF)
            return loss, None, None

    def _flow_step_with_intermediates(self, batch):
        """Same as `_step` but returns intermediates for optional GAN training."""
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x0, y = batch[0], batch[1]
            sr_out = batch[2] if len(batch) >= 3 else None
        else:
            raise ValueError("Unexpected batch format. Expected (x0, y) or (x0, y, sr_out).")

        rdm = (1 - torch.rand(x0.shape[0], device=x0.device)) * (self.T_rev - self.t_eps) + self.t_eps
        t = torch.clamp(rdm, max=float(self.T_rev))
        mean, std = self.ode.marginal_prob(x0, t, y)
        z = torch.randn_like(x0)
        xt = mean + std[:, None, None, None] * z
        der_std = self.ode.der_std(t)
        der_mean = self.ode.der_mean(x0, t, y)
        condVF = der_std * z + der_mean
        
        # Compute cutoff_ratio for bandwidth conditioning
        cutoff_ratio = None
        if self.bandwidth_conditioning and sr_out is not None:
            sr_target = float(getattr(self.data_module, "sampling_rate", 48000))
            cutoff_ratio = sr_out.float() / sr_target
        
        vectorfield = self(xt, t, y, cutoff_ratio=cutoff_ratio)
        
        # Use frequency-weighted loss if enabled and sr_out is available
        if self.freq_weighted_loss and sr_out is not None:
            loss, lf_loss, hf_loss = self._freq_weighted_loss(vectorfield, condVF, sr_out)
        else:
            loss = self._loss(vectorfield, condVF)
            lf_loss, hf_loss = None, None
        return loss, lf_loss, hf_loss, xt, t, y, vectorfield, std

    def _x0_from_vectorfield(self, xt, t, y, vectorfield, std):
        """
        Estimate x0 from vectorfield for Independent CFM.
        
        In Independent CFM:
          - Path: x_t = (1-t)*y + t*x0 + sigma_min*noise
          - Vector field: v = x0 - y (the HR-LR increment)
          
        Therefore: x0 = y + v (simply add vectorfield to condition!)
        
        This is much simpler than the old OU-based formula.
        """
        return y + vectorfield

    def replace_low_freq(self, x_hat, y, sr_out, sr_target=None, blend_bins=0):
        """
        Replace low-frequency part of generated spectrogram with input condition.
        
        This addresses the "low-frequency copying" issue: since LF is already known
        from the condition, we only need the model to generate HF content.
        
        Args:
            x_hat: Generated spectrogram (B, C, F, T) - in transformed domain
            y: Input condition spectrogram (B, C, F, T) - in transformed domain
            sr_out: Per-sample downsampled rate (B,) tensor or int
            sr_target: Target sampling rate (default: self.data_module.sampling_rate)
            blend_bins: Number of bins for crossover blending (0 = hard cutoff)
        
        Returns:
            Spectrogram with LF from y and HF from x_hat
        """
        if sr_target is None:
            sr_target = int(getattr(self.data_module, "sampling_rate", 48000))
        
        n_fft = int(getattr(self.data_module, "n_fft", 2048))
        freq_per_bin = float(sr_target) / float(n_fft)  # Hz per bin
        
        B, C, F, T = x_hat.shape
        out = x_hat.clone()
        
        # Handle both tensor and scalar sr_out
        if isinstance(sr_out, (int, float)):
            sr_out = torch.full((B,), sr_out, device=x_hat.device)
        
        for i in range(B):
            # Cutoff frequency = sr_out / 2 (Nyquist of the downsampled signal)
            cutoff_hz = float(sr_out[i].item()) / 2.0
            cutoff_bin = int(cutoff_hz / freq_per_bin)
            cutoff_bin = min(cutoff_bin, F)  # Clamp to valid range
            
            if blend_bins > 0 and cutoff_bin > blend_bins:
                # Smooth crossover: blend LF->HF transition
                blend_start = cutoff_bin - blend_bins
                blend_end = cutoff_bin + blend_bins
                blend_end = min(blend_end, F)
                
                # Hard copy below blend region
                out[i, :, :blend_start, :] = y[i, :, :blend_start, :]
                
                # Linear blend in crossover region
                for b in range(blend_start, blend_end):
                    alpha = float(b - blend_start) / float(blend_end - blend_start)
                    out[i, :, b, :] = (1 - alpha) * y[i, :, b, :] + alpha * x_hat[i, :, b, :]
            else:
                # Hard cutoff (original behavior)
                out[i, :, :cutoff_bin, :] = y[i, :, :cutoff_bin, :]
        
        return out

    def training_step(self, batch, batch_idx):
        # Default (no GAN): pure flow matching objective.
        if not getattr(self, "gan_enabled", False):
            loss, lf_loss, hf_loss = self._step(batch, batch_idx)
            self.log('train_loss', loss, on_step=True, on_epoch=True)
            if lf_loss is not None:
                self.log('train_loss_lf', lf_loss, on_step=True, on_epoch=True)
                self.log('train_loss_hf', hf_loss, on_step=True, on_epoch=True)
            return loss

        # GAN mode: manual optimization (PL2+ compatible)
        opt_g, opt_d = self.optimizers()
        loss_flow, lf_loss, hf_loss, xt, t, y, vf, std = self._flow_step_with_intermediates(batch)
        self.log("train_loss_flow", loss_flow, on_step=True, on_epoch=True)
        if lf_loss is not None:
            self.log('train_loss_lf', lf_loss, on_step=True, on_epoch=True)
            self.log('train_loss_hf', hf_loss, on_step=True, on_epoch=True)

        gan_active = bool(self.current_epoch >= int(getattr(self, "gan_warmup_epochs", 0)))

        # Warmup: only flow loss, update generator
        if not gan_active:
            opt_g.zero_grad(set_to_none=True)
            self.manual_backward(loss_flow)
            opt_g.step()
            opt_g.zero_grad(set_to_none=True)
            self.ema.update(self.dnn.parameters())
            self.log("train_loss", loss_flow, on_step=True, on_epoch=True)
            return loss_flow

        # Generator output (x0 estimate) -> waveform
        x0_hat = self._x0_from_vectorfield(xt, t, y, vf, std)
        target_len = int((self.data_module.num_frames - 1) * self.data_module.hop_length)
        x_wav = self.to_audio(batch[0], length=target_len).detach()
        xhat_wav = self.to_audio(x0_hat, length=target_len)

        # ----- Train discriminators -----
        disc_loss = loss_flow.new_tensor(0.0)
        y_df_r, y_df_g, _, _ = self.mpd(x_wav, xhat_wav.detach())
        l_f, _, _ = self._gan_discriminator_loss(y_df_r, y_df_g)
        disc_loss = disc_loss + l_f

        y_ds_r, y_ds_g, _, _ = self.msd(x_wav, xhat_wav.detach())
        l_s, _, _ = self._gan_discriminator_loss(y_ds_r, y_ds_g)
        disc_loss = disc_loss + l_s

        if bool(getattr(self, "gan_use_mbd", False)):
            y_db_r, y_db_g, _, _ = self.mbd(x_wav, xhat_wav.detach())
            l_b, _, _ = self._gan_discriminator_loss(y_db_r, y_db_g)
            disc_loss = disc_loss + l_b

        opt_d.zero_grad(set_to_none=True)
        self.manual_backward(disc_loss)
        opt_d.step()
        opt_d.zero_grad(set_to_none=True)

        # ----- Train generator -----
        gen_adv = loss_flow.new_tensor(0.0)
        gen_fm = loss_flow.new_tensor(0.0)

        y_df_r, y_df_g, fmap_f_r, fmap_f_g = self.mpd(x_wav, xhat_wav)
        l_gen_f, _ = self._gan_generator_loss(y_df_g)
        gen_adv = gen_adv + l_gen_f
        gen_fm = gen_fm + self._gan_feature_loss(fmap_f_r, fmap_f_g)

        y_ds_r, y_ds_g, fmap_s_r, fmap_s_g = self.msd(x_wav, xhat_wav)
        l_gen_s, _ = self._gan_generator_loss(y_ds_g)
        gen_adv = gen_adv + l_gen_s
        gen_fm = gen_fm + self._gan_feature_loss(fmap_s_r, fmap_s_g)

        if bool(getattr(self, "gan_use_mbd", False)):
            y_db_r, y_db_g, fmap_b_r, fmap_b_g = self.mbd(x_wav, xhat_wav)
            l_gen_b, _ = self._gan_generator_loss(y_db_g)
            gen_adv = gen_adv + l_gen_b
            gen_fm = gen_fm + self._gan_feature_loss(fmap_b_r, fmap_b_g)

        # log-mel loss (helps stabilize; set gan_mel_fmax=sr/2 to include HF)
        mel_real = torch.log(torch.clamp(self.mel_fn(x_wav.squeeze(1)), min=1e-5))
        mel_fake = torch.log(torch.clamp(self.mel_fn(xhat_wav.squeeze(1)), min=1e-5))
        loss_mel = F.l1_loss(mel_fake, mel_real)

        loss_wav = F.l1_loss(xhat_wav, x_wav) if float(self.gan_lambda_wav) > 0 else loss_flow.new_tensor(0.0)

        loss_g = (
            loss_flow
            + float(self.gan_lambda_adv) * gen_adv
            + float(self.gan_lambda_fm) * gen_fm
            + float(self.gan_lambda_mel) * loss_mel
            + float(self.gan_lambda_wav) * loss_wav
        )

        opt_g.zero_grad(set_to_none=True)
        self.manual_backward(loss_g)
        opt_g.step()
        opt_g.zero_grad(set_to_none=True)
        self.ema.update(self.dnn.parameters())

        self.log("train_loss_gan_d", disc_loss, on_step=True, on_epoch=True)
        self.log("train_loss_gan_adv", gen_adv, on_step=True, on_epoch=True)
        self.log("train_loss_gan_fm", gen_fm, on_step=True, on_epoch=True)
        self.log("train_loss_gan_mel", loss_mel, on_step=True, on_epoch=True)
        if float(self.gan_lambda_wav) > 0:
            self.log("train_loss_gan_wav", loss_wav, on_step=True, on_epoch=True)
        self.log("train_loss", loss_g, on_step=True, on_epoch=True)
        return loss_g

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self._step(batch, batch_idx)
        self.log('valid_loss', loss, on_step=False, on_epoch=True)

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

                # Get sampling rate info for bandwidth conditioning
                sr_target = int(getattr(self.data_module, "sampling_rate", 48000))
                
                sampler = sampling.get_white_box_solver(
                    "euler",
                    self.ode,
                    self,
                    Y=y_p,
                    T_rev=self.T_rev,
                    t_eps=self.t_eps,
                    N=int(self.sr_eval_steps),
                    sr_out=sr_out,
                    sr_target=sr_target,
                )
                xhat_p, _ = sampler()
                xhat = xhat_p[..., :T]

                # Low-frequency replacement: use condition LF, only evaluate model's HF generation
                if sr_out is not None:
                    xhat = self.replace_low_freq(xhat, y, sr_out, blend_bins=0)

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

                # Only log core metrics for cleaner SwanLab
                self.log("val_lsd_gen", lsd_gen, on_step=False, on_epoch=True)

                # Core metrics logged (bucketed metrics disabled for cleaner logs)
                # To enable per-SR metrics, uncomment the bucketed logging below

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

                # Only log core metrics for cleaner SwanLab
                self.log("val_pesq_gen_16k", pesq_gen, on_step=False, on_epoch=True)
                self.log("val_estoi_gen_16k", estoi_gen, on_step=False, on_epoch=True)
                self.log("val_si_sdr_gen", sisdr_gen, on_step=False, on_epoch=True)

                # Bucketed time-domain metrics disabled for cleaner logs

                # 保存验证音频样本 (每个采样率各保存一个)
                try:
                    import soundfile as sf
                    # 使用固定路径，避免访问可能阻塞的 trainer.log_dir
                    audio_save_dir = "logs/val_audio"
                    os.makedirs(audio_save_dir, exist_ok=True)
                    epoch = self.current_epoch
                    saved_rates = set()
                    sr_np = sr_out.detach().cpu().numpy().astype(int) if sr_out is not None else np.zeros(n, dtype=int)
                    for i in range(n):
                        sr_tag = int(sr_np[i])
                        if sr_tag in saved_rates:
                            continue
                        saved_rates.add(sr_tag)
                        gt_path = os.path.join(audio_save_dir, f"epoch{epoch:03d}_sr{sr_tag}_gt.wav")
                        lr_path = os.path.join(audio_save_dir, f"epoch{epoch:03d}_sr{sr_tag}_lr.wav")
                        sr_path = os.path.join(audio_save_dir, f"epoch{epoch:03d}_sr{sr_tag}_sr.wav")
                        sf.write(gt_path, x_wav[i], sr_hr)
                        sf.write(lr_path, y_wav[i], sr_hr)
                        sf.write(sr_path, xhat_wav[i], sr_hr)
                        print(f"[val_audio] 保存: {sr_path}")
                except Exception as e:
                    print(f"[val_audio] 保存失败: {e}")
                    warnings.warn(f"保存验证音频失败: {e}")

        return loss

    def forward(self, x, t, y, cutoff_ratio=None):
        """
        Forward pass through the vector field model.
        
        Args:
            x: Noisy spectrogram at time t (B, C, F, T)
            t: Diffusion timestep (B,)
            y: Condition spectrogram (B, C, F, T)
            cutoff_ratio: Optional bandwidth cutoff ratio = sr_out / sr_target (B,)
                          Only used when bandwidth_conditioning is enabled.
        """
        # Concatenate y as an extra channel
        dnn_input = torch.cat([x, y], dim=1)
        
        # Pass cutoff_ratio to backbone if bandwidth conditioning is enabled
        if self.bandwidth_conditioning and cutoff_ratio is not None:
            score = -self.dnn(dnn_input, t, cutoff_ratio=cutoff_ratio)
        else:
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



