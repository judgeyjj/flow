import argparse
import json
import os
from glob import glob
from os.path import join

import numpy as np
import torch

import yaml

from flowmse.backbones.shared import BackboneRegistry
from flowmse.data_module import SpecsDataModule
from flowmse.odes import ODERegistry
from flowmse.model import VFModel
from flowmse.util.inference import evaluate_sr_buckets


def _list_wavs_recursive(root_dir: str):
    files = glob(join(root_dir, "**", "*.wav"), recursive=True)
    files += glob(join(root_dir, "**", "*.WAV"), recursive=True)
    return sorted(set(files))


def _load_config_file(config_path):
    if not config_path:
        return {}
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if config_path.endswith((".yaml", ".yml")):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    elif config_path.endswith(".json"):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f) or {}
    else:
        raise ValueError("Config must be a .yaml/.yml or .json file")

    if not isinstance(cfg, dict):
        raise ValueError("Config file must parse into a dictionary/object")

    # Same flattening policy as train.py
    flat = {}
    for k, v in cfg.items():
        if isinstance(v, dict) and k in ("VFModel", "ODE", "Backbone", "DataModule", "pl.Trainer", "Trainer"):
            flat.update(v)
        else:
            flat[k] = v
    return flat


def _load_ckpt_to_model(model: VFModel, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    if isinstance(ckpt, dict) and "ema" in ckpt:
        try:
            model.ema.load_state_dict(ckpt["ema"])
        except Exception:
            pass


def main():
    p = argparse.ArgumentParser(description="FlowMSE SSR offline evaluation (bucketed).")
    p.add_argument("--config", type=str, default=None, help="YAML/JSON config (optional, recommended).")
    p.add_argument("--ckpt", type=str, required=True, help="Path to Lightning checkpoint (.ckpt).")
    p.add_argument("--test_dir", type=str, required=True, help="Folder containing HR wav files (recursive).")
    p.add_argument("--sr_target", type=int, default=None, help="Target HR sampling rate (defaults to config DataModule.sampling_rate or 48000).")
    p.add_argument("--sr_buckets", type=int, nargs="+", default=[8000, 16000, 24000, 32000], help="Bucket sampling rates for evaluation.")
    p.add_argument("--N", type=int, default=5, help="ODE steps for sampling.")
    p.add_argument("--num_files", type=int, default=0, help="Limit number of wavs (0 = all).")
    p.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu.")
    p.add_argument("--full_utt", action="store_true", help="Inference-side full-utterance evaluation (do NOT crop to training segment length).")
    args = p.parse_args()

    cfg = _load_config_file(args.config)
    backbone = cfg.get("backbone", None) or cfg.get("Backbone", None) or "dcunet"
    ode = cfg.get("ode", None) or cfg.get("ODE", None) or "flowmatching"

    # Ensure required dirs exist for SpecsDataModule init (we only use its STFT utilities)
    cfg.setdefault("train_dir", args.test_dir)
    cfg.setdefault("valid_dir", args.test_dir)

    sr_target = int(args.sr_target or cfg.get("sampling_rate", 48000))

    model = VFModel(
        backbone=backbone,
        ode=ode,
        data_module_cls=SpecsDataModule,
        **cfg,
    )
    _load_ckpt_to_model(model, args.ckpt)

    wavs = _list_wavs_recursive(args.test_dir)
    if args.num_files and args.num_files > 0:
        wavs = wavs[: int(args.num_files)]
    if len(wavs) == 0:
        raise ValueError(f"No wav files found under: {args.test_dir}")

    metrics = evaluate_sr_buckets(
        model,
        wavs,
        sr_target=sr_target,
        sr_buckets=args.sr_buckets,
        N=args.N,
        device=args.device,
        full_utt=bool(args.full_utt),
    )

    # Pretty print
    print("=== SSR bucketed evaluation ===")
    for r in sorted(metrics.keys()):
        m = metrics[r]
        print(
            f"[sr={r}] n={m.n} | "
            f"LSD(cond/gen)={m.lsd_cond:.4f}/{m.lsd_gen:.4f} | "
            f"SC(cond/gen)={m.sc_cond:.4f}/{m.sc_gen:.4f} | "
            f"PESQ16k(cond/gen)={m.pesq_cond_16k:.4f}/{m.pesq_gen_16k:.4f} | "
            f"ESTOI16k(cond/gen)={m.estoi_cond_16k:.4f}/{m.estoi_gen_16k:.4f} | "
            f"SI-SDR(cond/gen)={m.si_sdr_cond:.4f}/{m.si_sdr_gen:.4f}"
        )

    # Overall (simple average across buckets, weighted by n)
    total_n = sum(m.n for m in metrics.values())
    if total_n > 0:
        def wavg(field: str) -> float:
            vals = []
            ws = []
            for m in metrics.values():
                v = getattr(m, field)
                if not np.isnan(v):
                    vals.append(v * m.n)
                    ws.append(m.n)
            return float(np.sum(vals) / max(np.sum(ws), 1)) if vals else float("nan")

        print("=== Overall (weighted by n) ===")
        print(
            "LSD(cond/gen)={:.4f}/{:.4f} | SC(cond/gen)={:.4f}/{:.4f} | PESQ16k(cond/gen)={:.4f}/{:.4f} | "
            "ESTOI16k(cond/gen)={:.4f}/{:.4f} | SI-SDR(cond/gen)={:.4f}/{:.4f}".format(
                wavg("lsd_cond"),
                wavg("lsd_gen"),
                wavg("sc_cond"),
                wavg("sc_gen"),
                wavg("pesq_cond_16k"),
                wavg("pesq_gen_16k"),
                wavg("estoi_cond_16k"),
                wavg("estoi_gen_16k"),
                wavg("si_sdr_cond"),
                wavg("si_sdr_gen"),
            )
        )


if __name__ == "__main__":
    main()


