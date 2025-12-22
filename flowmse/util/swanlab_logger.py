from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

# 兼容 PyTorch Lightning 1.x 和 2.x
try:
    from pytorch_lightning.loggers.logger import Logger as LightningLoggerBase
    from pytorch_lightning.loggers.logger import rank_zero_experiment
except ImportError:
    try:
        from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment
    except ImportError:
        # PL 2.0+ 使用 Logger 基类
        from lightning.pytorch.loggers.logger import Logger as LightningLoggerBase
        from lightning.pytorch.loggers.logger import rank_zero_experiment

from pytorch_lightning.utilities.rank_zero import rank_zero_only


def _sanitize_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in metrics.items():
        if v is None:
            continue
        if isinstance(v, (int, float, np.floating, np.integer)):
            out[k] = float(v)
        elif torch.is_tensor(v):
            out[k] = float(v.detach().cpu().item())
        else:
            # best-effort cast
            try:
                out[k] = float(v)
            except Exception:
                continue
    return out


class SwanLabLogger(LightningLoggerBase):
    """
    SwanLab logger，兼容 PyTorch Lightning 1.x 和 2.x。

    SwanLab API:
      - swanlab.init(project=..., experiment_name=..., description=..., config=dict)
      - swanlab.log(dict, step=?)
    """

    def __init__(
        self,
        project: str,
        experiment_name: Optional[str] = None,
        description: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        try:
            import swanlab  # type: ignore
        except Exception as e:
            raise ImportError(
                "未安装 swanlab。请先 `pip install swanlab`，或不要选择 swanlab logger。"
            ) from e

        self._swanlab = swanlab
        self._project = project
        self._experiment_name = experiment_name
        self._description = description
        self._config = dict(config or {})
        self._experiment = None

    @property
    def name(self) -> str:
        return self._project

    @property
    def version(self) -> str:
        return self._experiment_name or "default"

    @property
    @rank_zero_experiment
    def experiment(self):
        if self._experiment is None:
            self._experiment = self._swanlab.init(
                project=self._project,
                experiment_name=self._experiment_name,
                description=self._description,
                config=self._config if self._config else None,
            )
        return self._experiment

    @rank_zero_only
    def log_hyperparams(self, params: Any) -> None:
        # PL passes either Namespace or dict-like
        try:
            params_dict = dict(params)
        except Exception:
            try:
                params_dict = vars(params)
            except Exception:
                params_dict = {}

        # If experiment has not been created yet, stash config for init()
        if self._experiment is None:
            self._config.update(params_dict)
            _ = self.experiment
            return

        # Otherwise, best-effort log as a single step
        try:
            self._swanlab.log({f"hparams/{k}": v for k, v in params_dict.items()})
        except Exception:
            pass

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        _ = self.experiment  # ensure init on rank0
        metrics = _sanitize_metrics(metrics)
        # avoid noisy keys
        metrics.pop("epoch", None)
        if not metrics:
            return
        try:
            if step is None:
                self._swanlab.log(metrics)
            else:
                # Some versions support step=, some don't.
                try:
                    self._swanlab.log(metrics, step=int(step))
                except TypeError:
                    self._swanlab.log(metrics)
        except Exception:
            pass

    @rank_zero_only
    def finalize(self, status: str) -> None:
        # Optional close hook (API name may vary).
        for fn in ("finish", "close", "end"):
            if hasattr(self._swanlab, fn):
                try:
                    getattr(self._swanlab, fn)()
                except Exception:
                    pass
                break
