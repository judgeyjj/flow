import argparse
from argparse import ArgumentParser
import json
import os
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import yaml
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from flowmse.backbones.shared import BackboneRegistry
from flowmse.data_module import SpecsDataModule
from flowmse.odes import ODERegistry
from flowmse.model import VFModel
from flowmse.util.swanlab_logger import SwanLabLogger

from datetime import datetime
import pytz

# 设置北京时间 (UTC+8)
cst = pytz.timezone('Asia/Shanghai')
now_cst = datetime.now(cst)
formatted_time_cst = now_cst.strftime("%Y%m%d%H%M%S")  # YYYYMMDDHHMMSS 格式


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

    # 支持扁平配置和分组配置 (VFModel/ODE/Backbone/DataModule/Trainer)
    flat = {}
    for k, v in cfg.items():
        if isinstance(v, dict) and k in ("VFModel", "ODE", "Backbone", "DataModule", "pl.Trainer", "Trainer"):
            flat.update(v)
        else:
            flat[k] = v
    return flat


def _num_gpus(gpus) -> int:
    """解析 gpus 参数，返回 GPU 数量"""
    if gpus is None:
        return 0
    if isinstance(gpus, bool):
        return int(gpus)
    if isinstance(gpus, int):
        if int(gpus) == -1:
            try:
                import torch
                return int(torch.cuda.device_count())
            except Exception:
                return 0
        return int(gpus)
    if isinstance(gpus, (list, tuple)):
        return len(gpus)
    try:
        g = int(gpus)
        if g == -1:
            try:
                import torch
                return int(torch.cuda.device_count())
            except Exception:
                return 0
        return g
    except Exception:
        # 例如 "0,1,2,3" 这种格式
        if isinstance(gpus, str) and "," in gpus:
            return len([x for x in gpus.split(",") if x.strip() != ""])
        return 0


def _get_num_nodes(trainer_args) -> int:
    """获取节点数量"""
    try:
        n = getattr(trainer_args, "num_nodes", None)
        return int(n) if n is not None else 1
    except Exception:
        return 1


def get_argparse_groups(parser):
    groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        groups[group.title] = argparse.Namespace(**group_dict)
    return groups


if __name__ == '__main__':
    # 第一步：获取配置文件路径并加载默认值
    config_parser = ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None, 
                               help="Path to a YAML/JSON config file. Values become argparse defaults; CLI overrides config.")
    config_args, _ = config_parser.parse_known_args()
    config_defaults = _load_config_file(config_args.config)

    # 动态参数解析
    base_parser = ArgumentParser(add_help=False)
    parser = ArgumentParser()
    for parser_ in (base_parser, parser):
        parser_.add_argument("--config", type=str, default=None, 
                            help="Path to a YAML/JSON config file. Values become argparse defaults; CLI overrides config.")
        parser_.add_argument("--backbone", type=str, choices=BackboneRegistry.get_all_names(), default="ncsnpp")
        parser_.add_argument("--ode", type=str, choices=ODERegistry.get_all_names(), default="flowmatching")    
        parser_.add_argument("--no_wandb", action='store_true', 
                            help="Turn off logging to W&B, using local default logger instead")
        parser_.add_argument("--logger", type=str, choices=("wandb", "tensorboard", "swanlab"), default=None, 
                            help="Logger backend. If not set, use wandb unless --no_wandb is set.")
        parser_.add_argument("--swanlab_project", type=str, default="FlowMSE-SR", help="SwanLab project name.")
        parser_.add_argument("--swanlab_experiment_name", type=str, default=None, 
                            help="SwanLab experiment name (defaults to auto-generated run name).")
        parser_.add_argument("--swanlab_description", type=str, default=None, help="SwanLab experiment description.")
        
    base_parser.set_defaults(**config_defaults)
    temp_args, _ = base_parser.parse_known_args()

    # 添加 VFModel、pl.Trainer、ODE 和 backbone 的参数
    backbone_cls = BackboneRegistry.get_by_name(temp_args.backbone)
    ode_class = ODERegistry.get_by_name(temp_args.ode)
    
    # 添加 Trainer 参数 (兼容 PL 2.0+)
    # 注意：pl.Trainer.add_argparse_args 在 PL 2.0 中已废弃
    try:
        parser = pl.Trainer.add_argparse_args(parser)
    except AttributeError:
        # PL 2.0+ 不再支持 add_argparse_args，手动添加常用参数
        trainer_group = parser.add_argument_group("pl.Trainer")
        trainer_group.add_argument("--max_epochs", type=int, default=200)
        trainer_group.add_argument("--gpus", type=int, default=None)
        trainer_group.add_argument("--devices", type=int, default=None)
        trainer_group.add_argument("--num_nodes", type=int, default=1)
        trainer_group.add_argument("--precision", type=str, default="32")
        trainer_group.add_argument("--accumulate_grad_batches", type=int, default=1)
        trainer_group.add_argument("--gradient_clip_val", type=float, default=None)
        trainer_group.add_argument("--check_val_every_n_epoch", type=int, default=1)
    
    VFModel.add_argparse_args(
        parser.add_argument_group("VFModel", description=VFModel.__name__))
    ode_class.add_argparse_args(
        parser.add_argument_group("ODE", description=ode_class.__name__))
    backbone_cls.add_argparse_args(
        parser.add_argument_group("Backbone", description=backbone_cls.__name__))
    
    # 添加数据模块参数
    data_module_cls = SpecsDataModule
    data_module_cls.add_argparse_args(
        parser.add_argument_group("DataModule", description=data_module_cls.__name__))
    parser.set_defaults(**config_defaults)
    
    # 解析参数并分组
    args = parser.parse_args()
    arg_groups = get_argparse_groups(parser)
    dataset_source = (
        getattr(args, "train_dir", None)
        or getattr(args, "valid_dir", None)
        or getattr(args, "base_dir", None)
        or getattr(args, "train_list", None)
        or getattr(args, "valid_list", None)
        or "dataset"
    )
    dataset = os.path.basename(os.path.normpath(dataset_source))
    trainer_ns = arg_groups.get("pl.Trainer", argparse.Namespace())
    
    # 兼容 PL 2.0+: 优先使用 devices，否则用 gpus
    num_gpus = _num_gpus(getattr(trainer_ns, "devices", None) or getattr(trainer_ns, "gpus", None))
    num_nodes = _get_num_nodes(trainer_ns)
    
    # 初始化模型
    model = VFModel(
        backbone=args.backbone, ode=args.ode, data_module_cls=data_module_cls,
        **{
            **vars(arg_groups['VFModel']),
            **vars(arg_groups['ODE']),
            **vars(arg_groups['Backbone']),
            **vars(arg_groups['DataModule']),
            "gpus": num_gpus,
        }
    )
    
    # 设置日志记录器
    name_save_dir_path = f"dataset_{dataset}_{formatted_time_cst}"
    logger_choice = args.logger or ("tensorboard" if args.no_wandb else "wandb")
    if logger_choice == "tensorboard":
        logger = TensorBoardLogger(save_dir="logs", name=name_save_dir_path)
    elif logger_choice == "swanlab":
        exp_name = args.swanlab_experiment_name or name_save_dir_path
        logger = SwanLabLogger(
            project=args.swanlab_project,
            experiment_name=exp_name,
            description=args.swanlab_description,
            config=config_defaults,
        )
    else:
        logger = WandbLogger(project="FlowMSE-SR", log_model=True, save_dir="logs", name=name_save_dir_path)
        # 避免 DDP 下重复上传代码
        @rank_zero_only
        def _log_code():
            logger.experiment.log_code(".")
        _log_code()

    # 设置回调函数
    model_dirpath = f"logs/{name_save_dir_path}_{logger.version}"
    checkpoint_callback_last = ModelCheckpoint(
        dirpath=model_dirpath, save_last=True, filename="{epoch}_last"
    )
    checkpoint_callback_valid = ModelCheckpoint(
        dirpath=model_dirpath,
        save_top_k=5,
        monitor="valid_loss",
        mode="min",
        filename="{epoch}_{valid_loss:.4f}",
    )
    callbacks = [checkpoint_callback_last, checkpoint_callback_valid]

    # 初始化 Trainer
    trainer_kwargs = {}
    world_size = int(max(num_gpus, 0) * max(num_nodes, 1))
    
    if num_gpus > 0:
        trainer_kwargs["accelerator"] = "gpu"
        trainer_kwargs["devices"] = num_gpus
        if world_size > 1:
            trainer_kwargs["strategy"] = DDPStrategy(find_unused_parameters=False)
    else:
        trainer_kwargs["accelerator"] = "cpu"
    
    # 兼容 PL 2.0+: 使用新的 Trainer 初始化方式
    try:
        trainer = pl.Trainer.from_argparse_args(
            arg_groups['pl.Trainer'],
            **trainer_kwargs,
            logger=logger, 
            log_every_n_steps=10,
            num_sanity_val_steps=1,
            callbacks=callbacks
        )
    except (AttributeError, TypeError):
        # PL 2.0+ 不再支持 from_argparse_args
        trainer = pl.Trainer(
            max_epochs=getattr(trainer_ns, "max_epochs", 200),
            num_nodes=num_nodes,
            precision=getattr(trainer_ns, "precision", "32"),
            accumulate_grad_batches=getattr(trainer_ns, "accumulate_grad_batches", 1),
            gradient_clip_val=getattr(trainer_ns, "gradient_clip_val", None),
            check_val_every_n_epoch=getattr(trainer_ns, "check_val_every_n_epoch", 1),
            **trainer_kwargs,
            logger=logger, 
            log_every_n_steps=10,
            num_sanity_val_steps=1,
            callbacks=callbacks
        )

    # 开始训练
    trainer.fit(model)