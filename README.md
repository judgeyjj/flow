## FlowMSE 语音超分（Speech Super-Resolution, SSR）

本仓库已改造成 **SSR（48kHz 目标采样率）**：
- **输入（条件）**：每个样本随机从 `{8000, 16000, 24000, 32000}` 选一个采样率，下采样后再上采样回 48kHz（每样本必做）
- **目标**：重建 48kHz 高分辨率语音
- **验证**：以 `valid_loss` 为主；同时记录 LSD/SC/PESQ/ESTOI/SI‑SDR（不计入 loss），并按 `sr_out` 动态分桶统计
- **离线推理评测**：按 8/16/24/32kHz 四桶硬编码逐桶评测，支持整句评测

## 安装

```bash
pip install -r requirements.txt
```

## 训练（推荐配置文件）

你已经把训练/验证路径写进了 `configs/sr_vctk_48k.yaml`，直接启动即可。

使用 1、2 号卡（0 开始编号）：

```bash
CUDA_VISIBLE_DEVICES=1,2 python train.py --config configs/sr_vctk_48k.yaml
```

## 离线评测（推理端，整句 + 分桶）

```bash
python evaluate.py \
  --ckpt /abs/path/to/your.ckpt \
  --test_dir /abs/path/to/vctk/test \
  --config configs/sr_vctk_48k.yaml \
  --N 5 \
  --full_utt
```


