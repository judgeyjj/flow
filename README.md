### 【角色设定】 

你是一个精通音频生成模型（Diffusion/Flow Matching）的 AI 算法工程师。你需要基于我提供的 FlowMSE 代码库（包含 model.py, data_module.py, backbones 等），修复模型严重的 Spectral Bias（频谱偏差） 问题。

### 【当前问题诊断】 

我正在训练一个基于 DiT/UNet + Flow Matching 的语音超分模型。目前模型存在严重缺陷：

1. “低频复制机”：模型只学会了生成低频，高频部分全是高斯噪声。
2.  “砖墙效应”：由于输入是 FFT 硬切断（Brick-wall cutoff）的，模型不知道高频从哪里开始缺失，导致无法生成纹理。
3. 推理缺陷：验证和推理阶段，模型生成的低频甚至不如输入的条件好，且没有利用输入的低频信息。
4. 能量感知弱：高频能量太弱（-40dB ~ -80dB），MSE Loss 导致模型为了总体误差最小化而忽略了高频。

### 【任务目标：参考 NU-Wave2 和 Bridge-SR 进行工程修复】 

请修改代码，实施以下 4 点具体的工程改进：

1. 注入带宽先验 (Bandwidth Conditioning)• 原理：参考 NU-Wave2 的 BSFT 思想。• 修改：• 在 backbones (如 DiT 或 NCSNpp) 中，增加一个输入参数 cutoff_freq (或 sr_out)。• 将 cutoff_freq 编码为 Embedding（类似 Timestep Embedding），并注入到网络的所有层中（加到 temb 或作为独立 Condition）。• 目的：告诉模型“截止频率是 16kHz，请只生成 16kHz 以上的内容”。

2. 频率加权损失 (High-Frequency Weighted Loss)• 原理：强制模型关注高频，忽略低频。• 修改：• 重写 model.py 中的 _loss 函数。• 利用 batch 中的 sr_out 信息计算截止频率的 Bin 索引。• Mask 机制：截止频率以下的 Loss 权重设为 0.1 或 0；截止频率以上的 Loss 权重设为 10.0 甚至更高。• 可选：尝试对高频部分的频谱目标（Target）进行预加重（Pre-emphasis），例如乘以一个增益系数，让其数值在 Loss 中更显著。

3. 验证/推理阶段的低频替换 (Low-Frequency Replacement)• 原理：低频是已知的，不需要生成。• 修改：• 在 validation_step 和推理代码中，增加一个后处理步骤 replace_low_freq。• 操作：Output_Spec[:cutoff] = Input_Condition_Spec[:cutoff]。• 直接把输入条件（LR）的低频部分“硬拷贝”回生成结果，确保低频完美，只看模型生成的高频。

4. 输入数据预处理 (Pre-emphasis / Gain)

   原理：你提到“给高频加 40dB 增益会很亮”。• 修改：• 在 data_module.py 或 model.py 的输入处理中，考虑对高频部分应用一个固定的增益（Gain），或者使用类似 log1p 的更强压缩，提升高频在数值上的“存在感”。