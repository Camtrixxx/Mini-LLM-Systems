# Mini-LLM-Systems

从零手写一整个 LLM 技术栈 —— Stanford **CS336: Language Modeling from Scratch**(Spring 2026)五个作业的实现、实验与报告。

从 BPE tokenizer 到 Transformer、从 Triton FlashAttention 到多卡 FSDP、从数据管线到 GRPO 对齐,全部亲手实现并在 8× A800-80GB 上跑通。

> 新手先读 [CS336-学习指南.md](CS336-学习指南.md)(课程概况、环境、数据下载、学习顺序),再看下面各作业。

## 完成进度

| # | 作业 | 主题 | 状态 | 标记 | 报告 |
|---|------|------|------|------|------|
| 1 | [`assignment1-basics`](assignment1-basics) | 基础 | ✅ 完整 | `a1-complete` | [报告](assignment1-basics/实验报告.md) |
| 2 | [`assignment2-systems`](assignment2-systems) | 系统 | ✅ 完整 | `a2-complete` | [报告](assignment2-systems/实验报告.md) |
| 3 | [`assignment3-scaling`](assignment3-scaling) | 缩放定律 | ⚠️ 部分 | — | — |
| 4 | [`assignment4-data`](assignment4-data) | 数据 | ⚠️ 部分 | — | — |
| 5 | [`assignment5-alignment`](assignment5-alignment) | 对齐 | ✅ 完整 | `a5-complete` | [报告](assignment5-alignment/实验报告.md) |

三个可本地完整完成的作业(1/2/5)均已收官,各有 `git tag`、实验报告与通过的单元测试。A3/A4 受外部依赖限制,只做了本地能验证的部分(见下)。

## 各作业亮点

### A1 · basics — 从零构建 LLM
- **BPE tokenizer**:纯 Python 实现,24/24 测试通过、与 tiktoken GPT-2 完全对齐;在 12GB OWT 上训练 vocab=32000(96 进程,138 分钟,峰值 10.3GB)
- **Transformer LM**:RMSNorm / RoPE / SwiGLU / AdamW / 训练循环全部自实现
- **Leaderboard**:OpenWebText 上 109M 模型、8 卡 DDP,验证 loss **3.259**

### A2 · systems — 性能与并行
- **FlashAttention-2**:Triton 手写前向 + PyTorch/compile 反向
- **数据并行**:手写 DDP(三种通信变体)
- **显存优化**:优化器状态分片(ZeRO-1)+ 全分片 FSDP
- 14/14 测试通过;含 profiling、通信/显存 accounting 基准

### A5 · alignment — 数学推理 RL
- **GRPO / DPO / SFT** 全套实现,26/26 测试通过
- **决定性结果**:OLMo-2-1B 全参 GRPO 200 步,让 GSM8K 准确率 **0.188 → 0.469**(+28pt,约 2.5×)—— 一条干净的 RL 学习曲线
- 诚实记录了强基座 + last-layer 设置下的负结果,并归因出"弱基座 × 全参 × 足量训练"三个必要条件

### A3 · scaling / A4 · data(部分)
- **A3**:主体依赖斯坦福内部训练 API,仅 isoflops 拟合本地可做
- **A4**:代码与单测可本地验证,完整数据管线需 375GB Common Crawl + Modal 算力

## 快速开始

统一用 [`uv`](https://docs.astral.sh/uv/) 管理环境(每个作业独立锁定,互不冲突):

```bash
cd assignment1-basics
uv run pytest              # 首次自动建虚拟环境并装依赖,然后跑测试
uv run python <script>.py  # 跑脚本
```

⚠️ 本机驱动为 CUDA 12.8,各作业 `pyproject.toml` 已把 torch 源钉到 `cu128`;A5 需 Python <3.13。详见[学习指南](CS336-学习指南.md#环境)。

## 仓库说明

- 各作业目录克隆自 [Stanford CS336 官方仓库](https://github.com/stanford-cs336),上游 commit 记录见 [UPSTREAM.md](UPSTREAM.md)
- 你写的代码在各作业的 `cs336_*/` 包目录;`tests/adapters.py` 是实现与测试的对接层
- 大文件(模型权重、数据、profiling 产物)已 gitignore;实验结果的小文件(config/log/summary)保留
