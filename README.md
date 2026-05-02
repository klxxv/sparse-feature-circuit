# Sparse Feature Circuit (ICLR 2025)

> **论文**: [Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models](https://arxiv.org/abs/2403.19647) (ICLR 2025)
> **作者**: Samuel Marks, Can Rager, et al.
> **项目页**: [features.baulab.info](https://features.baulab.info/)

## 核心思想

传统 circuit 分析使用粗粒度单元（attention heads / neurons）——这些单元是 polysemantic 的，难以解释。

**本文方法**: 使用 SAE 的**稀疏特征**作为 circuit 的基本单元，发现可解释的因果子网络。

## 复现目标

**目标模型** (4个):
1. **Llama 3** (8B) — LlamaScope SAE
2. **Qwen 3** (8B) — QwenScope SAE
3. **DeepSeek V4** — DeepSeek SAE
4. **Gemma 4** — Gemma Scope 2 SAE

1. ✅ 训练/加载 SAE 获取稀疏特征（4个模型）
2. ✅ 跨模型 circuit discovery：
   - 对 IOI、间接对象识别等任务比较不同模型的 circuit 结构
   - SAE 特征级别的因果图分析
3. ✅ circuit editing（干预特定特征改变模型行为）
4. ✅ 无监督 pipeline：自动发现行为 + 自动 circuit discovery

## 复现路线

### Phase 1: 基础 SAE + Activation Patching
```bash
cd scripts
uv run python activation_patching.py --model gpt2-small --task "IOI"
```

### Phase 2: Feature-Level Circuit Discovery
```bash
uv run python discover_circuit.py --model gpt2-small --feature-idx 1234
```

### Phase 3: Circuit Editing & Evaluation
```bash
uv run python edit_circuit.py --circuit-path checkpoints/circuit_ioi.pt
```

### Phase 4: 无监督 Pipeline
- 自动发现模型行为（logit lens / probing）
- 批量 circuit discovery
- 自动评估 circuit 质量

## 关键技术细节

- **Edge attribution**: 用激活梯度或 integrated gradients 计算特征间因果关系
- **Subnetwork selection**: 基于 attribution scores 阈值选择显著边
- **Circuit verification**: 通过 ablation 验证 circuit 对目标行为的充分性

## 目录结构

```
src/circuit/
├── __init__.py
├── discovery.py      # Circuit discovery algorithms
├── patching.py       # Activation patching utilities
├── editing.py        # Circuit editing interventions
├── graph.py          # Circuit graph representation
└── evaluation.py     # Circuit quality metrics
scripts/
├── activation_patching.py
├── discover_circuit.py
└── edit_circuit.py
configs/
└── default.yaml
```
