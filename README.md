# Transformer机器翻译项目

本项目实现了一个基于Transformer架构的英德机器翻译系统，使用IWSLT数据集进行训练和评估。

## 核心组件

### 模型架构
- **多头注意力机制**: 4个注意力头，支持并行计算
- **位置编码**: 正弦位置编码，处理序列位置信息
- **残差连接**: 解决深层网络梯度消失问题
- **层归一化**: 稳定训练过程，加速收敛

### 技术实现
- **编码器-解码器结构**: 2层编码器和2层解码器
- **Teacher Forcing**: 训练时使用真实目标序列
- **SentencePiece分词**: 支持子词级别的文本处理
- **梯度裁剪**: 防止梯度爆炸，稳定训练

## 环境要求

- Python 3.8+
- PyTorch 1.9.0+
- 其他依赖见 `requirements.txt`

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 快速训练测试（CPU，小规模数据）

```bash
python -u src/train.py --data_dir en-de --spm_en en-de/sp_en.model --spm_de en-de/sp_de.model --epochs 10 --batch_size 32 --max_examples 200
```

### 3. 完整训练（推荐GPU）

```bash
python -u src/train.py --data_dir en-de --spm_en en-de/sp_en.model --spm_de en-de/sp_de.model --epochs 50 --batch_size 64 --max_examples 10000
```

### 4. 模型评估

```bash
python src/evaluation.py
```

## 输出文件

训练完成后，会在 `results/` 目录下生成以下文件：

- `checkpoints/model.pt`: 训练好的模型权重
- `loss.png`: 训练损失曲线图
- `loss.json`: 损失数据（JSON格式）
- `sample_translations.txt`: 翻译样本（200个）
- `bleu_scores.json`: BLEU评估分数

## 项目结构

```
transformer/
├── en-de/                    # 英德语料数据
│   ├── train.tags.en-de.en   # 英语训练数据
│   ├── train.tags.en-de.de   # 德语训练数据
│   ├── sp_en.model           # 英语SentencePiece模型
│   └── sp_de.model           # 德语SentencePiece模型
├── src/                      # 源代码目录
│   ├── train.py              # 训练脚本
│   ├── evaluation.py         # 评估脚本
│   ├── model.py              # Transformer模型实现
│   ├── dataset.py            # 数据集处理
│   └── utils.py              # 工具函数
├── results/                  # 实验结果目录
│   ├── checkpoints/          # 模型检查点
│   ├── sample_translations.txt
│   ├── loss.png
│   └── bleu_scores.json
├── requirements.txt          # 依赖列表
└── README.md                # 项目说明文档
```

## 模型参数

- **模型维度**: 256
- **编码器/解码器层数**: 2层
- **注意力头数**: 4个
- **前馈网络维度**: 512
- **Dropout率**: 0.1
- **学习率**: 1e-3
- **批次大小**: 32
- **最大序列长度**: 128


## 技术细节

### 训练策略
- 使用AdamW优化器
- 梯度裁剪（最大梯度范数：1.0）
- Teacher Forcing训练
- 交叉熵损失函数（忽略填充token）

### 数据处理
- SentencePiece子词分词
- 动态填充到批次最大长度
- 支持变长序列处理

### 评估指标
- BLEU-1到BLEU-4分数
- 平滑处理避免短句子异常
- 支持多语言文本分词
