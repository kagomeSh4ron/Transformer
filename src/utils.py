"""
提供了Transformer模型所需的位置编码功能。
位置编码是Transformer架构的关键组件，用于为序列中的每个位置提供位置信息，
因为Transformer本身没有循环或卷积结构，无法直接感知位置信息。
"""

import math

import numpy as np
import torch


class PositionalEncoding(torch.nn.Module):
    """
    位置编码模块
    
    实现Transformer中的正弦位置编码，为序列中的每个位置生成唯一的位置向量。
    使用不同频率的正弦和余弦函数来编码位置信息，使得模型能够学习到相对位置关系。
    
    位置编码公式：
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    其中：
    - pos: 位置索引
    - i: 维度索引
    - d_model: 模型维度
    
    Args:
        d_model (int): 模型维度，必须与词嵌入维度相同
        max_len (int): 支持的最大序列长度，默认为5000
    """
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # 创建位置编码矩阵，形状为 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        
        # 生成位置索引，形状为 (max_len, 1)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        # 计算分母项：10000^(2i/d_model)，其中i是维度索引
        # 使用对数技巧避免数值计算问题：exp(log(10000) * 2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        # 计算位置编码
        # 偶数维度使用正弦函数
        pe[:, 0::2] = torch.sin(position * div_term)
        # 奇数维度使用余弦函数
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 添加批次维度，形状变为 (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # 将位置编码注册为缓冲区，这样它会被包含在模型的状态中
        # 但不会被当作可训练参数
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        前向传播
        
        将位置编码添加到输入张量中。位置编码会被广播到批次中的所有样本。
        
        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, d_model)
            
        Returns:
            torch.Tensor: 添加位置编码后的张量，形状与输入相同
        """
        # 将位置编码添加到输入张量
        # pe[:, :x.size(1)] 确保只使用与输入序列长度相同的位置编码
        x = x + self.pe[:, :x.size(1)]
        return x
