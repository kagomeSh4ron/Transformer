"""
手工实现Transformer，包含以下组件：
1. MultiHeadAttention
2. FeedForward
3. EncoderLayer/DecoderLayer
4. Encoder/Decoder
5. Transformer: 完整的Transformer模型
"""

import math

import torch
import torch.nn as nn

from utils import PositionalEncoding


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    
    实现Transformer中的多头自注意力机制，允许模型同时关注序列中不同位置的信息。
    通过将输入投影到多个子空间，并行计算多个注意力头，最后合并结果。
    
    Args:
        d_model (int): 模型维度
        num_heads (int): 注意力头数量
        dropout (float): Dropout概率，默认为0.1
    """
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        # 确保模型维度能被头数整除
        assert d_model % num_heads == 0
        
        self.d_k = d_model // num_heads  # 每个头的维度
        self.h = num_heads  # 注意力头数量
        
        # 线性变换层：用于生成Q、K、V矩阵
        self.q_linear = nn.Linear(d_model, d_model)  # Query变换
        self.k_linear = nn.Linear(d_model, d_model)  # Key变换
        self.v_linear = nn.Linear(d_model, d_model)  # Value变换
        self.out = nn.Linear(d_model, d_model)       # 输出变换
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        前向传播
        
        Args:
            q: Query张量 (batch_size, seq_len, d_model)
            k: Key张量 (batch_size, seq_len, d_model)
            v: Value张量 (batch_size, seq_len, d_model)
            mask: 注意力掩码，用于屏蔽某些位置
            
        Returns:
            torch.Tensor: 注意力输出 (batch_size, seq_len, d_model)
        """
        bs = q.size(0)  # 批次大小
        
        # 线性变换并重塑为多头格式
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        # 现在q,k,v的形状为: (bs, h, seq_len, d_k)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用掩码（如果提供）
        if mask is not None:
            # mask形状: (bs, 1, 1, seq_len) 或可广播的形状
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 计算注意力权重
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # 应用注意力权重到Value
        out = torch.matmul(attn, v)  # (bs, h, seq_len, d_k)
        
        # 重塑回原始格式并应用输出变换
        out = out.transpose(1, 2).contiguous().view(bs, -1, self.h * self.d_k)
        return self.out(out)


class FeedForward(nn.Module):
    """
    前馈神经网络
    
    实现Transformer中的位置前馈网络，包含两个线性变换和一个ReLU激活函数。
    这是Transformer架构中的标准组件，用于处理注意力层的输出。
    
    Args:
        d_model (int): 模型维度
        d_ff (int): 前馈网络的隐藏层维度
        dropout (float): Dropout概率，默认为0.1
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),  # 第一个线性变换
            nn.ReLU(),                 # ReLU激活函数
            nn.Dropout(dropout),       # Dropout正则化
            nn.Linear(d_ff, d_model)    # 第二个线性变换
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, seq_len, d_model)
            
        Returns:
            torch.Tensor: 输出张量 (batch_size, seq_len, d_model)
        """
        return self.net(x)


class EncoderLayer(nn.Module):
    """
    编码器层
    
    实现Transformer编码器的单个层，包含：
    1. 多头自注意力机制
    2. 前馈神经网络
    3. 残差连接和层归一化
    
    Args:
        d_model (int): 模型维度
        num_heads (int): 注意力头数量
        d_ff (int): 前馈网络隐藏层维度
        dropout (float): Dropout概率，默认为0.1
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)  # 第一个层归一化
        self.norm2 = nn.LayerNorm(d_model)  # 第二个层归一化
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, seq_len, d_model)
            src_mask: 源序列掩码，用于屏蔽填充位置
            
        Returns:
            torch.Tensor: 编码器层输出 (batch_size, seq_len, d_model)
        """
        # 自注意力子层：残差连接 + 层归一化
        _x = x
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        
        # 前馈网络子层：残差连接 + 层归一化
        x = self.norm2(x + self.dropout(self.ff(x)))
        
        return x


class DecoderLayer(nn.Module):
    """
    解码器层
    
    实现Transformer解码器的单个层，包含：
    1. 多头自注意力机制（带因果掩码）
    2. 多头交叉注意力机制
    3. 前馈神经网络
    4. 残差连接和层归一化
    
    Args:
        d_model (int): 模型维度
        num_heads (int): 注意力头数量
        d_ff (int): 前馈网络隐藏层维度
        dropout (float): Dropout概率，默认为0.1
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)    # 自注意力
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)  # 交叉注意力
        self.ff = FeedForward(d_model, d_ff, dropout)                     # 前馈网络
        self.norm1 = nn.LayerNorm(d_model)  # 第一个层归一化
        self.norm2 = nn.LayerNorm(d_model)  # 第二个层归一化
        self.norm3 = nn.LayerNorm(d_model)  # 第三个层归一化
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_mask=None, memory_mask=None):
        """
        前向传播
        
        Args:
            x: 解码器输入张量 (batch_size, tgt_seq_len, d_model)
            enc_out: 编码器输出张量 (batch_size, src_seq_len, d_model)
            tgt_mask: 目标序列掩码（因果掩码）
            memory_mask: 记忆掩码（源序列掩码）
            
        Returns:
            torch.Tensor: 解码器层输出 (batch_size, tgt_seq_len, d_model)
        """
        # 自注意力子层：残差连接 + 层归一化
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        
        # 交叉注意力子层：残差连接 + 层归一化
        x = self.norm2(x + self.dropout(self.cross_attn(x, enc_out, enc_out, memory_mask)))
        
        # 前馈网络子层：残差连接 + 层归一化
        x = self.norm3(x + self.dropout(self.ff(x)))
        
        return x


class Encoder(nn.Module):
    """
    Transformer编码器
    
    实现完整的Transformer编码器，包含：
    1. 词嵌入层
    2. 位置编码
    3. 多个编码器层
    4. 最终的层归一化
    
    Args:
        vocab_size (int): 词汇表大小
        d_model (int): 模型维度，默认为256
        N (int): 编码器层数量，默认为2
        heads (int): 注意力头数量，默认为4
        d_ff (int): 前馈网络隐藏层维度，默认为512
        dropout (float): Dropout概率，默认为0.1
    """
    
    def __init__(self, vocab_size, d_model=256, N=2, heads=4, d_ff=512, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)  # 词嵌入层
        self.pos = PositionalEncoding(d_model)          # 位置编码
        self.layers = nn.ModuleList([EncoderLayer(d_model, heads, d_ff, dropout) for _ in range(N)])
        self.norm = nn.LayerNorm(d_model)               # 最终层归一化

    def forward(self, src, src_mask=None):
        """
        前向传播
        
        Args:
            src: 源序列张量 (batch_size, src_seq_len)
            src_mask: 源序列掩码
            
        Returns:
            torch.Tensor: 编码器输出 (batch_size, src_seq_len, d_model)
        """
        # 词嵌入并缩放
        x = self.embed(src) * math.sqrt(self.embed.embedding_dim)
        
        # 添加位置编码
        x = self.pos(x)
        
        # 通过所有编码器层
        for layer in self.layers:
            x = layer(x, src_mask)
        
        # 最终层归一化
        return self.norm(x)


class Decoder(nn.Module):
    """
    Transformer解码器
    
    实现完整的Transformer解码器，包含：
    1. 词嵌入层
    2. 位置编码
    3. 多个解码器层
    4. 最终的层归一化和输出投影
    
    Args:
        vocab_size (int): 词汇表大小
        d_model (int): 模型维度，默认为256
        N (int): 解码器层数量，默认为2
        heads (int): 注意力头数量，默认为4
        d_ff (int): 前馈网络隐藏层维度，默认为512
        dropout (float): Dropout概率，默认为0.1
    """
    
    def __init__(self, vocab_size, d_model=256, N=2, heads=4, d_ff=512, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)  # 词嵌入层
        self.pos = PositionalEncoding(d_model)          # 位置编码
        self.layers = nn.ModuleList([DecoderLayer(d_model, heads, d_ff, dropout) for _ in range(N)])
        self.norm = nn.LayerNorm(d_model)               # 最终层归一化
        self.out = nn.Linear(d_model, vocab_size)       # 输出投影层

    def forward(self, tgt, enc_out, tgt_mask=None, memory_mask=None):
        """
        前向传播
        
        Args:
            tgt: 目标序列张量 (batch_size, tgt_seq_len)
            enc_out: 编码器输出张量 (batch_size, src_seq_len, d_model)
            tgt_mask: 目标序列掩码（因果掩码）
            memory_mask: 记忆掩码（源序列掩码）
            
        Returns:
            torch.Tensor: 解码器输出 (batch_size, tgt_seq_len, vocab_size)
        """
        # 词嵌入并缩放
        x = self.embed(tgt) * math.sqrt(self.embed.embedding_dim)
        
        # 添加位置编码
        x = self.pos(x)
        
        # 通过所有解码器层
        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask, memory_mask)
        
        # 最终层归一化和输出投影
        x = self.norm(x)
        return self.out(x)


class Transformer(nn.Module):
    """
    完整的Transformer模型
    
    实现用于机器翻译的完整Transformer架构，包含编码器和解码器。
    支持序列到序列的翻译任务，如英德翻译。
    
    Args:
        src_vocab (int): 源语言词汇表大小
        tgt_vocab (int): 目标语言词汇表大小
        d_model (int): 模型维度，默认为256
        N (int): 编码器/解码器层数量，默认为2
        heads (int): 注意力头数量，默认为4
        d_ff (int): 前馈网络隐藏层维度，默认为512
        dropout (float): Dropout概率，默认为0.1
    """
    
    def __init__(self, src_vocab, tgt_vocab, d_model=256, N=2, heads=4, d_ff=512, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads, d_ff, dropout)
        self.decoder = Decoder(tgt_vocab, d_model, N, heads, d_ff, dropout)

    def make_pad_mask(self, seq):
        """
        创建填充掩码
        
        用于屏蔽序列中的填充位置（ID为0的位置），防止模型关注填充token。
        
        Args:
            seq: 输入序列张量 (batch_size, seq_len)
            
        Returns:
            torch.Tensor: 填充掩码 (batch_size, 1, 1, seq_len)
        """
        # 创建布尔掩码：非零位置为True，零位置为False
        return (seq != 0).unsqueeze(1).unsqueeze(1)  # (bs, 1, 1, seq_len)

    def make_subsequent_mask(self, size):
        """
        创建因果掩码（下三角掩码）
        
        用于解码器自注意力，确保模型在生成第i个token时只能看到前i-1个token，
        防止信息泄露，保证自回归生成的性质。
        
        Args:
            size (int): 序列长度
            
        Returns:
            torch.Tensor: 因果掩码 (1, 1, size, size)
        """
        # 创建上三角矩阵（对角线上方为1）
        mask = torch.triu(torch.ones(1, 1, size, size), diagonal=1).bool()
        # 取反得到下三角掩码（对角线和下方为True）
        return ~mask

    def forward(self, src, tgt):
        """
        前向传播
        
        Args:
            src: 源序列张量 (batch_size, src_seq_len)
            tgt: 目标序列张量 (batch_size, tgt_seq_len)
            
        Returns:
            torch.Tensor: 模型输出 (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        # 创建源序列填充掩码
        src_mask = self.make_pad_mask(src)
        
        # 创建目标序列掩码：填充掩码 AND 因果掩码
        tgt_mask = self.make_pad_mask(tgt) & self.make_subsequent_mask(tgt.size(1)).to(src.device)
        
        # 编码器前向传播
        enc = self.encoder(src, src_mask)
        
        # 解码器前向传播
        out = self.decoder(tgt, enc, tgt_mask, src_mask)
        
        return out
