"""
机器翻译数据集处理模块

该模块实现了用于英德翻译的PyTorch数据集类，支持SentencePiece分词器的加载和使用。
主要功能包括：
1. 加载和预处理英德平行语料
2. 使用SentencePiece进行分词编码
3. 提供批次数据整理功能
"""

import os

import sentencepiece as spm
import torch
from torch.utils.data import Dataset


class TranslationDataset(Dataset):
    """
    英德翻译数据集类
    
    该类继承自PyTorch的Dataset，用于处理英德机器翻译的平行语料数据。
    支持SentencePiece分词器的加载和使用，提供数据预处理和批次整理功能。
    
    Args:
        data_dir (str): 数据目录路径，包含训练数据文件
        spm_en (str): 英语SentencePiece模型文件路径
        spm_de (str): 德语SentencePiece模型文件路径
        split (str): 数据集分割类型，默认为'train'
        max_examples (int, optional): 最大样本数量限制，用于快速测试
        max_len (int): 序列最大长度，超过此长度的序列会被截断
    """
    
    def __init__(self, data_dir, spm_en, spm_de, split='train', max_examples=None, max_len=128):
        # 构建源语言和目标语言数据文件路径
        self.src_file = os.path.join(data_dir, f"train.tags.en-de.en")  # 英语源文件
        self.tgt_file = os.path.join(data_dir, f"train.tags.en-de.de")  # 德语目标文件
        
        # 读取源语言和目标语言文本数据
        with open(self.src_file, 'r', encoding='utf-8') as f:
            src_lines = f.readlines()
        with open(self.tgt_file, 'r', encoding='utf-8') as f:
            tgt_lines = f.readlines()
        
        # 数据预处理：过滤空行和XML标签
        pairs = []
        for s, t in zip(src_lines, tgt_lines):
            s = s.strip()  # 去除首尾空白字符
            t = t.strip()
            
            # 跳过空行
            if s == '' or t == '':
                continue
            
            # 跳过XML标签行（如<s>、</s>等）
            if s.startswith('<') or t.startswith('<'):
                continue
            
            # 添加有效的平行语料对
            pairs.append((s, t))
        
        # 如果设置了最大样本数限制，则截取前N个样本（用于快速测试）
        if max_examples:
            pairs = pairs[:max_examples]
        
        self.pairs = pairs  # 存储所有有效的平行语料对
        
        # 初始化SentencePiece分词器
        self.sp_en = spm.SentencePieceProcessor()  # 英语分词器
        self.sp_de = spm.SentencePieceProcessor()  # 德语分词器
        
        # 加载预训练的SentencePiece模型
        self.sp_en.Load(spm_en)
        self.sp_de.Load(spm_de)
        
        self.max_len = max_len  # 序列最大长度限制

    def __len__(self):
        """
        返回数据集中的样本总数
        
        Returns:
            int: 数据集中的平行语料对数量
        """
        return len(self.pairs)

    def encode(self, text, sp, bos_id=1, eos_id=2):
        """
        使用SentencePiece分词器将文本编码为ID序列
        
        Args:
            text (str): 待编码的文本
            sp: SentencePiece分词器实例
            bos_id (int): 句子开始标记的ID，默认为1
            eos_id (int): 句子结束标记的ID，默认为2
            
        Returns:
            list: 包含BOS、编码ID和EOS的完整序列
        """
        # 使用SentencePiece将文本编码为ID列表
        ids = sp.EncodeAsIds(text)
        
        # 截断过长的序列，保留空间给BOS和EOS标记
        ids = ids[:self.max_len-2]
        
        # 添加句子开始和结束标记
        return [bos_id] + ids + [eos_id]

    def __getitem__(self, idx):
        """
        获取指定索引的数据样本
        
        Args:
            idx (int): 样本索引
            
        Returns:
            tuple: (源语言张量, 目标语言张量) 的元组
        """
        # 获取指定索引的平行语料对
        s, t = self.pairs[idx]
        
        # 分别对源语言和目标语言进行编码
        src_ids = self.encode(s, self.sp_en)  # 英语编码
        tgt_ids = self.encode(t, self.sp_de)  # 德语编码
        
        # 转换为PyTorch张量并返回
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)

    @staticmethod
    def collate_fn(batch):
        """
        批次数据整理函数，用于DataLoader
        
        该函数将不同长度的序列填充到相同长度，以便进行批次处理。
        使用零填充（padding）来处理变长序列。
        
        Args:
            batch (list): 包含多个(src_tensor, tgt_tensor)元组的列表
            
        Returns:
            tuple: (填充后的源语言批次张量, 填充后的目标语言批次张量)
        """
        # 分离源语言和目标语言数据
        srcs, tgts = zip(*batch)
        
        # 计算每个序列的长度
        src_lens = [len(x) for x in srcs]
        tgt_lens = [len(x) for x in tgts]
        
        # 找到批次中的最大长度
        max_src = max(src_lens)
        max_tgt = max(tgt_lens)
        
        # 创建填充后的张量（初始化为0）
        padded_src = torch.zeros(len(batch), max_src, dtype=torch.long)
        padded_tgt = torch.zeros(len(batch), max_tgt, dtype=torch.long)
        
        # 将实际数据复制到填充张量中
        for i, s in enumerate(srcs):
            padded_src[i, :len(s)] = s
        for i, t in enumerate(tgts):
            padded_tgt[i, :len(t)] = t
        
        return padded_src, padded_tgt
