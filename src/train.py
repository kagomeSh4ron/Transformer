"""
实现英德翻译的训练流程，包括：
1. 数据加载和预处理
2. 模型初始化和优化器设置
3. 训练循环和损失计算
4. 模型保存和翻译样本生成

"""

import argparse
import os

import torch
import torch.optim as optim
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dataset import TranslationDataset
from model import Transformer

# 禁用matplotlib导入以避免在某些环境中的NumPy C-extension ABI问题
# 使用文本回退保存器代替
_HAS_MATPLOTLIB = False
plt = None
from tqdm import tqdm


def save_plot(losses, path):
    """
    保存损失值图表
    
    如果matplotlib可用则保存为PNG图片，否则保存为文本文件。
    这样可以避免在某些环境中因matplotlib依赖问题导致的错误。
    
    Args:
        losses (List[float]): 损失值列表
        path (str): PNG文件的目标路径或TXT文件路径（回退方案）
    """
    if _HAS_MATPLOTLIB:
        # 使用matplotlib绘制损失曲线
        plt.figure()
        plt.plot(losses)
        plt.xlabel('steps')  # X轴标签：训练步数
        plt.ylabel('loss')  # Y轴标签：损失值
        plt.savefig(path)
    else:
        # 将损失值写入文本文件（作为PNG的替代方案）
        txt_path = os.path.splitext(path)[0] + '.txt'
        with open(txt_path, 'w', encoding='utf-8') as f:
            for i, v in enumerate(losses):
                f.write(f"{i}\t{v}\n")


def collate_fn(batch):
    """
    批次数据整理函数
    
    调用TranslationDataset的静态方法进行批次数据整理，
    将不同长度的序列填充到相同长度。
    
    Args:
        batch: 批次数据列表
        
    Returns:
        tuple: 整理后的批次数据
    """
    return TranslationDataset.collate_fn(batch)


def main():
    """
    主训练函数
    
    执行完整的训练流程：
    1. 解析命令行参数
    2. 初始化数据集和数据加载器
    3. 创建模型和优化器
    4. 执行训练循环
    5. 保存模型和生成翻译样本
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练Transformer机器翻译模型')
    parser.add_argument('--data_dir', default='en-de', help='数据目录路径')
    parser.add_argument('--spm_en', required=True, help='英语SentencePiece模型文件路径')
    parser.add_argument('--spm_de', required=True, help='德语SentencePiece模型文件路径')
    parser.add_argument('--epochs', type=int, default=2, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--max_examples', type=int, default=1000, help='最大训练样本数（用于快速测试）')
    parser.add_argument('--output_dir', type=str, default='results', help='输出目录')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    args = parser.parse_args()

    # 设备选择：优先使用GPU，否则使用CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)

    # 初始化数据集和数据加载器
    print("加载数据集...")
    dataset = TranslationDataset(args.data_dir, args.spm_en, args.spm_de, max_examples=args.max_examples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    # 获取词汇表大小
    src_vocab = dataset.sp_en.get_piece_size()  # 英语词汇表大小
    tgt_vocab = dataset.sp_de.get_piece_size()  # 德语词汇表大小
    
    print(f"英语词汇表大小: {src_vocab}")
    print(f"德语词汇表大小: {tgt_vocab}")
    print(f"训练样本数量: {len(dataset)}")

    # 初始化模型
    print("初始化模型...")
    model = Transformer(src_vocab, tgt_vocab)
    model = model.to(device)
    
    # 初始化优化器和损失函数
    optimizer = AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)  # 忽略填充token的损失

    # 训练循环
    losses = []
    step = 0
    model.train()
    
    print(f"开始训练，共{args.epochs}个epoch...")
    for epoch in range(args.epochs):
        bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for src, tgt in bar:
            # 将数据移动到指定设备
            src = src.to(device)
            tgt = tgt.to(device)
            
            # Teacher Forcing: 输入tgt[:-1]，目标tgt[1:]
            # 这样可以让模型在训练时看到正确的历史信息
            inp = tgt[:, :-1]    # 输入：去掉最后一个token
            target = tgt[:, 1:]  # 目标：去掉第一个token
            
            # 前向传播
            out = model(src, inp)  # 输出形状: (batch_size, seq_len, vocab_size)
            
            # 计算损失
            out_flat = out.reshape(-1, out.size(-1))  # 展平为2D张量
            target_flat = target.reshape(-1)           # 展平为1D张量
            loss = criterion(out_flat, target_flat)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪：防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # 更新参数
            optimizer.step()
            
            # 记录损失
            losses.append(loss.item())
            step += 1
            
            # 更新进度条
            bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # 保存模型检查点和损失图表
    print("保存模型...")
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'checkpoints', 'model.pt'))
    save_plot(losses, os.path.join(args.output_dir, 'loss.png'))
    
    # 生成翻译样本
    print("生成翻译样本...")
    model.eval()
    samples = []
    
    with torch.no_grad():
        # 对前200个样本进行翻译（或全部样本，取较小值）
        for i in range(min(200, len(dataset))):
            src_ids, tgt_ids = dataset[i]
            src = src_ids.unsqueeze(0).to(device)  # 添加批次维度
            
            # 贪心解码：逐步生成翻译
            ys = torch.tensor([1], dtype=torch.long).unsqueeze(0).to(device)  # 开始标记(BOS)
            
            for _ in range(80):  # 最大生成长度限制
                out = model(src, ys)
                prob = out[:, -1, :]  # 取最后一个位置的输出概率
                _, next_tok = prob.max(-1)  # 贪心选择概率最高的token
                ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)  # 添加到序列中
                
                # 如果生成结束标记，停止生成
                if next_tok.item() == 2:  # EOS标记
                    break
            
            # 解码为文本
            src_text = dataset.sp_en.DecodeIds(src_ids.tolist()[1:-1])  # 去掉BOS和EOS
            pred_text = dataset.sp_de.DecodeIds(ys.squeeze(0).tolist()[1:-1])
            tgt_text = dataset.sp_de.DecodeIds(tgt_ids.tolist()[1:-1])
            
            samples.append((src_text, pred_text, tgt_text))
    
    # 保存翻译样本到文件
    with open(os.path.join(args.output_dir, 'sample_translations.txt'), 'w', encoding='utf-8') as f:
        for s, p, t in samples:
            f.write('SRC:\t' + s + '\n')    # 源语言
            f.write('PRED:\t' + p + '\n')   # 预测翻译
            f.write('TGT:\t' + t + '\n\n')  # 目标翻译
    
    print(f"训练完成！输出文件保存在: {args.output_dir}")


if __name__ == '__main__':
    main()
