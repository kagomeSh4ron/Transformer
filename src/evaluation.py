"""
机器翻译模型评估模块

该模块实现了基于BLEU指标的机器翻译质量评估功能，包括：
1. 翻译数据加载和解析
2. 多语言文本分词处理
3. BLEU-1到BLEU-4分数计算
4. 评估结果保存和输出

使用NLTK库进行BLEU分数计算，支持平滑处理以避免短句子分数异常。
"""

import json
import re
from datetime import datetime

import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

# 下载nltk所需资源（首次运行需下载，后续可注释）
# 这些资源用于文本分词和BLEU分数计算
nltk.download('punkt')


def load_translation_data(file_path):
    """
    从文本文件加载翻译数据
    
    解析训练脚本生成的翻译结果文件，提取源语言、预测翻译和目标翻译。
    文件格式要求：每行按 "SRC: ..." "PRED: ..." "TGT: ..." 交替出现，
    一组样本包含SRC、PRED、TGT各一行。
    
    Args:
        file_path (str): 翻译结果文件路径
        
    Returns:
        list: 包含翻译数据的字典列表，每个字典包含'src'、'pred'、'tgt'字段
    """
    data = []
    
    # 读取文件内容，过滤空行
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # 按组解析（每组3行：SRC、PRED、TGT）
    for i in range(0, len(lines), 3):
        if i + 2 >= len(lines):
            break  # 忽略不完整的组
        
        src_line = lines[i]      # 源语言行
        pred_line = lines[i+1]  # 预测翻译行
        tgt_line = lines[i+2]   # 目标翻译行
        
        # 提取冒号后的内容（去除前缀标签）
        src = re.sub(r'^SRC:\s*', '', src_line)    # 去除"SRC:"前缀
        pred = re.sub(r'^PRED:\s*', '', pred_line) # 去除"PRED:"前缀
        tgt = re.sub(r'^TGT:\s*', '', tgt_line)   # 去除"TGT:"前缀
        
        # 添加到数据列表
        data.append({
            'src': src,
            'pred': pred,
            'tgt': tgt
        })
    
    return data


def tokenize(text, language='german'):
    """
    文本分词函数
    
    使用NLTK的word_tokenize进行分词，支持多语言处理。
    这里主要针对德语和英语进行分词。
    
    Args:
        text (str): 待分词的文本
        language (str): 语言类型，默认为'german'
        
    Returns:
        list: 分词后的token列表
    """
    # 英语和德语可直接用nltk的word_tokenize
    return word_tokenize(text, language=language.lower())


def calculate_bleu_scores(data):
    """
    计算批量样本的BLEU分数
    
    计算BLEU-1到BLEU-4的分数以及平均分数。
    使用平滑处理避免短句子分数异常，这是BLEU评估的标准做法。
    
    Args:
        data (list): 包含翻译数据的字典列表
        
    Returns:
        dict: 包含各种BLEU分数的字典
            - avg_bleu1: BLEU-1平均分数
            - avg_bleu2: BLEU-2平均分数  
            - avg_bleu3: BLEU-3平均分数
            - avg_bleu4: BLEU-4平均分数
            - sample_count: 样本数量
    """
    # 使用平滑处理（避免短句子分数异常）
    smoothing = SmoothingFunction().method4
    
    # 初始化分数列表
    bleu1_scores = []
    bleu2_scores = []
    bleu3_scores = []
    bleu4_scores = []
    
    # 计算每个样本的BLEU分数
    for item in data:
        pred = tokenize(item['pred'])  # 预测翻译分词
        tgt = tokenize(item['tgt'])    # 目标翻译分词
        
        # 参考译文需用列表包裹（支持多个参考译文，这里只有1个）
        references = [tgt]
        
        # 计算不同n-gram的BLEU分数
        bleu1 = sentence_bleu(references, pred, weights=(1, 0, 0, 0), smoothing_function=smoothing)
        bleu2 = sentence_bleu(references, pred, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
        bleu3 = sentence_bleu(references, pred, weights=(0.33, 0.33, 0.34, 0), smoothing_function=smoothing)
        bleu4 = sentence_bleu(references, pred, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
        
        # 添加到分数列表
        bleu1_scores.append(bleu1)
        bleu2_scores.append(bleu2)
        bleu3_scores.append(bleu3)
        bleu4_scores.append(bleu4)
    
    # 计算平均分数（转换为百分比）
    avg_bleu1 = sum(bleu1_scores) / len(bleu1_scores) * 100
    avg_bleu2 = sum(bleu2_scores) / len(bleu2_scores) * 100
    avg_bleu3 = sum(bleu3_scores) / len(bleu3_scores) * 100
    avg_bleu4 = sum(bleu4_scores) / len(bleu4_scores) * 100
    
    return {
        'avg_bleu1': round(avg_bleu1, 2),
        'avg_bleu2': round(avg_bleu2, 2),
        'avg_bleu3': round(avg_bleu3, 2),
        'avg_bleu4': round(avg_bleu4, 2),
        'sample_count': len(data)
    }


def main():
    """
    主评估函数
    
    执行完整的评估流程：
    1. 加载翻译结果数据
    2. 计算BLEU分数
    3. 保存评估结果到JSON文件
    4. 输出评估摘要
    """
    # 翻译结果文件路径
    file_path = 'results/sample_translations.txt'
    
    print("开始评估机器翻译模型...")
    print(f"加载翻译数据: {file_path}")
    
    # 加载翻译数据
    data = load_translation_data(file_path)
    if not data:
        print("未找到有效数据，请检查文件格式！")
        return
    
    print(f"成功加载 {len(data)} 个翻译样本")
    
    # 计算BLEU分数
    print("计算BLEU分数...")
    results = calculate_bleu_scores(data)
    
    # 输出评估结果摘要
    print("\n=== 评估结果摘要 ===")
    print(f"样本数量: {results['sample_count']}")
    print(f"BLEU-1: {results['avg_bleu1']:.2f}%")
    print(f"BLEU-2: {results['avg_bleu2']:.2f}%")
    print(f"BLEU-3: {results['avg_bleu3']:.2f}%")
    print(f"BLEU-4: {results['avg_bleu4']:.2f}%")
    
    # 准备保存的数据
    output_data = {
        'evaluation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),  # 评估时间
        'sample_count': results['sample_count'],                           # 样本数量
        'bleu_scores': {
            'BLEU-1': results['avg_bleu1'],
            'BLEU-2': results['avg_bleu2'],
            'BLEU-3': results['avg_bleu3'],
            'BLEU-4': results['avg_bleu4']
        }
    }
    
    # 保存为JSON格式
    json_file_path = 'results/bleu_scores.json'
    print(f"\n保存评估结果到: {json_file_path}")
    
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print("评估完成！")
    print(f"\n结果已保存到:")
    print(f"- JSON格式: {json_file_path}")


if __name__ == "__main__":
    main()