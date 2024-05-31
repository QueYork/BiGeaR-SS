"""
@author:chenyankai
@file:parse.py
@time:2021/11/10
"""
import argparse

def parse_args():
    parse = argparse.ArgumentParser(description='Lossless')
    
    # 数据集
    parse.add_argument('--dataset', type=str, default='movie', help='accessible datasets from [movie, gowalla, yelp, book, pinterest]')
    
    # 测试时测 top-几 ，数组形式 [20, 40, 60, 80, 100]
    parse.add_argument('--topks', nargs='+', type=int, default=[20, 100], help='top@k test list')      
    
    # 训练集/测试集 信息
    parse.add_argument('--train_file', type=str, default='train.txt')
    parse.add_argument('--test_file', type=str, default='test.txt')
    parse.add_argument('--train_batch', type=int, default=2048, help='batch size in training')
    parse.add_argument('--test_batch', type=int, default=100, help='batch size in testing')
    
    # hop 层数
    parse.add_argument('--layers', type=int, default=2, help='the layer number')
    
    # 层内 embedding 维度
    parse.add_argument('--dim', type=int, default=256, help='embedding dimension')
    
    # 学习率
    parse.add_argument('--lr', type=float, default=0.001, help='learning rate')
    
    # L2 正则化 λ
    parse.add_argument('--weight', type=float, default=1e-4, help='the weight of l2 norm')     
    
    # Tensorboard 开关
    parse.add_argument('--tensorboard', type=bool, default=True, help='enable tensorboard')
    
    # 迭代数
    parse.add_argument('--epoch', type=int, default=20)
    
    # 随机种子
    parse.add_argument('--seed', type=int, default=2021, help='random seed')
    
    # 模型
    parse.add_argument('--model', type=str, default='bgr', help='models to be trained from [lossless]')
    
    # 保存参数开关
    parse.add_argument('--save_embed', type=int, default=1, help='save embedding or not')
    
    # w_k 计算中的 λ_2
    parse.add_argument('--lmd2', type=float, default=0.1, help='lambda of for weighting the ranking distillation')
    
    # R
    parse.add_argument('--R', type=int, default=100, help='Top-R of ranking distillation.')
    
    # 计算全精度 top-R 开关
    parse.add_argument('--compute_rank', type=int, default=1)  
                              
    # 用于 Normal_Ddelta 计算中
    parse.add_argument('--norm_a', type=float, default=1., help='normal distribution')                           
    
    # 负采样个数
    parse.add_argument('--neg_ratio', type=int, default=8, help='the ratio of negative sampling')
    
    parse.add_argument('--bpr_neg_num', type=int, default=1, help='number of negative samples included in K-pair BPR loss function')
    
    parse.add_argument('--eps', type=float, default=1e-20, help='epsilon in gumbel sampling')
    
    # It regulates the overall inclination of the synthesized hard negative examples to approach the positive examples.
    parse.add_argument("--alpha", type=float, default='0.5', help="Control the trend of the model to positive contributation")
    
    # ArgumentParser 通过 parse_args() 方法解析参数，获取到命令行中输入的参数，需在命令行中进行调用才可正常运行
    return parse.parse_args() 