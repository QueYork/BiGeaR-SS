"""
@author:chenyankai, queyue
@file:parse.py
@time:2024/6/28
"""
import argparse

def parse_args():
    parse = argparse.ArgumentParser(description='Lossless')

    parse.add_argument('--dataset', type=str, default='movie', help='accessible datasets from [movie, gowalla, yelp, book, pinterest]')
    parse.add_argument('--topks', nargs='+', type=int, default=[20, 100], help='top@k test list')      
    parse.add_argument('--train_file', type=str, default='train.txt')
    parse.add_argument('--test_file', type=str, default='test.txt')
    parse.add_argument('--train_batch', type=int, default=2048, help='batch size in training')
    parse.add_argument('--test_batch', type=int, default=100, help='batch size in testing')
    parse.add_argument('--layers', type=int, default=2, help='the layer number')
    parse.add_argument('--dim', type=int, default=256, help='embedding dimension')
    parse.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parse.add_argument('--weight', type=float, default=1e-4, help='the weight of l2 norm')     
    parse.add_argument('--tensorboard', type=bool, default=True, help='enable tensorboard')
    parse.add_argument('--epoch', type=int, default=20)
    parse.add_argument('--seed', type=int, default=2021, help='random seed')
    parse.add_argument('--model', type=str, default='bgr', help='models to be trained from [lossless]')
    parse.add_argument('--save_embed', type=int, default=1, help='save embedding or not')
    parse.add_argument('--lmd2', type=float, default=0.1, help='lambda of for weighting the ranking distillation')
    parse.add_argument('--R', type=int, default=100, help='Top-R of pos ranking distillation .')
    parse.add_argument('--N', type=int, default=100, help='Top-N of neg ranking distillation.')
    parse.add_argument('--compute_rank', type=int, default=1)  
    parse.add_argument('--norm_a', type=float, default=1., help='normal distribution')                           
    parse.add_argument('--neg_ratio', type=int, default=8, help='the ratio of negative sampling')
    parse.add_argument('--bpr_neg_num', type=int, default=1, help='number of negative samples included in K-pair BPR loss function')
    parse.add_argument('--eps', type=float, default=1e-20, help='epsilon in gumbel sampling')
    parse.add_argument("--reg", type=float, default='1.0', help="Control the extent of positive mixing")
    
    return parse.parse_args() 