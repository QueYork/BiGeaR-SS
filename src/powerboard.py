"""
@author:chenyankai, queyue
@file:powerboard.py
@time:2024/6/28
"""
import os
import sys
from os.path import join
import torch
from src.parse import parse_args
from pprint import pprint

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()
CODE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = join(CODE_PATH, '../dataset')
C_SOURCE_PATH = join(CODE_PATH, 'sources')
BOARD_PATH = join(CODE_PATH, 'results')
FILE_PATH = join(CODE_PATH, 'checkpoints')
sys.path.append(C_SOURCE_PATH)

if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)


all_dataset = ['gowalla', 'yelp', 'book', 'movie', 'pinterest']
batch_ranking_list = []
all_model = ['bgr']
GPU = torch.cuda.is_available()
SEED = args.seed
DEVICE = torch.device('cuda' if GPU else 'cpu')
norm_a = args.norm_a

if args.dataset not in all_dataset:
    raise NotImplementedError(f"Haven't supported {args.dataset} yet!, try {all_dataset}")

from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=RuntimeWarning)


def cprint(words: str):
    print(f"\033[0;30;43m{words}\033[0m")


print('===========config================')
pprint(args)
print('===========end===================')