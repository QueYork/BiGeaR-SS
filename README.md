# BiGeaR-SS
 
This is the PyTorch implementation for the paper:

"Learning Binarized Graph Representations With Dual Negative Sample Synthesis In Efficient Collaborative Filtering." *Chen Yankai, Que Yue, Chen Ma, and Irwin King.* TOIS'24

Download all ckpts: https://drive.google.com/drive/folders/1_-pSM804crmluP8PFl2RXO7LTNdd-g4v?usp=sharing

Linux cmd for running pre-training:
```
nohup python -u main_pretrain.py --dataset movie --lr 1e-3 --weight 1e-4 --epoch 40 --neg_ratio 8 --bpr_neg_num 4 --N 50 >/dev/null 2>&1 &
nohup python -u main_pretrain.py --dataset gowalla --lr 1e-3 --weight 5e-5 --epoch 20 --neg_ratio 36 --bpr_neg_num 4 --N 25 >/dev/null 2>&1 &
nohup python -u main_pretrain.py --dataset pinterest --lr 5e-4 --weight 1e-4 --epoch 25 --neg_ratio 8 --bpr_neg_num 4 --N 25 >/dev/null 2>&1 &
nohup python -u main_pretrain.py --dataset yelp --lr 5e-4 --weight 1e-4 --epoch 50 --neg_ratio 48 --bpr_neg_num 4 --N 25 >/dev/null 2>&1 &
nohup python -u main_pretrain.py --dataset book --lr 5e-4 --weight 1e-6 --epoch 100 --neg_ratio 48 --bpr_neg_num 4 --N 25 >/dev/null 2>&1 &
```

running quantization training:
```
nohup python -u main_quant.py --dataset movie --lr 1e-3 --weight 1e-4 --neg_ratio 8 --N 50 --epoch 1000 >/dev/null 2>&1 &
nohup python -u main_quant.py --dataset gowalla --lr 1e-3 --weight 5e-5 --epoch 800 --neg_ratio 4 --N 25 >/dev/null 2>&1 &
nohup python -u main_quant.py --dataset pinterest --lr 5e-4 --weight 1e-4 --epoch 150 --N 25 --neg_ratio 20 >/dev/null 2>&1 &
nohup python -u main_quant.py --dataset yelp --lr 5e-4 --weight 1e-4 --epoch 200 --N 25 --neg_ratio 20 >/dev/null 2>&1 &
nohup python -u main_quant.py --dataset book --lr 5e-4 --weight 1e-6 --epoch 400 --neg_ratio 2 --reg 0.01 --N 25 >/dev/null 2>&1 &
```