ps -ef | grep tensorboard | awk '{print $2}' | xargs kill -9 
tensorboard --port 6007 --logdir /root/bigear/src/results/

负采样 & BPR
nohup python -u main_pretrain.py --dataset movie --lr 1e-3 --weight 1e-4 --epoch 40 --neg_ratio 8 --bpr_neg_num 5 --alpha 2 --N 50 >/dev/null 2>&1 &
nohup python -u main_pretrain.py --dataset gowalla --lr 1e-3 --weight 5e-5 --epoch 20 --neg_ratio 20 --bpr_neg_num 4 --alpha 0.3 --N 100 >/dev/null 2>&1 &
nohup python -u main_pretrain.py --dataset pinterest --lr 5e-4 --weight 1e-4 --epoch 25 --neg_ratio 32 --bpr_neg_num 12 --alpha 0.7 --N 25 >/dev/null 2>&1 &
nohup python -u main_pretrain.py --dataset yelp --lr 5e-4 --weight 1e-4 --epoch 50 --neg_ratio 42 --bpr_neg_num 4 --alpha 0.1 --N 25 >/dev/null 2>&1 &
nohup python -u main_pretrain.py --dataset book --lr 5e-4 --weight 1e-6 --epoch 100 --neg_ratio 36 --bpr_neg_num 4 >/dev/null 2>&1 &

正常 quant
nohup python -u main_quant.py --dataset movie --lr 1e-3 --weight 1e-4 --alpha 0.5 --neg_ratio 2 --N 50 --epoch 1000 >/dev/null 2>&1 &
nohup python -u main_quant.py --dataset gowalla --lr 1e-3 --weight 5e-5 --epoch 1000 --alpha 0.1 --neg_ratio 4 --N 25 >/dev/null 2>&1 &
nohup python -u main_quant.py --dataset pinterest --lr 5e-4 --weight 1e-4 --epoch 60 --N 25 --neg_ratio 16 --alpha 0.5 >/dev/null 2>&1 &
nohup python -u main_quant.py --dataset yelp --lr 5e-4 --weight 1e-4 --epoch 400 --N 25 --neg_ratio 4 --alpha 0.1 >/dev/null 2>&1 &
nohup python -u main_quant.py --dataset book --lr 5e-4 --weight 1e-6 --epoch 400 >/dev/null 2>&1 &

data_loader: 数据集导入 & train test split & 构建稀疏矩阵
    LoadData类: 封装了用到的所有数据集操作

model: 各个模型类
    BasicModel 类: 基类模板

    BiGeaR_tch 类: Teacher 类，基础全精度 lightGCN-based 模型，包含 tch loss 计算
        summary 函数用于创建每层的 top-R items 以供蒸馏

    Normal_Ddelta 类: 自定义二值化微分，使用正态分布近似 Dirac delta 函数
        继承自 torch.autograd, 相当于自定义了 torch 求微分的方式，包含 forward 和 backward 过程
        forward 进行二值化操作
        backward 计算梯度

    Quant_layer 类: obtain binarized embedding layer (apply forward in Normal_Ddelta)
    
    BiGeaR 类: 主模型类，包含蒸馏、同 tch 的交互、quant loss计算

evals: tch & std loss 传播、全精度 & 半精度训练、测试集预测
    BGRLoss_tch 类: teacher loss 传播

    BGRLoss_quant 类: student loss 传播

    Train_full 函数: 采样过程！

    Train_quant 函数: 采样过程！

    Inference 函数：prediction & performance evaluation

main_pretrain: 全精度预训练的 main 函数（调用 Train_full）
    search 函数: 找到最佳的迭代次数

    pretrain 函数：正式预训练

main_quant: 半精度训练的 main 函数 (调用 Train_quant)
    quant 函数：训练 

utils: 各种功能依赖函数 & metrics 函数 

