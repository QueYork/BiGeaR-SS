
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

