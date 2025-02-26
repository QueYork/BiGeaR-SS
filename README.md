# BiGeaR++
 
This is the PyTorch implementation for the paper:

"Learning Binarized Representations with Pseudo-positive Sample Enhancement for Efficient Graph Collaborative Filtering." *Yankai Chen, Yue Que, Xinning Zhang, Chen Ma, and Irwin King.*

This implementation is adapted from [BiGeaR](https://github.com/yankai-chen/BiGeaR), the predecessor of BiGeaR++.

## Environment Requirement

The code runs well under python 3.6. The required packages are referred to <b>env.txt</b>.

## To Start With

<li> <b>You can directly use our teacher embeddings for binarization</b>:
	
1. Firstly, download the embedding checkpoints via [link](https://drive.google.com/drive/folders/1_-pSM804crmluP8PFl2RXO7LTNdd-g4v?usp=sharing). Unzip them to the path "BiGeaR/src/checkpoints/".
	
2. Then, run the codes for each dataset. 
```
python main_quant.py --dataset movie --lr 1e-3 --weight 1e-4 --neg_ratio 8 --N 50 --epoch 1000
python main_quant.py --dataset gowalla --lr 1e-3 --weight 5e-5 --epoch 800 --neg_ratio 4 --N 25
python main_quant.py --dataset pinterest --lr 5e-4 --weight 1e-4 --epoch 150 --N 25 --neg_ratio 20
python main_quant.py --dataset yelp --lr 5e-4 --weight 1e-4 --epoch 200 --N 25 --neg_ratio 20
python main_quant.py --dataset book --lr 5e-4 --weight 1e-6 --epoch 400 --neg_ratio 2 --reg 0.01 --N 25
```

<li> <b>Alternatively</b>, 

1. You can also train the model from scratch to train the teacher embedding checkpoints for each dataset.  
```
python main_pretrain.py --dataset movie --lr 1e-3 --weight 1e-4 --epoch 40 --neg_ratio 8 --N 50
python main_pretrain.py --dataset gowalla --lr 1e-3 --weight 5e-5 --epoch 20 --neg_ratio 36 --N 25
python main_pretrain.py --dataset pinterest --lr 5e-4 --weight 1e-4 --epoch 25 --neg_ratio 8 --N 25
python main_pretrain.py --dataset yelp --lr 5e-4 --weight 1e-4 --epoch 50 --neg_ratio 48 --N 25
python main_pretrain.py --dataset book --lr 5e-4 --weight 1e-6 --epoch 100 --neg_ratio 48 --N 25
```

2. Then conduct binarization with <b>main_quant.py</b> similarly for each dataset.</li>


Thank you for your interest in our work!

