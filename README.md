# BiGeaR
 
This is the PyTorch implementation for the paper:

"Learning Binarized Graph Representations with Multi-faceted Quantization Reinforcement for Top-K Recommendation." *Chen, Yankai, Huifeng Guo, Yingxue Zhang, Chen Ma, Ruiming Tang, Jingjie Li, and Irwin King.* SIGKDD'22.
It is available in [ACM digital library](https://dl.acm.org/doi/abs/10.1145/3534678.3539452).



## Environment Requirement

The code runs well under python 3.6. The required packages are referred to <b>env.txt</b>.


## Datasets

All datasets are available via [link](https://drive.google.com/file/d/11RrEMaQ5zlChzUj7VteI4Kolclz7Hr-r/view?usp=sharing). Upzip these datasets to the folder "BiGeaR/dataset/". 

## To Start With

<li> <b>You can directly use our teacher embeddings for binarization</b>:
	
1. Firstly, download the embedding checkpoints via [link](https://drive.google.com/file/d/1nGMvAegcfcvErV90mgAUOteWhhzptGPS/view?usp=sharing). Unzip them to the path "BiGeaR/src/checkpoints/".
	
2. Then, run the codes for each dataset. For example, for movie dataset:
```

python main_quant.py --epoch 1000 --dataset movie --model bgr --dim 256 --save_embed 1 --compute_rank 1 --lr 1e-3 --weight 1e-4

```
3. Please refer to <b>run_bin.py</b> for other dataset settings. </li>


<li> <b>Alternatively</b>, 

1. You can also train the model from scratch to train the teacher embedding checkpoints for dataset xxx:  

```

python main_pretrain.py --epoch 1000 --dataset xxx --model bgr --dim 256 --save_embed 1 --compute_rank 1

```

2. Then conduct binarization with <b>main_quant.py</b> similarly for dataset xxx.</li>


## Citation
Please kindly cite our paper if you find this code useful for your research:

```
@inproceedings{chen2022learning,
  author={Chen, Yankai and Guo, Huifeng and Zhang, Yingxue and Ma, Chen and Tang, Ruiming and Li, Jingjie and King, Irwin},
  title     = {Learning Binarized Graph Representations with Multi-faceted Quantization Reinforcement for Top-K Recommendation},
  booktitle = {{KDD} '22: The 28th {ACM} {SIGKDD} Conference on Knowledge Discovery
               and Data Mining, Washington, DC, USA, August 14 - 18, 2022},
  pages     = {168--178},
  publisher = {{ACM}},
  year      = {2022}
}

```
