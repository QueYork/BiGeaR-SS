import os
import subprocess

command = "python main_quant.py --epoch 1000 --dataset movie --model bgr --dim 256 " \
          " --save_embed 1 --compute_rank 1 --lr 1e-3 --weight 1e-4  "
print('Running', command)
subprocess.call(command, shell=True)


command = "python main_quant.py --epoch 1000 --dataset gowalla --model bgr --dim 256 " \
          " --save_embed 1 --compute_rank 1 --lr 1e-3 --weight 5e-5  "
print('Running', command)
subprocess.call(command, shell=True)

command = "python main_quant.py --epoch 1000 --dataset pinterest --model bgr --dim 256 " \
          " --save_embed 1 --compute_rank 1 --lr 5e-4 --weight 1e-4 "
print('Running', command)
subprocess.call(command, shell=True)

command = "python main_quant.py --epoch 1000 --dataset yelp --model bgr --dim 256 " \
          " --save_embed 1 --compute_rank 1 --lr 5e-4 --weight 1e-4  "
print('Running', command)
subprocess.call(command, shell=True)


command = "python main_quant.py --epoch 1000 --dataset book --model bgr --dim 256 " \
          " --save_embed 1 --compute_rank 1 --lr 5e-4 --weight 1e-6  "
print('Running', command)
subprocess.call(command, shell=True)


