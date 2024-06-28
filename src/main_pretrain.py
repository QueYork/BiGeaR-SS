"""
@author:chenyankai, queyue
@file:main.py
@time:2024/6/28
"""
import os
import sys
from os.path import join
import warnings
warnings.filterwarnings('ignore')

PATH = os.path.dirname(os.path.abspath(__file__))
ROOT = join(PATH, '../')
sys.path.append(ROOT)

from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
import src.data_loader as data_loader
import datetime
import pytz
import torch
import logging
import src.powerboard as board
import src.utils as utils
import src.evals as evals
import src.model as model

MODEL = {
    'bgr': model.BiGeaR_tch,
}
LOSS_F = {
    'bgr': evals.BGRLoss_tch,
}

def pretrain():
    utils.set_seed(board.SEED)
    print('--SEED--:', board.SEED)

    dataset = data_loader.LoadData(data_name=board.args.dataset)
    model = MODEL[board.args.model](dataset=dataset)
    model = model.to(board.DEVICE)
    loss_f = LOSS_F[board.args.model](model)

    # log file path
    path = join(board.BOARD_PATH, board.args.dataset)
    timezone = pytz.timezone('Asia/Shanghai')
    nowtime = datetime.datetime.now(tz=timezone)
    log_path = join(path, nowtime.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + board.args.model)

    # init tensorboard
    if board.args.tensorboard:
        summarizer: SummaryWriter = SummaryWriter(log_path)
    else:
        summarizer = None
        board.cprint('tensorboard disabled.')

    try:
        max_recall20 = 0

        # logger initializer
        log_name = utils.create_log_name(log_path)
        utils.log_config(path=log_path, name=log_name, level=logging.DEBUG, console_level=logging.DEBUG, console=True)
        logging.info(board.args)

        for epoch in range(board.args.epoch):
            info = evals.Train_full(dataset=dataset, model=model, epoch=epoch, loss_f=loss_f,
                                    neg_ratio=board.args.neg_ratio, summarizer=summarizer)

            board.cprint(f'[testing at epoch-{epoch}]')
            results = evals.Inference(dataset=dataset, model=model, epoch=epoch, summarizer=summarizer)

            logging.info(f'[testing at epoch-{epoch}]')
            logging.info(results)

            logging.info(f'EPOCH[{epoch + 1}/{board.args.epoch}] {info} ')
            
            if max_recall20 < results['recall'][0]:
                max_recall20 = results['recall'][0]
                logging.info(f'Summary at recall = {max_recall20}')
                
                if board.args.compute_rank == 1:
                    file = f"{board.args.model}-{board.args.dataset}-{board.args.dim}-rank.pth.tar"
                    weight_file = os.path.join(board.FILE_PATH, file)

                    model.summary()
                    
                    torch.save({'RP_embed': model.RP_embed,
                                'pos_rank_LW': model.pos_rank,
                                'neg_rank_LW': model.neg_rank,
                                'hard_rank_LW': model.hard_neg_rank  
                                },
                               weight_file)
                else:
                    file = f"{board.args.model}-{board.args.dataset}-{board.args.bin_dim}.pth.tar"
                    weight_file = os.path.join(board.FILE_PATH, file)

                    torch.save({'RP_embed': model.RP_embed},
                               weight_file)

    except:
        raise NotImplementedError('Error in running main file')

    finally:
        if board.args.tensorboard:
            summarizer.close()

if __name__ == '__main__':
    pretrain()