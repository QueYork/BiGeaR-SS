"""
@author:chenyankai
@file:main.py
@time:2021/11/11
"""
import os
import sys
from os.path import join

PATH = os.path.dirname(os.path.abspath(__file__))
ROOT = join(PATH, '../')
sys.path.append(ROOT)

from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
import src.data_loader as data_loader
import datetime
import pytz
import logging
import src.powerboard as board
import src.utils as utils
import src.evals as evals
import src.model as model

MODEL = {
    'bgr': model.BiGeaR
}
LOSS_F = {
    'bgr': evals.BGRLoss_quant
}

def quant():
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
        max_recall20 = []
        # logger initializer
        log_name = utils.create_log_name(log_path)
        utils.log_config(path=log_path, name=log_name, level=logging.DEBUG, console_level=logging.DEBUG, console=True)
        logging.info(board.args)

        for epoch in range(board.args.epoch):

            info = evals.Train_quant(dataset=dataset, model=model, epoch=epoch, loss_f=loss_f,
                                     neg_ratio=board.args.neg_ratio, summarizer=summarizer)

            board.cprint(f'[testing at epoch-{epoch}]')
            results = evals.Inference(dataset=dataset, model=model, epoch=epoch, summarizer=summarizer)

            logging.info(f'[testing at epoch-{epoch}]')
            logging.info(results)

            max_recall20.append(results['recall'][0])
            logging.info(f'EPOCH[{epoch + 1}/{board.args.epoch}] {info} ')

        logging.info(nowtime.strftime("%m-%d-%Hh%Mm%Ss:  ") + str(max(max_recall20)) + "\n")

    except:
        raise NotImplementedError('Error in running main file')

    finally:
        if board.args.tensorboard:
            summarizer.close()


if __name__ == '__main__':
    quant()