"""
@author:chenyankai
@file:utils.py
@time:2021/11/11
"""

import src.powerboard as board
import torch
import numpy as np
from src.data_loader import LoadData
from sklearn.metrics import roc_auc_score
import os
import logging

try:
    from cppimport import imp_from_filepath
    from os.path import join, dirname
    import sys

    path = join(dirname(__file__), "sources/sampling.cpp")
    sampling = imp_from_filepath(path)
    sampling.seed(board.SEED)
    board.cprint('sampling.cpp file successfully loaded')
except:
    board.cprint('cpp file not loaded')


# **************************utils**************************
def update_temp(temp, epoch, min_temp):
    if epoch >= board.args.temp_spot:
        return max(board.args.temp_decay * temp, min_temp)
    else:
        return temp


def uniform_sampler(dataset, neg_ratio=1):
    dataset: LoadData
    num_user = dataset.get_num_users()
    num_item = dataset.get_num_items()
    num_train_instance = dataset.get_num_train_instance()
    all_pos = dataset.get_all_pos()
    samples = sampling.sample_negative(num_user, num_item, num_train_instance, neg_ratio, all_pos)
    return samples


def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def get_weight_file_name():
    pass


def minibatch(*tensors, **kwargs):
    batch_size = kwargs.get('batch_size', board.args.train_batch)

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays):
    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('inputs to shuffle should be equal length')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)
    if len(arrays) == 1:
        return arrays[0][shuffle_indices]
    else:
        return tuple(x[shuffle_indices] for x in arrays)


def create_log_name(dir_path):
    log_count = 0
    file_path = os.path.join(dir_path, 'log{:d}.log'.format(log_count))
    while os.path.exists(file_path):
        log_count += 1
        file_path = os.path.join(dir_path, 'log{:d}.log'.format(log_count))
    return 'log{:d}.log'.format(log_count)


def log_config(path=None, name=None, level=logging.DEBUG, console_level=logging.DEBUG, console=True):
    if not os.path.exists(path):
        os.makedirs(path)
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)
    logging.root.handlers = []
    logpath = os.path.join(path, name)
    print("All logs will be saved to %s" % logpath)

    logging.root.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logfile = logging.FileHandler(logpath)
    logfile.setLevel(level)
    logfile.setFormatter(formatter)
    logging.root.addHandler(logfile)

    if console:
        logconsole = logging.StreamHandler()
        logconsole.setLevel(console_level)
        logconsole.setFormatter(formatter)
        logging.root.addHandler(logconsole)
    return path


class timer:
    from time import time
    Time_point = [-1]  # global time record
    Named_timeP = {}

    def __init__(self, time_point=None, **kwargs):
        if kwargs.get('name'):
            self.name = kwargs['name']
            if self.name not in timer.Named_timeP:
                timer.Named_timeP[self.name] = 0.
        else:
            self.name = False
            self.tp = time_point or timer.Time_point

    def __enter__(self):
        self.start = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.name:
            timer.Named_timeP[self.name] += timer.time() - self.start
        else:
            self.tp.append(timer.time() - self.start)

    @staticmethod
    def get():
        if len(timer.Time_point) > 1:
            return timer.Time_point.pop()
        else:
            return -1

    @staticmethod
    def dict(keys=None):
        phase = '--'
        if keys is None:
            for key, value in timer.Named_timeP.items():
                phase += f'{key}: {value:.2f}s--'
        else:
            for key in keys:
                value = timer.Named_timeP[key]
                phase += f'{key}:{value:.2f}s--'
        return phase

    @staticmethod
    def zero(keys=None):
        if keys is None:
            for key, value in timer.Named_timeP.items():
                timer.Named_timeP[key] = 0
        else:
            for key in keys:
                timer.Named_timeP[key] = 0


# **************************End of utils**************************


# **************************Metrics**************************
def Recall_Precision_K(true_data, hit_data, k):
    """
    :param true_data: positive items of each user
    :param hit_data: hit items of each user
    :param k: topk
    :return:
    """
    assert len(true_data) == len(hit_data)
    hit_sum = np.sum(hit_data[:, :k], axis=1)
    true_n = np.array([len(x) for x in true_data])
    recall = hit_sum / true_n
    precision = hit_sum / k
    return np.sum(recall), np.sum(precision)


def NDCG_K(true_data, hit_data, k):
    """
    :param true_data: positive items of each users
    :param hit_data: hit items of each user
    :param k:
    :return:
    """
    assert len(true_data) == len(hit_data)
    idea_data = np.array(hit_data)
    idea_data.sort(axis=1)
    idea_data = np.flip(idea_data, axis=1)
    idea_data = idea_data[:, :k]
    # rel_i = 1 or 0, 2^{rel_i} - 1 = 1 or 0
    idcg = np.sum(idea_data * 1. / np.log2(np.arange(2, k + 2)), axis=1)

    hit_data = hit_data[:, :k]
    dcg = hit_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


def AUC(true_data, score_data):
    """
    :return: auc values
    """
    auc = []
    for i, data in enumerate(true_data):
        score_data_i = score_data[i]
        r_all = np.zeros(len(score_data_i))
        data = np.array(data, dtype=int)
        r_all[data] = 1
        true_label = r_all[score_data_i >= 0]
        if np.all(true_label == 0):
            auc.append(0.)
        else:
            pred_label = score_data_i[score_data_i >= 0]
            auc.append(roc_auc_score(true_label, pred_label))
    return np.sum(auc)


def get_hit_data(true_data, pred_data):
    """
    :return: return True or False label for pred_data
    """
    hit = []
    for i in range(len(true_data)):
        ground_truth = true_data[i]
        predict = pred_data[i]
        pred = list(map(lambda x: x in ground_truth, predict))
        pred = np.array(pred).astype('float')
        hit.append(pred)
    return np.array(hit)
# **************************End of Metrics**************************
