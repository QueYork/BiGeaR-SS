"""
@author:chenyankai
@file:data_loader.py
@time:2021/11/11
"""
import os
from os.path import join
import torch
import numpy as np
from torch.utils.data import Dataset
import scipy.sparse as sp
import src.powerboard as board
from time import time


class LoadData(Dataset):
    def __init__(self, data_name='music'):
        path = join(board.DATA_PATH, data_name)
        print(f'dataset loading [{path}]')
        self.mode_dict = {'train': 0, 'test': 1}
        self.mode = self.mode_dict['train']
        self.path = path
        train_file = join(self.path, board.args.train_file)
        print(train_file)
        test_file = join(self.path, board.args.test_file)
        self.n_users = 0
        self.m_items = 0
        self.data_name = data_name

        train_unique_users, train_users, train_items = [], [], []
        test_unique_users, test_users, test_items = [], [], []
        self.train_instance_size = 0
        self.test_instance_size = 0

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip()
                    l = l.strip('\n').split(' ')

                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    train_unique_users.append(uid)
                    train_users.extend([uid] * len(items))
                    train_items.extend(items)
                    self.n_users = max(self.n_users, uid)
                    self.m_items = max(self.m_items, max(items))
                    self.train_instance_size += len(items)
        self.train_unique_users = np.array(train_unique_users)
        self.train_users = np.array(train_users)
        self.train_items = np.array(train_items)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    if len(l) > 1:
                        uid = int(l[0])
                        items = [int(i) for i in l[1:]]
                        test_unique_users.append(uid)
                        test_users.extend([uid] * len(items))
                        test_items.extend(items)
                        self.n_users = max(self.n_users, uid)
                        self.m_items = max(self.m_items, max(items))
                        self.test_instance_size += len(items)
                    else:
                        raise NotImplementedError(uid)
        self.test_unique_users = np.array(test_unique_users)
        self.test_users = np.array(test_users)
        self.test_items = np.array(test_items)
        self.m_items += 1
        self.n_users += 1

        self.Graph = None
        print(f'{self.train_instance_size} interactions for training')
        print(f'{self.test_instance_size} interactions for testing')
        print(f'{self.data_name} sparsity:  '
              f'{(self.train_instance_size + self.test_instance_size) / self.n_users / self.m_items}')

        # (users, items), bipartite graph
        self.user_item_net = sp.csr_matrix((np.ones(self.train_instance_size), (self.train_users, self.train_items)),
                                           shape=(self.n_users, self.m_items))

        self.users_D = np.array(self.user_item_net.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1.
        self.items_D = np.array(self.user_item_net.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.

        # pre-process
        self.all_pos = self._get_user_posItems(list(range(self.n_users)))
        # 找 all_neg
        self.all_neg = self._get_user_negItems(list(range(self.n_users)))
        
        self.test_dict = self._build_test()
        print(f"{self.data_name} is loaded")

    def get_num_users(self):
        return self.n_users

    def get_num_items(self):
        return self.m_items

    def get_num_train_instance(self):
        return self.train_instance_size

    def get_test_dict(self):
        return self.test_dict

    def get_all_pos(self):
        return self.all_pos
    
    def get_all_neg(self):
        return self.all_neg

    def _build_test(self):
        """
        :return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.test_items):
            user = self.test_users[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def _get_user_posItems(self, user_index):
        pos_items = []
        for user in user_index:
            pos_items.append(self.user_item_net[user].nonzero()[1])
        return pos_items

    def _get_user_negItems(self, user_index):
        neg_items = []
        for user in user_index:
            all_items = np.arange(self.m_items)
            neg_items.append(np.setdiff1d(all_items, self.all_pos[user], assume_unique=True))
        return neg_items

    def _convert_sp_mat_to_sp_tensor(self, matrix):
        """
        sparse matrix converting between scipy.sparse and torch.sparse
        :return: scipy.sparse_matrix of input matrix
        """
        coo = matrix.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def load_sparse_graph(self):
        print('loading adjacency matrix')
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(os.path.join(self.path, 's_pre_adj_mat.npz'))
                print('load graph successfully ...')
                norm_adj = pre_adj_mat
            except:
                print('generating graph')
                start = time()
                adj_mat = sp.lil_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                R = self.user_item_net.tolil()

                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f'construction complete with {end - start}s, saving norm_mat...')
                sp.save_npz(os.path.join(self.path, 's_pre_adj_mat.npz'), norm_adj)

            self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph = self.Graph.coalesce()
        return self.Graph.to(board.DEVICE)


    def load_graph_dropEdge(self):
        list = np.arange(self.train_instance_size)
        new_index = np.random.choice(list, size=int(self.train_instance_size * (1 - board.args.dropout_ratio)), replace=False)
        new_user = np.array(self.train_users)[new_index]
        new_item = np.array(self.train_items)[new_index]
        indicator = np.ones_like(new_user, dtype=np.float32)
        masked_graph = sp.csr_matrix((indicator, (new_user, new_item)), shape=(self.n_users, self.m_items))
        R = masked_graph.tolil()

        adj_mat = sp.lil_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
        adj_mat[:self.n_users,  self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)

        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()

        tmp_graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
        tmp_graph = tmp_graph.coalesce()

        return tmp_graph.to(board.DEVICE)



