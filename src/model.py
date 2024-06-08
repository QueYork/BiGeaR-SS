"""
@author:chenyankai
@file:model.py
@time:2021/11/11
"""
import torch
import torch.nn as nn
import src.powerboard as board
import src.data_loader as data_loader
import numpy as np
import math


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def get_scores(self, user_index):
        raise NotImplementedError


class BiGeaR_tch(BasicModel):
    def __init__(self, dataset):
        super(BiGeaR_tch, self).__init__()
        self.dataset: data_loader.LoadData = dataset

        self.__init_model()

    def __init_model(self):
        # 用户数
        self.num_users = self.dataset.get_num_users()
        # 物品数
        self.num_items = self.dataset.get_num_items()
        # 维数
        self.dim = board.args.dim
        # 跳数
        self.num_layers = board.args.layers
        # 用户和物品的原始 embedding
        self.user_embed = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.dim)
        self.item_embed = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.dim)
        # BiGeaR 论文中的 w_l，第 l 层 embedding 的权重
        self.lambdas = self.compute_concat_scaler()
        # K-pair BPR
        self.K = board.args.bpr_neg_num
        # Negative sample number
        self.n_negs = board.args.neg_ratio
        # DINS 论文中的 β
        self.alpha = board.args.alpha
        
        self.pos_rank = None
        self.neg_rank = None

        # 将原始 embedding 以正态分布初始化
        nn.init.normal_(self.user_embed.weight, std=0.1)
        nn.init.normal_(self.item_embed.weight, std=0.1)
        board.cprint('initializing with NORMAL distribution.')

        self.f = nn.Sigmoid()
        # 稀疏交互矩阵，注意这不是传统 u-i 矩阵，而是 [#users + #items, #users + #items] 的邻接矩阵
        self.Graph = self.dataset.load_sparse_graph()

    # 计算 w_l, 随层数增高而线性增长
    def compute_concat_scaler(self):
        lambdas = [torch.tensor((float)(x + 1) / (self.num_layers + 1)) for x in range(self.num_layers + 1)]
        return lambdas

    # Embedding aggregation
    def aggregate_embed(self): 
        user_embed = self.user_embed.weight # [#users, dim]
        item_embed = self.item_embed.weight # [#items, dim]

        # 连接用户和物品的原始 embedding
        con_original_embed = torch.cat([user_embed, item_embed], dim=0) # [#users + #items, dim]
        # 将乘以第一个缩放因子 self.lambdas[0] 后的原始 embedding 添加到列表中
        con_embed_list = [con_original_embed * self.lambdas[0]] # [1, #users + #items, dim]

        for layer in range(self.num_layers):
            # 利用稀疏矩阵，下一层 embedding 结果 = 交互矩阵 * 当前 embedding，意义是邻居 embedding 的加和
            con_original_embed = torch.sparse.mm(self.Graph, con_original_embed) # [#users + #items, dim]
            # 将下一层 * 缩放因子后加入列表
            con_embed_list.append(con_original_embed * self.lambdas[layer + 1]) # [1 + layer, #users + #items, dim]
        
        con_embed_list = torch.stack(con_embed_list, dim=1)
        return con_embed_list[:self.num_users, :], con_embed_list[self.num_users:, :] # [n_entity, num_layer + 1, dim]
    
    def pooling(self, embed: torch.Tensor):
        return embed.view(embed.shape[:-2] + (-1,)) 
    
    def _BPR_loss(self, user_embed, pos_embed, neg_embed):
        pos_scores = torch.mul(user_embed, pos_embed)
        pos_scores = torch.sum(pos_scores, dim=1)
        # neg_scores = torch.mul(user_embed, neg_embed)
        # neg_scores = torch.sum(neg_scores, dim=1)
        # loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        neg_scores = torch.sum(torch.mul(user_embed.unsqueeze(dim=1), neg_embed), axis=-1)  # [batch_size, K]
        loss = torch.mean(torch.log(1+torch.exp(neg_scores - pos_scores.unsqueeze(dim=1)).sum(dim=1)))
        return loss

    # MixGCF 负样本合成

    # def negative_synthesizing(self, con_origin_user_embed, con_origin_item_embed, user_index, neg_candidates, pos_item):
    #     batch_size = user_index.shape[0]
    #     u_e, p_e = con_origin_user_embed[user_index], con_origin_item_embed[pos_item]
        
    #     """positive mixing"""
    #     seed = torch.rand(batch_size, 1, p_e.shape[1], 1).to(p_e.device)
        
    #     # 负样本池
    #     n_e = con_origin_item_embed[neg_candidates]  # [batch_size, n_negs, (1 + num_layer) * dim]
    
    #     # 对负样本池的所有样本执行正混合
    #     n_e_ = seed * p_e.unsqueeze(dim=1) + (1 - seed) * n_e  # mixing
        
        
    #     """hop mixing""" # 从负样本池中筛选出每一层的embedding, 合成一个负样本输出
    #     # 筛选策略：对每一层，找到使得 当层用户embedding * 当层负样本 embedding 最大的那个 embedding
        
    #     scores = (u_e.unsqueeze(dim=1) * n_e_).sum(dim=-1)  # [batch_size, n_negs, num_layer+1]
    #     indices = torch.max(scores, dim=1)[1].detach()
    #     neg_items_emb_ = n_e_.permute([0, 2, 1, 3])  # [batch_size, num_layer+1, n_negs, dim]
        
    #     # [batch_size, (1 + num_layer), dim]
    #     return neg_items_emb_[[[i] for i in range(batch_size)], range(neg_items_emb_.shape[1]), indices, :]
    
    
    # DINS 负样本合成
    
    def negative_synthesizing(self, con_origin_user_embed, con_origin_item_embed, user_index, neg_candidates, pos_item):
        batch_size = user_index.shape[0]
        u_e, p_e = con_origin_user_embed[user_index], con_origin_item_embed[pos_item]
        
        """Hard Boundary Definition"""
        n_e = con_origin_item_embed[neg_candidates] # [batch_size, n_negs, (1 + num_layer), dim]
        scores = (u_e.unsqueeze(dim=1) * n_e).sum(dim=-1)  # [batch_size, n_negs, (1 + num_layer)]
        indices = torch.max(scores, dim=1)[1].detach()  
        neg_items_emb_ = n_e.permute([0, 2, 1, 3])  # [batch_size, (1 + num_layer), n_negs, dim]
        neg_items_embedding_hardest = neg_items_emb_[[[i] for i in range(batch_size)], 
                                                     range(neg_items_emb_.shape[1]), 
                                                     indices, :]   # [batch_size, num_layer+1, dim]
        
        """Dimension Independent Mixup"""
        neg_scores = torch.exp(u_e * neg_items_embedding_hardest) 
        total_sum = self.alpha * torch.exp ((u_e * p_e)) + neg_scores   
        neg_weight = neg_scores / total_sum  
        pos_weight = 1 - neg_weight   

        return pos_weight * p_e + neg_weight * neg_items_embedding_hardest  # mixing
    
    
    def loss(self, user_index, pos_index, neg_index):
        user_emb, item_emb = self.aggregate_embed()
        
        # 池化前的 embedding
        # u_e: [batch_size, (1 + num_layer), dim]
        # pos_e: [batch_size, (1 + num_layer), dim]
        # neg_e: [batch_size, K, (1 + num_layer), dim]
        nocon_u_e = user_emb[user_index]
        nocon_pos_e = item_emb[pos_index]
        nocon_neg_e = []
        for k in range(self.K):
            nocon_neg_e.append(self.negative_synthesizing(user_emb, 
                                                    item_emb, 
                                                    user_index, 
                                                    neg_index[:, k*self.n_negs: (k+1)*self.n_negs], 
                                                    pos_index))
        nocon_neg_e = torch.stack(nocon_neg_e, dim=1)
        
        # 池化后的 embedding
        # u_e: [batch_size, (1 + num_layer) * dim]
        # pos_e: [batch_size, (1 + num_layer) * dim]
        # neg_e: [batch_size, K, (1 + num_layer) * dim]
        user_emb = self.pooling(user_emb)
        item_emb = self.pooling(item_emb)
        
        u_e = user_emb[user_index]
        pos_e = item_emb[pos_index]
        neg_e = self.pooling(nocon_neg_e)
        
        # 正则化使用 layer_idx = 0 计算
        reg_loss = 0.5 * (torch.norm(nocon_u_e[:, 0, :]) ** 2
                       + torch.norm(nocon_pos_e[:, 0, :]) ** 2
                       + torch.norm(nocon_neg_e[:, :, 0, :]) ** 2) / float(len(user_index))
        
        loss1 = self._BPR_loss(u_e, pos_e, neg_e)
        return loss1, reg_loss

    def get_scores(self, user_index):
        all_user_embed, all_item_embed = self.aggregate_embed()
        
        all_user_embed = self.pooling(all_user_embed)
        all_item_embed = self.pooling(all_item_embed)
        
        user_embed = all_user_embed[user_index.long()]
        scores = self.f(torch.matmul(user_embed, all_item_embed.t()))
        return scores

    def summary(self):
        user_embed = self.user_embed.weight
        item_embed = self.item_embed.weight
        self.RP_embed = [user_embed, item_embed]

        con_original_embed = torch.cat([user_embed, item_embed])
        pos_rank_list_LW = []
        neg_rank_list_LW = []
        con_embed_list = []
        g = self.Graph
        for layer in range(self.num_layers + 1):
            if layer == 0:
                con_embed_i = con_original_embed * self.lambdas[0]
                con_embed_list = [con_embed_i]

            else:
                con_original_embed = torch.sparse.mm(g, con_original_embed)
                con_embed_i = con_original_embed * self.lambdas[layer]
                con_embed_list.append(con_embed_i)

            con_origin_users_embed, con_origin_items_embed = torch.split(con_embed_i, [self.num_users, self.num_items])
            score_i = torch.matmul(con_origin_users_embed, con_origin_items_embed.t())
            _, pos_rank_i = torch.topk(score_i, largest=True, k=board.args.R)
            _, neg_rank_i = torch.topk(score_i, largest=False, k=board.args.R)
            pos_rank_list_LW.append(pos_rank_i.int())
            neg_rank_list_LW.append(neg_rank_i.int())

        con_output = torch.cat(con_embed_list, dim=1)
        agg_user_embed, agg_item_embed = torch.split(con_output, [self.num_users, self.num_items])
        all_score = torch.matmul(agg_user_embed, agg_item_embed.t())
        _, pos_rank_end = torch.topk(all_score, largest=True, k=board.args.R)
        _, neg_rank_end = torch.topk(all_score, largest=False, k=board.args.R)

        self.pos_rank = torch.stack(pos_rank_list_LW, dim=2)
        self.neg_rank = torch.stack(neg_rank_list_LW, dim=2)

    def batch_rank(self, user_tensor, item_tensor):
        embed_list = [user_tensor, item_tensor]
        pos_rank_list = []
        neg_rank_list = []

        import src.utils as utils
        for b_user_tensor, b_item_tensor in utils.minibatch(embed_list):
            score_i = torch.matmul(b_user_tensor, b_item_tensor.t())
            _, pos_rank_i = torch.topk(score_i, largest=True, k=board.args.R)
            _, neg_rank_i = torch.topk(score_i, largest=False, k=board.args.R)
            pos_rank_list.append(pos_rank_i.int())
            neg_rank_list.append(neg_rank_i.int())

        pos_rank_list = torch.cat(pos_rank_list, dim=0)
        neg_rank_list = torch.cat(neg_rank_list, dim=0)
        return pos_rank_list, neg_rank_list

    def summary_forOOM(self):
        user_embed = self.user_embed.weight
        item_embed = self.item_embed.weight
        self.RP_embed = [user_embed, item_embed]

        con_original_embed = torch.cat([user_embed, item_embed])
        pos_rank_list_LW = []
        neg_rank_list_LW = []
        con_embed_list = []
        g = self.Graph
        for layer in range(self.num_layers + 1):
            if layer == 0:
                con_embed_i = con_original_embed * self.lambdas[0]
                con_embed_list = [con_embed_i]

            else:
                con_original_embed = torch.sparse.mm(g, con_original_embed)
                con_embed_i = con_original_embed * self.lambdas[layer]
                con_embed_list.append(con_embed_i)

            con_origin_users_embed, con_origin_items_embed = torch.split(con_embed_i, [self.num_users, self.num_items])
            pos_rank_i, neg_rank_i = self.batch_rank(con_origin_users_embed, con_origin_items_embed)
            pos_rank_list_LW.append(pos_rank_i)
            neg_rank_list_LW.append(neg_rank_i)

        self.pos_rank = torch.stack(pos_rank_list_LW, dim=2)
        self.neg_rank = torch.stack(neg_rank_list_LW, dim=2)


# approximate the Dirac-delta function by using the normal distribution
# https://en.wikipedia.org/wiki/Dirac_delta_function
class Normal_Ddelta(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        binary_encoding = torch.sign(x)
        n = x[0].nelement()
        m = x.norm(1, 1, keepdim=True).div(n)
        scaler = m.expand(x.size())
        encodes = binary_encoding.mul(scaler)
        return encodes

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        output = board.norm_a / (math.pi ** 0.5) * torch.exp(-(board.norm_a * input) ** 2)
        return torch.clamp(output, min=1e-8, max=1.) * grad_output


class Quant_layer(nn.Module):
    def __init__(self):
        super(Quant_layer, self).__init__()
        self.dim = board.args.dim

    def binarize(self, X):
        return Normal_Ddelta.apply(X)

    def forward(self, X):
        bin_output = self.binarize(X)
        return bin_output


class BiGeaR(BasicModel):
    def __init__(self, dataset):
        super(BiGeaR, self).__init__()
        self.dataset: data_loader.LoadData = dataset

        self.__init_model()

    def __init_model(self):
        self.num_users = self.dataset.get_num_users()
        self.num_items = self.dataset.get_num_items()
        self.dim = board.args.dim
        self.num_layers = board.args.layers
        self.user_embed_std = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.dim)
        self.item_embed_std = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.dim)

        self.lambdas = self.compute_concat_scaler()
        self.id_weight = self.compute_ID_weight()
        self.quant_layer = Quant_layer()
        self.f = nn.Sigmoid()

        self.Graph = self.dataset.load_sparse_graph()

        self.alpha = board.args.alpha

        self.compute_rank = board.args.compute_rank
        if self.compute_rank == 1:
            self.pos_rank_tch, self.neg_rank_tch = self.get_tch_rank()
        else:
            self.score_tch = self.get_tch_score()

    def compute_concat_scaler(self):
        lambdas = [torch.tensor((float)(x + 1) / (self.num_layers + 1)) for x in range(self.num_layers + 1)]
        return lambdas

    def compute_ID_weight(self):
        topR = board.args.R
        weight = [np.exp(-(r + 1) * board.args.lmd2) for r in range(topR)]
        return torch.FloatTensor(weight).to(board.DEVICE)

    def get_tch_score(self):
        # step 1: compute the aggregated embeddings from the teacher model
        print('use pretarined teacher embedding')
        file = f"bgr-{board.args.dataset}-{board.args.dim}.pth.tar"
        import os
        pretrain_file = os.path.join(board.FILE_PATH, file)
        state_dict = torch.load(pretrain_file, map_location=torch.device('cpu'))
        board.cprint(f"loaded model weights from {pretrain_file}")
        self.user_embed_std.weight.data.copy_(state_dict[0])
        self.item_embed_std.weight.data.copy_(state_dict[1])

        user_embed_tch = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.dim).to(board.DEVICE)
        item_embed_tch = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.dim).to(board.DEVICE)
        user_embed_tch.weight.data.copy_(state_dict[0])
        item_embed_tch.weight.data.copy_(state_dict[1])
        user_embed_tch.weight.requires_grad = False
        item_embed_tch.weight.requires_grad = False

        user_embed = user_embed_tch.weight
        item_embed = item_embed_tch.weight
        con_embed = torch.cat([user_embed, item_embed])
        g = self.Graph

        con_output = [con_embed * self.lambdas[0]]
        for layer in range(self.num_layers):
            con_embed = torch.sparse.mm(g, con_embed)
            con_output.append(con_embed * self.lambdas[layer + 1])

        con_output = torch.cat(con_output, dim=1)

        agg_user_embed_tch, agg_item_embed_tch = torch.split(con_output, [self.num_users, self.num_items])

        # step 2: compute the scores and rankings
        tch_score = torch.matmul(agg_user_embed_tch, agg_item_embed_tch.t())
        return tch_score

    def get_tch_rank(self):
        print('use pretarined teacher embedding')
        file = f"bgr-{board.args.dataset}-{board.args.dim}-rank.pth.tar"
        import os
        pretrain_file = os.path.join(board.FILE_PATH, file)
        state_dict = torch.load(pretrain_file, map_location=torch.device('cpu'))
        board.cprint(f"loaded model weights from {pretrain_file}")
        self.user_embed_std.weight.data.copy_(state_dict['RP_embed'][0])
        self.item_embed_std.weight.data.copy_(state_dict['RP_embed'][1])

        return state_dict['pos_rank_LW'].to(board.DEVICE), state_dict['neg_rank_LW'].to(board.DEVICE)

    def aggregate_embed_std(self):
        user_embed = self.user_embed_std.weight
        item_embed = self.item_embed_std.weight

        con_original_embed = torch.cat([user_embed, item_embed])
        con_embed_list = [con_original_embed]
        bin_quant_embed = self.quant_layer(con_original_embed)
        bin_quant_list = [bin_quant_embed * self.lambdas[0]]

        for layer in range(self.num_layers):
            con_original_embed = torch.sparse.mm(self.Graph, con_original_embed)
            con_embed_list.append(con_original_embed)
            bin_quant_embed = self.quant_layer(con_original_embed)
            bin_quant_list.append(bin_quant_embed * self.lambdas[layer + 1])

        con_embed_list = torch.stack(con_embed_list, dim=1)
        bin_quant_list = torch.stack(bin_quant_list, dim=1)
        return con_embed_list[:self.num_users, :], con_embed_list[self.num_users:, :], bin_quant_list[:self.num_users, :], bin_quant_list[self.num_users:, :]

    def pooling(self, embed: torch.Tensor):
        return embed.view(embed.shape[:-2] + (-1,)) 
    
    def _BPR_loss(self, user_embed, pos_embed, neg_embed):
        pos_scores = torch.mul(user_embed, pos_embed)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(user_embed, neg_embed)
        neg_scores = torch.sum(neg_scores, dim=1)
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        return loss

    def _ID_loss(self, user_index, user_embed, item_embed):
        num_user = user_index.shape[0]

        user_emb_lw = torch.split(user_embed[user_index], self.dim, dim=1)
        item_emb_lw = torch.split(item_embed, self.dim, dim=1)
        index = torch.arange(0, num_user).view(num_user, 1).to(board.DEVICE)

        rd_loss = []
        for i in range(self.num_layers + 1):
            if self.compute_rank == 1:
                tch_topr_index = self.pos_rank_tch[user_index, :, i].long()
            else:
                raise NotImplementedError

            std_scores_i = torch.matmul(user_emb_lw[i], item_emb_lw[i].t())
            std_scores_topR_i = std_scores_i[index, tch_topr_index]
            _, std_topr_index_i = torch.topk(std_scores_i, largest=True, k=board.args.R)

            loss = torch.mean(
                torch.mean(torch.mul(torch.nn.functional.softplus(-std_scores_topR_i), self.id_weight), dim=-1))
            rd_loss.append(loss)

        rd_loss = torch.stack(rd_loss, dim=0)
        return rd_loss

    # MixGCF 负样本合成

    # def negative_synthesizing(self, con_origin_user_embed, con_origin_item_embed, user_index, neg_candidates, pos_item):
    #     batch_size = user_index.shape[0]
    #     u_e, p_e = con_origin_user_embed[user_index], con_origin_item_embed[pos_item]
        
    #     """positive mixing"""
    #     seed = torch.rand(batch_size, 1, p_e.shape[1], 1).to(p_e.device)
        
    #     # 负样本池
    #     n_e = con_origin_item_embed[neg_candidates]  # [batch_size, n_negs, (1 + num_layer) * dim]
    
    #     # 对负样本池的所有样本执行正混合
    #     n_e_ = seed * p_e.unsqueeze(dim=1) + (1 - seed) * n_e  # mixing
        
        
    #     """hop mixing""" # 从负样本池中筛选出每一层的embedding, 合成一个负样本输出
    #     # 筛选策略：对每一层，找到使得 当层用户embedding * 当层负样本 embedding 最大的那个 embedding
        
    #     scores = (u_e.unsqueeze(dim=1) * n_e_).sum(dim=-1)  # [batch_size, n_negs, num_layer+1]
    #     indices = torch.max(scores, dim=1)[1].detach()
    #     neg_items_emb_ = n_e_.permute([0, 2, 1, 3])  # [batch_size, num_layer+1, n_negs, dim]
        
    #     # [batch_size, (1 + num_layer), dim]
    #     return neg_items_emb_[[[i] for i in range(batch_size)], range(neg_items_emb_.shape[1]), indices, :]
    

    # DINS 负样本合成
    
    def negative_synthesizing(self, con_origin_user_embed, con_origin_item_embed, user_index, neg_candidates, pos_item):
        batch_size = user_index.shape[0]
        u_e, p_e = con_origin_user_embed[user_index], con_origin_item_embed[pos_item]
        
        """Hard Boundary Definition"""
        n_e = con_origin_item_embed[neg_candidates] # [batch_size, n_negs, (1 + num_layer), dim]
        scores = (u_e.unsqueeze(dim=1) * n_e).sum(dim=-1)  # [batch_size, n_negs, (1 + num_layer)]
        indices = torch.max(scores, dim=1)[1].detach()  
        neg_items_emb_ = n_e.permute([0, 2, 1, 3])  # [batch_size, (1 + num_layer), n_negs, dim]
        neg_items_embedding_hardest = neg_items_emb_[[[i] for i in range(batch_size)], 
                                                     range(neg_items_emb_.shape[1]), 
                                                     indices, :]   # [batch_size, num_layer+1, dim]
        
        """Dimension Independent Mixup"""
        neg_scores = torch.exp(u_e * neg_items_embedding_hardest) 
        total_sum = self.alpha * torch.exp ((u_e * p_e)) + neg_scores   
        neg_weight = neg_scores / total_sum  
        pos_weight = 1 - neg_weight   

        return pos_weight * p_e + neg_weight * neg_items_embedding_hardest  # mixing

    def loss(self, user_index, pos_index, neg_index):
        full_user_emb, full_item_emb, \
        bin_user_emb, bin_item_emb = self.aggregate_embed_std()

        bin_neg_e = self.negative_synthesizing(bin_user_emb, bin_item_emb, user_index, neg_index, pos_index)
        
        bin_user_emb = self.pooling(bin_user_emb)
        bin_item_emb = self.pooling(bin_item_emb)
        
        u_e = bin_user_emb[user_index]
        pos_e = bin_item_emb[pos_index]
        neg_e = self.pooling(bin_neg_e)
        
        loss1 = self._BPR_loss(u_e, pos_e, neg_e)
        loss2 = self._ID_loss(user_index, bin_user_emb, bin_item_emb)
    
        reg_loss = 0.5 * (self.user_embed_std(user_index).norm(2).pow(2) +
                          self.item_embed_std(pos_index).norm(2).pow(2) +
                          self.item_embed_std(neg_index).norm(2).pow(2)) / float(len(user_index))
        
        return loss1, loss2, reg_loss



    # def loss(self, user_index, pos_index, neg_index):
    #     user_con_embed = self.user_embed_std(user_index)
    #     pos_con_embed = self.item_embed_std(pos_index)
    #     neg_con_embed = self.item_embed_std(neg_index)

    #     con_origin_user_embed, con_origin_item_embed, \
    #     bin_quant_user_embed, bin_quant_item_embed = self.aggregate_embed_std()
        
    #     bin_quant_user_embed = self.pooling(bin_quant_user_embed)
    #     bin_quant_item_embed = self.pooling(bin_quant_item_embed)

    #     bin_quant_user_agg_embed = bin_quant_user_embed[user_index]
    #     bin_quant_pos_agg_embed = bin_quant_item_embed[pos_index]
    #     bin_quant_neg_agg_embed = bin_quant_item_embed[neg_index]

    #     reg_loss = 0.5 * (user_con_embed.norm(2).pow(2) +
    #                       pos_con_embed.norm(2).pow(2) +
    #                       neg_con_embed.norm(2).pow(2)) / float(len(user_index))

    #     loss1 = self._BPR_loss(bin_quant_user_agg_embed, bin_quant_pos_agg_embed, bin_quant_neg_agg_embed)
    #     loss2 = self._ID_loss(user_index, bin_quant_user_embed, bin_quant_item_embed)
    #     return loss1, loss2, reg_loss

    def get_scores(self, user_index):
        _, _, bin_all_user_embed, bin_all_item_embed = self.aggregate_embed_std()
        
        bin_all_user_embed = self.pooling(bin_all_user_embed)
        bin_all_item_embed = self.pooling(bin_all_item_embed)
        
        user_embed = bin_all_user_embed[user_index.long()]
        scores = torch.matmul(user_embed, bin_all_item_embed.t())
        return scores
