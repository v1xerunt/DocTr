import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F

from recbole.utils import InputType
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss, EmbLoss


class Doctr(GeneralRecommender):

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(Doctr, self).__init__(config, dataset)

        # load dataset info
        self.n_users = dataset.user_num
        self.n_items = dataset.item_num
        self.use_pretrain = True
        
        pretrained_user_emb = dataset.user_feat['user_feat']
        pretrained_item_emb = dataset.item_feat['item_feat']
        pretrained_item_emb_norm = dataset.item_feat['item_feat_2']
        self.user_info = dataset.user_feat['user_info'] # Trial category code
        self.user_summary = dataset.user_feat['user_summary'] # Trial summary
        self.item_his = dataset.item_feat['item_his'] # Clinician history

        if self.use_pretrain:
            self.user_embedding = nn.Embedding.from_pretrained(pretrained_user_emb.to(torch.float32), freeze=True) # User embedding: Trial description
            self.item_embedding_individual = nn.Embedding.from_pretrained(pretrained_item_emb.to(torch.float32), freeze=True) # Item embedding: Clinician resource icd codes
            self.item_embedding_global = nn.Embedding.from_pretrained(pretrained_item_emb_norm.to(torch.float32), freeze=True) # Item embedding: Clinician resource icd codes
            self.user_info = nn.Embedding.from_pretrained(self.user_info.to(torch.float32), freeze=True) # User info: Trial ICD code
            self.user_summary = nn.Embedding.from_pretrained(self.user_summary.to(torch.float32), freeze=True) # User summary: Trial summary
            self.item_his = nn.Embedding.from_pretrained(self.item_his.to(torch.float32), freeze=True) # Item history: Clinician history enrolled trials - pool of user_embedding
        else:
            self.user_embedding = nn.Embedding(self.n_users, pretrained_user_emb.shape[1])
            self.item_embedding = nn.Embedding(self.n_items, pretrained_item_emb.shape[1])
            self.user_info = nn.Embedding(self.n_users, self.user_info.shape[1])
            self.user_summary = nn.Embedding(self.n_users, self.user_summary.shape[1])
            self.item_his = nn.Embedding(self.n_items, self.item_his.shape[1])
            xavier_normal_initialization(self.user_embedding.weight)
            xavier_normal_initialization(self.item_embedding.weight)
            xavier_normal_initialization(self.user_info.weight)
            xavier_normal_initialization(self.user_summary.weight)
            xavier_normal_initialization(self.item_his.weight)

        self.item_embedding = self.item_embedding_global
        trial_size = self.user_embedding.weight.shape[1]
        clinician_size = self.item_embedding.weight.shape[1]
        
        self.n_layers = 2
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)
        
        self.hidden_size = config['hidden_size']
        self.dropout = nn.Dropout(p=0.3)
        
        self.trial_info_encoder = nn.Sequential(
            nn.Linear(self.user_info.weight.shape[1], self.hidden_size),
            nn.LeakyReLU(),
        )
        self.trial_summary_encoder = nn.Sequential(
            nn.Linear(self.user_summary.weight.shape[1], self.hidden_size),
            nn.LeakyReLU(),
        )
        self.trial_embd_encoder = nn.Sequential(
            nn.Linear(trial_size, self.hidden_size),
            nn.LeakyReLU(),
        )
        
        self.c_embd_encoder = nn.Sequential(
            nn.Linear(clinician_size, self.hidden_size),
            nn.LeakyReLU(),
        )
        
        self.norm_embd_encoder = nn.Sequential(
            nn.Linear(clinician_size, self.hidden_size),
            nn.LeakyReLU(),
        )
        
        self.c_history_encoder = nn.Sequential(
            nn.Linear(self.item_his.weight.shape[1], self.hidden_size),
            nn.LeakyReLU(),
        )
        
        plus_dim = 0
        self.t_transform = nn.Sequential(
            nn.Linear(self.hidden_size*3 + plus_dim, self.hidden_size),
            nn.LeakyReLU(),
        )
        self.c_transform = nn.Sequential(
            nn.Linear(self.hidden_size*2 + plus_dim, self.hidden_size),
            nn.LeakyReLU(),
        )
        
        self.alpha = nn.Parameter(torch.FloatTensor(1, 1).fill_(0.5))
        self.loss = BPRLoss()
        self.reg_loss = EmbLoss()
        
        
    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.n_users, inter_M_t.col),
                    [1] * inter_M_t.nnz,
                )
            )
        )
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL
    
    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding_individual.weight
        user_embeddings = self.trial_embd_encoder(user_embeddings)
        item_embeddings = self.norm_embd_encoder(item_embeddings)
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings = self.dropout(all_embeddings)
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
            
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        #lightgcn_all_embeddings = lightgcn_all_embeddings[:, -1, :]
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items]
        )
        return user_all_embeddings, item_all_embeddings
    
    def get_t_embd(self, user):
        user_e = self.user_embedding(user)   
        user_info = self.user_info(user)
        user_summary = self.user_summary(user)
        
        #Encode 
        user_info = self.trial_info_encoder(user_info)
        user_summary = self.trial_summary_encoder(user_summary)
        user_e = self.trial_embd_encoder(user_e)
        enrich_t = torch.cat((user_info, user_summary, user_e), dim=1)
        return enrich_t
    
    def get_c_embd(self, item):
        item_e = self.item_embedding_individual(item)
        item_his = self.item_his(item)
        
        item_e = self.c_embd_encoder(item_e)
        item_his = self.c_history_encoder(item_his)
        enrich_c = torch.cat((item_e, item_his), dim=1)
        return enrich_c

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        
        enrich_t = self.get_t_embd(user)
        enrich_cpos = self.get_c_embd(pos_item)
        enrich_cneg = self.get_c_embd(neg_item)
                
        user_all_embeddings, item_all_embeddings = self.forward()
        structure_user = user_all_embeddings[user]
        structure_pos_item = item_all_embeddings[pos_item]
        structure_neg_item = item_all_embeddings[neg_item]
        
        enrich_t = self.t_transform(enrich_t)
        enrich_cpos = self.c_transform(enrich_cpos)
        enrich_cneg = self.c_transform(enrich_cneg)
        
        spatial_pos_score = torch.mul(structure_user, structure_pos_item).sum(dim=1)
        spatial_neg_score = torch.mul(structure_user, structure_neg_item).sum(dim=1)
        
        out_pos = torch.mul(enrich_t, enrich_cpos).sum(dim=1)
        out_neg = torch.mul(enrich_t, enrich_cneg).sum(dim=1)
        
        out_pos = self.alpha*out_pos + spatial_pos_score
        out_neg = self.alpha*out_neg + spatial_neg_score
        
        loss = self.loss(out_pos, out_neg)

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        enrich_t = self.get_t_embd(user)
        enrich_c = self.get_c_embd(item)
        
        user_all_embeddings, item_all_embeddings = self.forward()
        structure_user = user_all_embeddings[user]
        structure_item = item_all_embeddings[item]
        
        enrich_t = self.t_transform(enrich_t)
        enrich_c = self.c_transform(enrich_c)
        
        spatial_score = torch.mul(structure_user, structure_item).sum(dim=1)
        out = torch.mul(enrich_t, enrich_c).sum(dim=1)
        scores = self.alpha*out + spatial_score

        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]

        enrich_t = self.get_t_embd(user)
        enrich_c = self.get_c_embd(torch.arange(self.n_items).to(enrich_t.device))
        
        user_all_embeddings, item_all_embeddings = self.forward()
        structure_user = user_all_embeddings[user]
        
        enrich_t = self.t_transform(enrich_t)
        enrich_c = self.c_transform(enrich_c)
        
        spatial_score = torch.matmul(structure_user, item_all_embeddings.transpose(0, 1))
        out = torch.matmul(enrich_t, enrich_c.transpose(0, 1))
        scores = self.alpha*out + spatial_score
        

        return scores