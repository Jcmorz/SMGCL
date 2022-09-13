import torch
import csv
from torch import nn, optim

from torch.nn import functional as F
from torchvision.ops import focal_loss
import pytorch_lightning as pl
from sklearn import metrics

from .bgnn import BGNNA, BGCNA
from .model_help import BaseModel
from .dataset import PairGraphData
from . import MODEL_REGISTRY


def constructNet(drug_dis_matrix):
    # 构造两个分别为(269, 269)和(598, 598)的全0矩阵
    drug_matrix = torch.zeros((drug_dis_matrix.shape[0], drug_dis_matrix.shape[0]), dtype=torch.int32)
    dis_matrix = torch.zeros((drug_dis_matrix.shape[1], drug_dis_matrix.shape[1]), dtype=torch.int32)

    mat1 = torch.cat((drug_dis_matrix, drug_matrix), 1)
    mat2 = torch.cat((dis_matrix, drug_dis_matrix.T), 1)
    adj = torch.cat((mat1, mat2), 0)
    return adj


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, num_embeddings, out_channels, dropout, negative_slope, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.input_dim = num_embeddings
        self.output_dim = out_channels
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(num_embeddings, out_channels)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_channels, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.LeakyReLU = nn.LeakyReLU(self.negative_slope)

    def forward(self, x, edge, embedding):         # embedding -- h  ; adg -- edge
        if not hasattr(self, "edge_index"):
            edge_index = torch.sparse_coo_tensor(*edge)     # 将用元组存储的矩阵转换成稀疏tensor
            self.register_buffer("edge_index", edge_index)
        edge_index = self.edge_index
        edge_index = edge_index.to_dense()      # 将稀疏tensor转换成稠密矩阵

        Wh = torch.mm(embedding, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)

        # e = self._prepare_attentional_mechanism_input(Wh)               # 普通注意力聚合
        e = self._prepare_query_aware_mechanism_input(Wh, edge_index)   # 查询感知注意力聚合

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(edge_index > 0, e, zero_vec)   # torch.where(condition, x, y) -- When True (nonzero), yield x, otherwise yield y
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            embedding = F.elu(h_prime)
        else:
            embedding = h_prime

        x = F.embedding(x, embedding)
        x = F.normalize(x)
        return x

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.output_dim, :])
        Wh2 = torch.matmul(Wh, self.a[self.output_dim:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.LeakyReLU(e)

    def _prepare_query_aware_mechanism_input(self, Wh, edge_index):
        similarity_matrix = torch.mm(Wh, Wh.T)
        row = edge_index.size(0)
        for i in range(row):
            nonzero_index = torch.nonzero(edge_index[i])
            nonzere_index = torch.squeeze(nonzero_index, 1)
            adj_Wh = F.embedding(nonzere_index, Wh)
            adj_Wh_avg = torch.sum(adj_Wh,0)/adj_Wh.size(0)
            alpha = torch.matmul(Wh[i], adj_Wh_avg.T)
            # print(alpha)
            # print(similarity_matrix[i])
            similarity_matrix[i] = similarity_matrix[i] + alpha
            # print(similarity_matrix[i])

        return similarity_matrix


    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.input_dim) + ' -> ' + str(self.output_dim) + ')'


class NeighborEmbedding1(nn.Module):
    def __init__(self, num_embeddings, out_channels=128, dropout=0.5, cached=True, bias=True, lamda=0.8, share=True):
        super(NeighborEmbedding1, self).__init__()

        self.shutcut = nn.Linear(in_features=num_embeddings, out_features=out_channels)

        self.bgnn = BGCNA(in_channels=num_embeddings, out_channels=out_channels,
                          cached=cached, bias=bias, lamda=lamda, share=share)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = out_channels

    # 按边分batch
    def forward(self, x, edge_index, embedding):
        embedding = self.bgnn(embedding, edge_index=edge_index)
        embedding = self.dropout(embedding)

        drug_embedding = F.embedding(x[0, :], embedding)
        drug_embedding = F.normalize(drug_embedding)
        disease_embedding = F.embedding(x[1, :], embedding)
        disease_embedding = F.normalize(disease_embedding)

        return drug_embedding, disease_embedding


class NeighborEmbedding2(nn.Module):
    def __init__(self, num_embeddings, out_channels=128, dropout=0.5, cached=True, bias=True, lamda=0.8, share=True):
        super(NeighborEmbedding2, self).__init__()

        self.shutcut = nn.Linear(in_features=num_embeddings, out_features=out_channels)

        self.bgnn = BGCNA(in_channels=num_embeddings, out_channels=out_channels,
                          cached=cached, bias=bias, lamda=lamda, share=share)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = out_channels

    # 按边分batch后取出对应的边对应的节点
    def forward(self, x, y, edge_index, embedding):
        embedding = self.bgnn(embedding, edge_index=edge_index)
        embedding = self.dropout(embedding)

        drug_embedding = F.embedding(x, embedding)
        drug_embedding = F.normalize(drug_embedding)
        disease_embedding = F.embedding(y, embedding)
        disease_embedding = F.normalize(disease_embedding)

        return drug_embedding, disease_embedding

class InteractionEmbedding(nn.Module):
    def __init__(self, n_drug, n_disease, embedding_dim, dropout=0.5):
        super(InteractionEmbedding, self).__init__()
        self.drug_project = nn.Linear(n_drug, embedding_dim, bias=False)
        self.disease_project = nn.Linear(n_disease, embedding_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.output_dim = embedding_dim

    def forward(self, association_pairs, drug_embedding, disease_embedding):
        drug_embedding = torch.diag(torch.ones(drug_embedding.shape[0], device=drug_embedding.device))
        disease_embedding = torch.diag(torch.ones(disease_embedding.shape[0], device=disease_embedding.device))

        drug_embedding = self.drug_project(drug_embedding)
        disease_embedding = self.disease_project(disease_embedding)

        drug_embedding = F.embedding(association_pairs[0,:], drug_embedding)        # 从第二个位置中，按照第一个位置的取出相应索引。
        disease_embedding = F.embedding(association_pairs[1,:], disease_embedding)

        associations = drug_embedding*disease_embedding

        associations = F.normalize(associations)    # With the default arguments it uses the Euclidean norm over vectors along dimension 11 for normalization.
        associations = self.dropout(associations)
        return associations

class InteractionDecoder(nn.Module):
    def __init__(self, in_channels, hidden_dims=(256, 64), out_channels=1, dropout=0.5):
        super(InteractionDecoder, self).__init__()
        decoder = []
        in_dims = [in_channels]+list(hidden_dims)
        out_dims = hidden_dims
        for in_dim, out_dim in zip(in_dims, out_dims):
            decoder.append(nn.Linear(in_features=in_dim, out_features=out_dim))
            decoder.append(nn.ReLU(inplace=True))
            decoder.append(nn.Dropout(dropout))
        decoder.append(nn.Linear(hidden_dims[-1], out_channels))
        decoder.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        return self.decoder(x)


@MODEL_REGISTRY.register()
class CSL(BaseModel):
    DATASET_TYPE = "PairGraphDataset"
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("CSL model config")
        parser.add_argument("--embedding_dim", default=128, type=int, help="编码器关联嵌入特征维度")
        parser.add_argument("--neighbor_embedding_dim", default=64, type=int, help="编码器邻居特征维度")
        parser.add_argument("--hidden_dims", type=int, default=(128, 64), nargs="+", help="解码器每层隐藏单元数")
        parser.add_argument("--lr", type=float, default=5e-4)
        parser.add_argument("--dropout", type=float, default=0.4)
        parser.add_argument("--pos_weight", type=float, default=1.0, help="no used, overwrited, use for bce loss")
        parser.add_argument("--alpha", type=float, default=0.5, help="use for focal loss")
        parser.add_argument("--gamma", type=float, default=2.0, help="use for focal loss")
        parser.add_argument("--lamda", type=float, default=0.8, help="weight for bgnn")
        parser.add_argument("--loss_fn", type=str, default="focal", choices=["bce", "focal"])
        parser.add_argument("--separate", default=False, action="store_true")

        parser.add_argument("--get_dropout", type=float, default=0.0, help="dropout setting of GAT. default is 0.0")
        parser.add_argument("--negative_slope", type=float, default=1e-2, help="negative slopof LeadyReLU. default is 1e-2")
        return parent_parser

    def __init__(self, n_drug, n_disease, embedding_dim=64, neighbor_embedding_dim=32, hidden_dims=(64, 32),
                 lr=5e-4, dropout=0.5, pos_weight=1.0, alpha=0.5, gamma=2.0, lamda=0.8, loss_fn="focal", separate=False,
                 gat_dropout=0.0, negative_slope=1e-2, **config):
        super(CSL, self).__init__()
        # lr=0.1
        self.n_drug = n_drug
        self.n_disease = n_disease
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.gat_dropout = gat_dropout
        self.leakeyRelu_negative_slope = negative_slope

        #ssl
        self.num = 100
        self.K = 10
        self.dim_embedding = neighbor_embedding_dim

        self.register_buffer("pos_weight", torch.tensor(pos_weight))
        self.register_buffer("alpha", torch.tensor(alpha))
        self.register_buffer("gamma", torch.tensor(gamma))
        "rank bce mse focal"
        self.loss_fn_name = loss_fn
        share = not separate

        self.drug_embedding_compress = nn.Linear(n_drug, embedding_dim, bias=False)
        self.disease_embedding_compress = nn.Linear(n_disease, embedding_dim, bias=False)

        self.bgcn_encoder1 = NeighborEmbedding1(num_embeddings=embedding_dim,
                                              out_channels=neighbor_embedding_dim,
                                              dropout=dropout, lamda=lamda, share=share)
        self.bgcn_encoder2 = NeighborEmbedding2(num_embeddings=embedding_dim,
                                              out_channels=neighbor_embedding_dim,
                                              dropout=dropout, lamda=lamda, share=share)

        self.drug_att_encoder = GraphAttentionLayer(num_embeddings=n_drug,
                                                    out_channels=neighbor_embedding_dim,
                                                    dropout=gat_dropout, negative_slope=negative_slope,concat=False)
        self.disease_att_encoder = GraphAttentionLayer(num_embeddings=n_disease,
                                                       out_channels=neighbor_embedding_dim,
                                                       dropout=gat_dropout, negative_slope=negative_slope, concat=False)

        self.interaction_encoder = InteractionEmbedding(n_drug=n_drug, n_disease=n_disease,
                                                        embedding_dim=embedding_dim, dropout=dropout)

        # merged_dim = self.disease_bgcn_encoder.output_dim + self.drug_bgcn_encoder.output_dim + self.interaction_encoder.output_dim
        merged_dim = self.disease_att_encoder.output_dim + self.drug_att_encoder.output_dim + self.interaction_encoder.output_dim
        self.decoder = InteractionDecoder(in_channels=merged_dim, hidden_dims=hidden_dims, dropout=dropout)
        self.config = config
        self.lr = lr
        self.save_hyperparameters()


    def forward(self, interaction_pairs, drug_edge, disease_edge, drug_embedding, disease_embedding):

        drug_index = interaction_pairs[0, :].unique()
        disease_index = interaction_pairs[1, :].unique()

        interaction_tuple = (interaction_pairs, torch.ones(interaction_pairs.size(dim=1)), (drug_embedding.shape[0],disease_embedding.shape[0]))
        interaction_matrix = torch.sparse_coo_tensor(*interaction_tuple)
        big_matrix = constructNet(interaction_matrix.to_dense()).to_sparse()
        big_matrix_tuple = (big_matrix.indices(), big_matrix.values(), big_matrix.size())

        drug_bgcn_input_embedding = self.drug_embedding_compress(drug_embedding)
        disease_bgcn_input_embedding = self.disease_embedding_compress(disease_embedding)
        bgcn_input_embedding = torch.cat((drug_bgcn_input_embedding, disease_bgcn_input_embedding), 0)
        # drug_bgcn_embedding, disease_bgcn_embedding = self.bgcn_encoder(interaction_pairs, big_matrix, bgcn_input_embedding)  # 第二个位置的矩阵必须传一个方阵

        # 对比学习准备
        drug_bgcn_embedding, disease_bgcn_embedding = self.bgcn_encoder2(drug_index, disease_index, big_matrix, bgcn_input_embedding)


        drug_att_embedding = self.drug_att_encoder(drug_index, drug_edge, drug_embedding)
        disease_att_embedding = self.disease_att_encoder(disease_index, disease_edge, disease_embedding)
        # drug_att_embedding = self.drug_att_encoder(interaction_pairs[0, :], drug_edge, drug_embedding)
        # disease_att_embedding = self.disease_att_encoder(interaction_pairs[1, :], disease_edge, disease_embedding)


        # ssl
        # 以att为主要，以bgcn为辅助
        # pos_prob_bgcn_drug = self.SimilarityCompute(drug_bgcn_embedding, disease_bgcn_embedding)
        # pos_prob_bgcn_disease = pos_prob_bgcn_drug.T
        # # choose top-10 users as positive samples and randomly choose 10 users as negative and get their embedding
        # pos_drug, neg_drug, pos_disease, neg_disease = self.topk_func_random(pos_prob_bgcn_drug, pos_prob_bgcn_disease, drug_bgcn_embedding, disease_bgcn_embedding)
        # loss_drug = self.SSL_topk(drug_att_embedding, pos_disease, neg_disease) / drug_index.shape[0]
        # loss_disease = self.SSL_topk(disease_att_embedding, pos_drug, neg_drug) / disease_index.shape[0]


        # 以bgcn为主要，以att为辅助
        pos_prob_att_drug = self.SimilarityCompute(drug_att_embedding, disease_att_embedding)
        pos_prob_att_disease = pos_prob_att_drug.T
        # # choose top-10 users as positive samples and randomly choose 10 users as negative and get their embedding
        pos_drug, neg_drug, pos_disease, neg_disease = self.topk_func_random(pos_prob_att_drug, pos_prob_att_disease, drug_att_embedding, disease_att_embedding)
        loss_drug = self.SSL_topk(drug_bgcn_embedding, pos_disease, neg_disease) / drug_index.shape[0]
        loss_disease = self.SSL_topk(disease_bgcn_embedding, pos_drug, neg_drug) / disease_index.shape[0]


        interaction_embedding = self.interaction_encoder(interaction_pairs, drug_embedding, disease_embedding)


        # 重新将节点表示拼凑成边batch
        # x = interaction_pairs[0, :]
        # y = interaction_pairs[1, :]
        # drug_bgcn_final_embedding = F.embedding(x - x[0], drug_bgcn_embedding)
        # disease_bgcn_final_embedding = F.embedding(y, disease_bgcn_embedding)
        drug_bgcn_embedding, disease_bgcn_embedding = self.bgcn_encoder1(interaction_pairs, big_matrix, bgcn_input_embedding)
        # drug_att_embedding = F.embedding(x - x[0], drug_att_embedding)
        # disease_att_embedding = F.embedding(y, disease_att_embedding)


        # 以att为主要
        # embedding = torch.cat([drug_att_embedding, interaction_embedding, disease_att_embedding], dim=-1)
        # 以bgcn为主要
        embedding = torch.cat([drug_bgcn_embedding, interaction_embedding, disease_bgcn_embedding], dim=-1)
        # 去掉对比学习，两路嵌入简单融合
        # embedding = torch.cat([(drug_bgcn_embedding+drug_att_embedding)/2, interaction_embedding, (disease_bgcn_embedding+disease_att_embedding)/2], dim=-1)

        score = self.decoder(embedding)


        # return score.reshape(-1)
        return score.reshape(-1), loss_drug, loss_disease



    def ssl(self, embed1_view1, embed2_view1, embed1_view2):
        pos_prob_gcn = self.SimilarityCompute(embed1_view1, embed2_view1)
        pos_prob_att = self.SimilarityCompute(embed1_view2, embed1_view2)
        # choose top-10 users as positive samples and randomly choose 10 users as negative and get their embedding
        pos_emb_gcn, neg_emb_gcn, pos_emb_att, neg_emb_att = self.topk_func_random(pos_prob_gcn, pos_prob_att, embed1_view1,
                                                                                   embed1_view2)
        contrast_loss = self.SSL_topk(embed1_view1, pos_emb_gcn, neg_emb_gcn)
        contrast_loss += self.SSL_topk(embed1_view2, pos_emb_att, neg_emb_att)
        return contrast_loss


    def SimilarityCompute(self, userEmbeddingA, userEmbeddingB):
        similarity = torch.matmul(userEmbeddingA, userEmbeddingB.T)
        pos = torch.softmax(similarity, 0)
        return pos


    def topk_func_random(self, score1, score2, drug_emb, disease_emb):
        self.num = int(score1.shape[0]/2)
        self.K = int(score1.shape[0]/5)
        # print(f"num is {self.num}")
        # print(f"K is {self.K}")

        values, pos_drug_index = score1.topk(self.num, dim=0, largest=True, sorted=True)
        values, pos_disease_index = score2.topk(self.num, dim=0, largest=True, sorted=True)
        pos_drug = torch.FloatTensor(self.K, score1.size(1), self.dim_embedding).fill_(0)
        neg_drug = torch.FloatTensor(self.K, score1.size(1), self.dim_embedding).fill_(0)
        pos_disease = torch.FloatTensor(self.K, score2.size(1), self.dim_embedding).fill_(0)
        neg_disease = torch.FloatTensor(self.K, score2.size(1), self.dim_embedding).fill_(0)


        for i in torch.arange(self.K):
            pos_drug[i] = drug_emb[pos_drug_index[i]]
            pos_disease[i] = disease_emb[pos_disease_index[i]]
        random_slices = torch.randint(self.K, self.num, (self.K,))  # choose negative items
        for i in torch.arange(self.K):
            neg_drug[i] = drug_emb[pos_drug_index[random_slices[i]]]
            neg_disease[i] = disease_emb[pos_disease_index[random_slices[i]]]
        return pos_drug, neg_drug, pos_disease, neg_disease


    def SSL_topk(self, sess_emb, pos, neg):
        def score(x1, x2):
            return torch.sum(torch.mul(x1, x2), 2)

        pos = torch.reshape(pos, (sess_emb.size(0), self.K, self.dim_embedding))
        neg = torch.reshape(neg, (sess_emb.size(0), self.K, self.dim_embedding))
        pos_score = score(sess_emb.unsqueeze(1).repeat(1, self.K, 1), F.normalize(pos, p=2, dim=-1))
        neg_score = score(sess_emb.unsqueeze(1).repeat(1, self.K, 1), F.normalize(neg, p=2, dim=-1))
        pos_score = torch.sum(torch.exp(pos_score / 0.1), 1)
        neg_score = torch.sum(torch.exp(neg_score / 0.1), 1)
        con_loss = -torch.sum(torch.log(pos_score / (pos_score + neg_score)))
        return con_loss


    def loss_fn(self, predict, label, u, v, u_edge, v_edge, loss_drug = 0, loss_disease = 0, reduction="sum"):
        bce_loss = self.bce_loss_fn(predict, label, self.pos_weight)
        focal_loss = self.focal_loss_fn(predict, label, gamma=self.gamma, alpha=self.alpha)
        mse_loss = self.mse_loss_fn(predict, label, self.pos_weight)
        rank_loss = self.rank_loss_fn(predict, label)
        #
        u_graph_loss = self.graph_loss_fn(x=u, edge=u_edge, cache_name="ul",
                                          # topk=5,
                                          topk = self.config["drug_neighbor_num"],
                                          reduction=reduction)
        v_graph_loss = self.graph_loss_fn(x=v, edge=v_edge, cache_name="vl",
                                          # topk=5,
                                          topk = self.config["disease_neighbor_num"],
                                          reduction=reduction)
        graph_loss = u_graph_loss * self.lambda1 + v_graph_loss * self.lambda2


        loss = {}
        loss.update(bce_loss)
        loss.update(focal_loss)
        loss.update(mse_loss)
        loss.update(rank_loss)
        loss["loss_graph"] = graph_loss
        loss["loss_graph_u"] = u_graph_loss
        loss["loss_graph_v"] = v_graph_loss
        loss["loss"] = loss[f"loss_{self.loss_fn_name}"] + graph_loss + 0.1 * (loss_drug + loss_disease)
        # loss["loss"] = loss["loss"] + graph_loss

        return loss


    def step(self, batch:PairGraphData):
        interaction_pairs = batch.interaction_pair
        label = batch.label
        drug_edge = batch.u_edge
        disease_edge = batch.v_edge
        drug_embedding = batch.u_embedding
        disease_embedding = batch.v_embedding
        u = self.interaction_encoder.drug_project.weight.T
        v = self.interaction_encoder.disease_project.weight.T

        # predict= self.forward(interaction_pairs, drug_edge, disease_edge, drug_embedding, disease_embedding)
        predict , loss_drug, loss_disease = self.forward(interaction_pairs, drug_edge, disease_edge, drug_embedding, disease_embedding)     # 开启对比学习
        if not self.training:
            predict = predict[batch.valid_mask.reshape(*predict.shape)]
            label = label[batch.valid_mask]


        # ans = self.loss_fn(predict=predict, label=label, u=u, v=v, u_edge=drug_edge, v_edge=disease_edge)
        ans = self.loss_fn(predict=predict, label=label, u=u, v=v, u_edge=drug_edge, v_edge=disease_edge, loss_drug = loss_drug, loss_disease = loss_disease)       # 开启对比学习


        ans["predict"] = predict
        ans["label"] = label
        return ans


    def training_step(self, batch, batch_idx=None):
        return self.step(batch)


    def validation_step(self, batch, batch_idx=None):
        return self.step(batch)


    def configure_optimizers(self):
        optimizer = optim.Adam(lr=self.lr, params=self.parameters(), weight_decay=1e-4)
        lr_scheduler = optim.lr_scheduler.CyclicLR(optimizer,
                                                   base_lr=0.05*self.lr,    # Initial learning rate which is the lower boundary in the cycle for each parameter group.
                                                   max_lr=self.lr,          # Upper learning rate boundaries in the cycle for each parameter group.
                                                   gamma=0.95,              # Constant in ‘exp_range’ scaling function: gamma**(cycle iterations) Default: 1.0
                                                   mode="exp_range",
                                                   step_size_up=4,          # Number of training iterations in the increasing half of a cycle. Default: 2000
                                                   cycle_momentum=False)    # If True, momentum is cycled inversely to learning rate between ‘base_momentum’ and ‘max_momentum’. Default: True
        return [optimizer], [lr_scheduler]


    @property
    def lambda1(self):
        max_value = 0.125
        value = self.current_epoch/18.0*max_value
        return torch.tensor(value, device=self.device)


    @property
    def lambda2(self):
        max_value = 0.0625
        value = self.current_epoch / 18.0 * max_value
        return torch.tensor(value, device=self.device)



