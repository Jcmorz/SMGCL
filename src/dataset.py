from collections import namedtuple
from . import DATA_TYPE_REGISTRY
from .dataloader import Dataset
from .utils import select_topk,RWR
import torch
import torch.nn.functional as F

PairGraphData = namedtuple("PairGraphData", ["u_edge", "v_edge",
                                             "u_embedding", "v_embedding",
                                             "label", "interaction_pair", "valid_mask"])

@DATA_TYPE_REGISTRY.register()
class PairGraphDataset(Dataset):
    def __init__(self, dataset, mask, fill_unkown=True, stage="train", **kwargs):
        fill_unkown = fill_unkown if stage=="train" else False
        super(PairGraphDataset, self).__init__(dataset, mask, fill_unkown=fill_unkown, stage=stage, **kwargs)
        self.interaction_edge = self.interaction_edge
        self.label = self.label.reshape(-1)
        self.valid_mask = self.valid_mask.reshape(-1)
        self.u_edge = self.get_u_edge()
        self.v_edge = self.get_v_edge()

        # 构造interaction-aware similarity graph
        # interaction_tuple = (self.interaction_edge, torch.ones(self.interaction_edge.size(dim=1)),(self.size_u, self.size_v))
        # interaction_matrix = torch.sparse_coo_tensor(*interaction_tuple).to_dense()
        # drug_sim = torch.sparse_coo_tensor(*self.u_edge).to_dense()
        # dis_sim = torch.sparse_coo_tensor(*self.v_edge).to_dense()
        # A_prime_c = torch.multiply(interaction_matrix @ interaction_matrix.T, drug_sim).to_sparse()
        # A_prime_t = torch.multiply(interaction_matrix.T @ interaction_matrix, dis_sim).to_sparse()
        # self.u_edge = (A_prime_c.indices(), A_prime_c.values(), A_prime_c.size())
        # self.v_edge = (A_prime_t.indices(), A_prime_t.values(), A_prime_t.size())


        # # #DRWBNCF产生初始嵌入
        # self.u_embedding = select_topk(self.u_embedding, 20)
        # self.v_embedding = select_topk(self.v_embedding, 20)
        # # # self.u_embedding = torch.sparse_coo_tensor(*self.u_edge).to_dense()
        # # # self.v_embedding = torch.sparse_coo_tensor(*self.v_edge).to_dense()

        # # CSL产生初始嵌入: 执行带重启随机游走，生成药物和疾病节点的初始表示
        # # self.u_embedding = F.softmax(RWR(self.u_embedding, 0.9), dim=1)
        # # self.v_embedding = F.softmax(RWR(self.v_embedding, 0.9), dim=1)
        self.u_embedding = RWR(self.u_embedding, 0.1)
        self.v_embedding = RWR(self.v_embedding, 0.1)
        # # self.u_embedding = select_topk(self.u_embedding, 20)
        # # self.v_embedding = select_topk(self.v_embedding, 20)



    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        '''
        如果在类中定义了__getitem__()方法，那么它的实例对象（假设为P）就可以以P[key]形式取值，当实例对象做P[key]运算时，就会调用类中的__getitem__()方法。
        当对类的属性进行下标的操作时，首先会被__getitem__() 拦截，从而执行在__getitem__()方法中设定的操作
        '''
        label = self.label[index]
        interaction_edge = self.interaction_edge[:, index]
        valid_mask = self.valid_mask[index]
        data = PairGraphData(u_edge=self.u_edge,
                             v_edge=self.v_edge,
                             label=label,
                             valid_mask=valid_mask,
                             interaction_pair=interaction_edge,
                             u_embedding=self.u_embedding,
                             v_embedding=self.v_embedding,)
        return data