import torch
import torch.nn as nn
import torch.nn.functional as F

from rotation_utils import Ortho6d2Mat

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)

class Reconstructor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.pts_per_kpt = cfg.pts_per_kpt
        self.ndim = cfg.ndim
        
        self.pos_enc = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, self.ndim, 1),
        )
        
        self.mlp = nn.Sequential(
            nn.Conv1d(self.ndim, self.ndim, 1),
            nn.ReLU(),
            nn.Conv1d(self.ndim, self.ndim, 1),
        )
        
        self.shape_decoder = nn.Sequential(
            nn.Conv1d(2*self.ndim, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 3*self.pts_per_kpt, 1),
        )

    def forward(self, kpt_3d, kpt_feature):
        """
        Args:
            kpt_3d: (b, 3, kpt_num)
            kpt_feature: (b, c, kpt_num)

        Returns:
            recon_model: (b, 3, pts_per_kpt*kpt_num)
        """
        b = kpt_3d.shape[0]
        kpt_num = kpt_3d.shape[2]
        pos_enc_3d = self.pos_enc(kpt_3d) # (b, c, kpt_num)
        kpt_feature = self.mlp(kpt_feature) # (b, c, kpt_num)
        
        global_feature = torch.mean(pos_enc_3d + kpt_feature, dim=2, keepdim=True) # (b, c, 1)
        recon_feature = torch.cat([global_feature.repeat(1, 1, kpt_num), kpt_feature], dim=1) # (b, 2c, kpt_num)        
        # (b, 3*pts_per_kpt, kpt_num)
        recon_delta = self.shape_decoder(recon_feature)
        # (b, pts_per_kpt*kpt_num, 3)
        recon_delta = recon_delta.transpose(1, 2).reshape(b, kpt_num*self.pts_per_kpt, 3).contiguous()
        # (b, pts_per_kpt*kpt_num, 3)
        kpt_3d_interleave = kpt_3d.transpose(1, 2).repeat_interleave(self.pts_per_kpt, dim=1).contiguous()
        
        recon_model = (recon_delta + kpt_3d_interleave).transpose(1, 2).contiguous()
        return recon_model, recon_delta

class LocalGlobal(nn.Module):
    def __init__(self, cfg):
        super(LocalGlobal, self).__init__()
        self.K = cfg.K
        self.d_model = cfg.d_model
        # build attention layer
        self.attn_layer = LGAttnLayer(cfg.AttnLayer)
        self.fc_delta1 = nn.Sequential(
            nn.Linear(cfg.d_model*2, cfg.d_model),
            nn.ReLU(),
            nn.Linear(cfg.d_model, cfg.d_model)
        )
        self.fc_delta2 = nn.Sequential(
            nn.Linear(cfg.d_model*2, cfg.d_model),
            nn.ReLU(),
            nn.Linear(cfg.d_model, cfg.d_model)
        )
        self.fc_delta_3 = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, cfg.d_model),
            nn.ReLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.fc_delta_4 = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, cfg.d_model),
            nn.ReLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
    
    def forward(self, kpt_feature, kpt_3d, obj_feature, pts):
        """_summary_

        Args:
            kpt_feature (_type_): (b, kpt_num, 2c)
            kpt_3d (_type_): (b, kpt_num, 3)
            obj_feature (_type_): (b, n, 2c)
            pts (_type_): (b, n, 3)

        Returns:
            kpt_feature: (b, kpt_num, 2c)
        """

        # (b, kpt_num, 1, 3) - (b, 1, n, 3) = (b, kpt_num, n, 3)     
        dis_mat = torch.norm(kpt_3d.unsqueeze(2) - pts.unsqueeze(1), dim=3)
        knn_idx = dis_mat.argsort()[:, :, :self.K]
        knn_xyz = index_points(pts, knn_idx) # # (b, kpt_num, k, 3) 
        knn_feature = index_points(obj_feature, knn_idx)  # (b, kpt_num, k, 2c)
        
        kpt_3d = self.fc_delta_3(kpt_3d) # (b, kpt_num, 2c)
        knn_xyz = self.fc_delta_4(knn_xyz) # (b, kpt_num, k, 2c)
        
        kpt_combined = torch.cat([kpt_feature, kpt_3d], dim=-1) # (b, kpt_num, 4c)
        knn_combined = torch.cat([knn_feature, knn_xyz], dim=-1) # (b, kpt_num, k, 4c)
        kpt_combined = self.fc_delta1(kpt_combined)
        knn_combined = self.fc_delta2(knn_combined)
        output = self.attn_layer(kpt_feature, kpt_combined, obj_feature, knn_combined)
        
        
        
        return output

class LGAttnBlock(nn.Module):
    def __init__(self, d_model=256, num_heads=4, dim_ffn=256, dropout=0.0, dropout_attn=None):
        super(LGAttnBlock, self).__init__()
        if dropout_attn is None:
            dropout_attn = dropout
        self.multihead_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout, inplace=False)
        self.dropout2 = nn.Dropout(dropout, inplace=False)
        self.dropout3 = nn.Dropout(dropout, inplace=False)
        self.ffn = nn.Sequential(nn.Linear(d_model, dim_ffn),
                                 nn.ReLU(), nn.Dropout(dropout, inplace=False),
                                 nn.Linear(dim_ffn, d_model))
        self.fuse_mlp = nn.Sequential(
            nn.Linear(2*d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos
    
    
    def forward(self, kpt_feature, kpt_combined, obj_feature, knn_combined):
        """_summary_

        Args:
            kpt_feature (_type_): (b, kpt_num, 2c)
            kpt_combined (_type_): (b, kpt_num, 2c)
            obj_feature (_type_): (b, n, 2c)
            knn_combined (_type_): (b, kpt_num, k, 2c)
        """
        kpt_num = kpt_combined.shape[1]
        k = knn_combined.shape[2]
        cc = kpt_combined.shape[-1]
        # cross-attn
        kpt_combined = (kpt_combined.unsqueeze(2).repeat(1, 1, k, 1)).view(-1, kpt_num*k, cc)  # (b, kpt_num*k, 2c)
        kpt_combined1 = self.norm1(kpt_combined) # (b, kpt_num*k, 2c)
        knn_combined = knn_combined.view(-1, kpt_num*k, cc) # (b, kpt_num*k, 2c)
        
        output, _ = self.multihead_attn(query=kpt_combined1,
                                   key=knn_combined,
                                   value=knn_combined
                                  )
        kpt_combined = kpt_combined + self.dropout1(output) # (b, kpt_num*k, 2c)
        kpt_combined = kpt_combined.view(-1, kpt_num, k * cc)  # (b, kpt_num, k * 2c)
        kpt_combined = self.fc(kpt_combined) # (b, kpt_num, 2c)
        
        # self-attn
        # 对 kpt_combined 进行平均池化并广播，得到全局特征
        kpt_global = torch.mean(kpt_combined, dim=1)  # (b, 2c)
        kpt_global = kpt_global.unsqueeze(1).expand(-1, kpt_num, -1)  # (b, kpt_num, 2c)
        input = self.fuse_mlp(torch.cat([kpt_combined, kpt_global], dim=-1)) # (b, kpt_num, 2c)
        input1 = self.norm2(input)
        output, _ = self.self_attn(input1, 
                                               input1, 
                                               value=input1
                                                )
        kpt_feature = input + self.dropout2(output) # (b, kpt_num, 2c)
        
        # ffn
        kpt_feature2 = self.norm3(kpt_feature)
        kpt_feature2 = self.ffn(kpt_feature2)
        kpt_feature2 = kpt_feature + self.dropout3(kpt_feature2)
        
        return kpt_feature2

class LGAttnLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.block_num = cfg.block_num
        self.d_model = cfg.d_model
        self.num_head = cfg.num_head
        self.dim_ffn = cfg.dim_ffn
        
        # build attention blocks
        self.attn_blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.attn_blocks.append(LGAttnBlock(d_model=self.d_model, num_heads=self.num_head, dim_ffn=self.dim_ffn, dropout=0.0, dropout_attn=None))

        
    def forward(self, kpt_feature, kpt_combined, obj_feature, knn_combined):

        
        for i in range(self.block_num):
            kpt_feature = self.attn_blocks[i](kpt_feature, kpt_combined, obj_feature, knn_combined)
        
        return kpt_feature

class GeometricAwareFeatureAggregator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.block_num = cfg.block_num
        self.K = cfg.K
        self.d_model = cfg.d_model
        
        assert self.K.__len__() == self.block_num
        
        # build GAFA blocks
        self.GAFA_blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.GAFA_blocks.append(GAFA_block(self.K[i], self.d_model))
        
    def forward(self, kpt_feature, kpt_3d, pts_feature, pts):
        """
        Args:
            kpt_feature: (b, kpt_num, dim)
            kpt_3d: (b, kpt_num, 3)
            pts_feature: (b, n, dim)
            pts: (b, n, 3)

        Returns:
            kpt_feature: (b, kpt_num, dim)
        """
        for i in range(self.block_num):
            kpt_feature = self.GAFA_blocks[i](kpt_feature, kpt_3d, pts_feature, pts)

        return kpt_feature


class GAFA_block(nn.Module):
    def __init__(self, k, d_model):
        super().__init__()
        self.k = k
        self.d_model = d_model
        
        self.fc_in = nn.Sequential(
            nn.Conv1d(d_model, d_model, 1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, 1),
        )
        
        self.fc_delta = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.fc_delta_1 = nn.Sequential(
            nn.Linear(d_model*2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        self.fc_delta_l = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        
        self.fc_delta_abs = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        
        self.fuse_mlp = nn.Sequential(
            nn.Conv1d(3*d_model, d_model, 1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, 1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
        )
        
        self.out_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        self.relu = nn.ReLU()
        self.tau = 5.0
    
    def forward(self, kpt_feature, kpt_3d, pts_feature, pts):
        """_summary_

        Args:
            kpt_feature (_type_): (b, kpt_num, 2c)
            kpt_3d (_type_): (b, kpt_num, 3)
            pts_feature (_type_): (b, n, 2c)
            pts (_type_): (b, n, 3)

        Returns:
            kpt_feature: (b, kpt_num, 2c)
        """
        # (b, kpt_num, 1, 3) - (b, 1, n, 3) = (b, kpt_num, n, 3)     
        dis_mat = torch.norm(kpt_3d.unsqueeze(2) - pts.unsqueeze(1), dim=3)
        knn_idx = dis_mat.argsort()[:, :, :self.k]
        knn_xyz = index_points(pts, knn_idx)
        knn_feature = index_points(pts_feature, knn_idx)
        
        # (b, kpt_num, 2c)
        pre = kpt_feature 
        kpt_feature = self.fc_in(kpt_feature.transpose(1, 2))
        
        # (b, kpt_num, 1, 3) - (b, kpt_num, k, 3) = (b, kpt_num, k, 3)
        pos_enc = kpt_3d.unsqueeze(2) - knn_xyz
        pos_enc = self.fc_delta(pos_enc)
        
        # (b, kpt_num, k, 2c)
        knn_feature = self.fc_delta_1(torch.cat([knn_feature, pos_enc], dim=-1))
        
        sim = F.cosine_similarity(kpt_feature.transpose(1, 2).unsqueeze(2).repeat(1, 1, self.k, 1), knn_feature, dim=-1)
        sim = F.softmax(sim / self.tau, dim=2)
        # (b, kpt_num, 1, k) @ (b, kpt_num, k, 2c) -> (b, kpt_num, 1, 2c) -> (b, kpt_num, 2c)
        kpt_feature = torch.matmul(sim.unsqueeze(2), knn_feature).squeeze(2)
        kpt_feature = F.relu(kpt_feature + pre)
        
        # (b, kpt_num, 2c)
        pre = kpt_feature
        pos_enc_abs = self.fc_delta_abs(kpt_3d)
        
        # (b, kpt_num, 1, 3) - (b, 1, kpt_num, 3) = (b, kpt_num, kpt_num, 3)
        dis_mat_l = kpt_3d.unsqueeze(2) - kpt_3d.unsqueeze(1)
        pos_enc_abs_l = self.fc_delta_l(dis_mat_l)
        pos_enc_abs_l = torch.mean(pos_enc_abs_l, dim=2)
        
        # (b, kpt_num, 2c)
        pos_enc_l = pos_enc_abs + pos_enc_abs_l
        kpt_num = kpt_3d.shape[1]
        # (b, 1, 2c)
        kpt_global = torch.mean(kpt_feature, dim=1, keepdim=True)
        # (b, 4c, kpt_num) -> (b, 2c, kpt_num)
        kpt_feature = self.fuse_mlp(torch.cat([kpt_feature.transpose(1, 2), kpt_global.transpose(1, 2).repeat(1, 1, kpt_num), pos_enc_l.transpose(1, 2)], dim=1))
        # (b, kpt_num, 2c)
        kpt_feature = F.relu(kpt_feature.transpose(1, 2) + pre)
        pre = kpt_feature
        kpt_feature = self.out_mlp(kpt_feature)
        
        return self.relu(pre + kpt_feature)
    
class NOCS_Predictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.bins_num = cfg.bins_num
        self.cat_num = cfg.cat_num
        bin_lenth = 1 / self.bins_num
        half_bin_lenth = bin_lenth / 2
        self.bins_center = torch.linspace(start=-0.5, end=0.5-bin_lenth, steps=self.bins_num).view(self.bins_num, 1).cuda()
        self.bins_center = self.bins_center + half_bin_lenth # (bins_num, 1)
        
        self.nocs_mlp = nn.Sequential(
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
        )
        
        self.self_attn_layer = SelfAttnLayer(cfg.AttnLayer)
        
        self.atten_mlp = nn.Sequential(
            nn.Conv1d(256, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, self.cat_num*3*self.bins_num, 1), 
        )
    
    def forward(self, kpt_feature, index):
        """_summary_

        Args:
            kpt_feature: (b, kpt_num, dim)
        Return:
            kpt_nocs:           (b, kpt_num, 3)
        """
        b, kpt_num, c = kpt_feature.shape
        kpt_feature = self.nocs_mlp(kpt_feature.transpose(1, 2)).transpose(1, 2) 
        kpt_feature = self.self_attn_layer(kpt_feature)
        # (b, self.cat_num*3*bins_num, kpt_num)
        attn = self.atten_mlp(kpt_feature.transpose(1, 2)) 
        attn = attn.view(b*self.cat_num, 3*self.bins_num, kpt_num).contiguous()
        attn = torch.index_select(attn, 0, index)
        attn = attn.view(b, 3, self.bins_num, kpt_num).contiguous()
        attn = F.softmax(attn, dim=2)
        attn = attn.permute(0, 3, 1, 2)
        kpt_nocs = torch.matmul(attn, self.bins_center).squeeze(-1)

        return kpt_nocs
        
class AttnBlock(nn.Module):
    def __init__(self, d_model=256, num_heads=4, dim_ffn=256, dropout=0.0, dropout_attn=None):
        super(AttnBlock, self).__init__()
        if dropout_attn is None:
            dropout_attn = dropout
        self.multihead_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout, inplace=False)
        self.dropout2 = nn.Dropout(dropout, inplace=False)
        self.dropout3 = nn.Dropout(dropout, inplace=False)
        self.ffn = nn.Sequential(nn.Linear(d_model, dim_ffn),
                                 nn.ReLU(), nn.Dropout(dropout, inplace=False),
                                 nn.Linear(dim_ffn, d_model))
        
    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos
    
    
    def forward(self, kpt_query, input_feature):
        # cross-attn
        kpt_query2 = self.norm1(kpt_query)
        kpt_query2, attn = self.multihead_attn(query=kpt_query2,
                                   key=input_feature,
                                   value=input_feature
                                  )
        kpt_query = kpt_query + self.dropout1(kpt_query2)
        
        # self-attn
        kpt_query2 = self.norm2(kpt_query)
        kpt_query2, _ = self.self_attn(kpt_query2, 
                                               kpt_query2, 
                                               value=kpt_query2
                                                )
        kpt_query = kpt_query + self.dropout2(kpt_query2)
        
        # ffn
        kpt_query2 = self.norm3(kpt_query)
        kpt_query2 = self.ffn(kpt_query2)
        kpt_query = kpt_query + self.dropout3(kpt_query2)
        
        return kpt_query, attn
    
class AttnLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.block_num = cfg.block_num
        self.d_model = cfg.d_model
        self.num_head = cfg.num_head
        self.dim_ffn = cfg.dim_ffn
        
        # build attention blocks
        self.attn_blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.attn_blocks.append(AttnBlock(d_model=self.d_model, num_heads=self.num_head, dim_ffn=self.dim_ffn, dropout=0.0, dropout_attn=None))

        
    def forward(self, batch_kpt_query, input_feature):
        """
            update kpt_query to instance-specific queries
        Args:
            batch_kpt_query: b, kpt_num, dim
            input_feature: b, dim, n

        Returns:
            batch_kpt_query: b, kpt_num, dim
            attn:  b, kpt_num, n
        """     
        # (b, kpt_num, c)  (b, kpt_num, n)
        for i in range(self.block_num):
            batch_kpt_query, attn = self.attn_blocks[i](batch_kpt_query, input_feature)
        
        return batch_kpt_query, attn       
    
class SelfAttnBlock(nn.Module):
    def __init__(self, d_model=256, num_heads=4, dim_ffn=256, dropout=0.0, dropout_attn=None):
        super().__init__()
        if dropout_attn is None:
            dropout_attn = dropout
            
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout, inplace=False)
        self.dropout2 = nn.Dropout(dropout, inplace=False)
        self.ffn = nn.Sequential(nn.Linear(d_model, dim_ffn),
                                 nn.ReLU(), nn.Dropout(dropout, inplace=False),
                                 nn.Linear(dim_ffn, d_model))
        
    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos
    
    
    def forward(self, kpt_query):
        
        # self-attn
        kpt_query2 = self.norm1(kpt_query)
        kpt_query2, _ = self.self_attn(kpt_query2, 
                                               kpt_query2, 
                                               value=kpt_query2
                                                )
        kpt_query = kpt_query + self.dropout1(kpt_query2)
        
        # ffn
        kpt_query2 = self.norm2(kpt_query)
        kpt_query2 = self.ffn(kpt_query2)
        kpt_query = kpt_query + self.dropout2(kpt_query2)
        
        return kpt_query
    
class SelfAttnLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.block_num = cfg.block_num
        self.d_model = cfg.d_model
        self.num_head = cfg.num_head
        self.dim_ffn = cfg.dim_ffn
        
        # build attention blocks
        self.attn_blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.attn_blocks.append(SelfAttnBlock(d_model=self.d_model, num_heads=self.num_head, dim_ffn=self.dim_ffn, dropout=0.0, dropout_attn=None))

        
    def forward(self, batch_kpt_query):
        """
        Args:
            batch_kpt_query: b, kpt_num, dim
            
        Returns:
            batch_kpt_query: b, kpt_num, dim
        """    
        # (b, kpt_num, c)  (b, kpt_num, n)
        for i in range(self.block_num):
            batch_kpt_query = self.attn_blocks[i](batch_kpt_query)
        
        return batch_kpt_query    
    
class InstanceAdaptiveKeypointDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.kpt_num = cfg.kpt_num
        self.query_dim = cfg.query_dim
        # initialize shared kpt_query for all categories
        self.kpt_query = nn.Parameter(torch.empty(self.kpt_num, self.query_dim)) # (kpt_num, query_dim)
        nn.init.xavier_normal_(self.kpt_query)
        
        # build attention layer
        self.attn_layer = AttnLayer(cfg.AttnLayer)
        
    def forward(self, fused_feature):
        """_summary_

        Args:
            fused_feature (_type_): (b, n, 2c)
            cls:                (b, )
        """
        b, n, _ = fused_feature.shape
        
        input_feature = fused_feature.transpose(1,2)  # (b, 2c, n)
        
        batch_kpt_query = self.kpt_query.unsqueeze(0).repeat(b, 1, 1)
        # (b, kpt_num, 2c)  (b, kpt_num, n)
        batch_kpt_query, attn = self.attn_layer(batch_kpt_query, fused_feature) 
        
        # cos similarity <a, b> / |a|*|b|
        norm1 = torch.norm(batch_kpt_query, p=2, dim=2, keepdim=True) 
        norm2 = torch.norm(input_feature, p=2, dim=1, keepdim=True) 
        heatmap = torch.bmm(batch_kpt_query, input_feature) / (norm1 * norm2 + 1e-7)
        heatmap = F.softmax(heatmap / 0.1, dim=2)
        
        return batch_kpt_query, heatmap
        
class PoseSizeEstimator(nn.Module):
    def __init__(self):
        super(PoseSizeEstimator, self).__init__()
        
        self.pts_mlp1 = nn.Sequential(
            nn.Conv1d(3, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 1),
            nn.ReLU(),
        )
        self.pts_mlp2 = nn.Sequential(
            nn.Conv1d(3, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 1),
            nn.ReLU(),
        )
        self.pose_mlp1 = nn.Sequential(
            nn.Conv1d(64+64+256, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
        )
        self.pose_mlp2 = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.rotation_estimator = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 6),
        )
        self.translation_estimator = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )
        self.size_estimator = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )
        
    def forward(self, pts1, pts2, pts1_local):
        pts1 = self.pts_mlp1(pts1.transpose(1,2))
        pts2 = self.pts_mlp2(pts2.transpose(1,2))
        pose_feat = torch.cat([pts1, pts1_local.transpose(1,2), pts2], dim=1) # b, c, n

        pose_feat = self.pose_mlp1(pose_feat) # b, c, n
        pose_global = torch.mean(pose_feat, 2, keepdim=True) # b, c, 1
        pose_feat = torch.cat([pose_feat, pose_global.expand_as(pose_feat)], 1) # b, 2c, n
        pose_feat = self.pose_mlp2(pose_feat).squeeze(2) # b, c

        r = self.rotation_estimator(pose_feat)
        r = Ortho6d2Mat(r[:, :3].contiguous(), r[:, 3:].contiguous()).view(-1,3,3)
        t = self.translation_estimator(pose_feat)
        s = self.size_estimator(pose_feat)
        return r,t,s
    
class FeatureFusion(nn.Module):
    def __init__(self, cfg):
        super(FeatureFusion, self).__init__()
        self.attn_layer = SelfAttnLayer(cfg.AttnLayer)
        
    def forward(self, rgb_local, pts_local):
        """_summary_

        Args:
            rgb_local (_type_): (b, c, n)
            pts_local (_type_): (b, c, n)

        Returns:
            fused_feature: (b, c, n)
        """
        fused_feature = torch.cat((pts_local, rgb_local), dim=1).transpose(1, 2)  # (b, n, 2c)
        fused_feature = self.attn_layer(fused_feature) # (b, n, 2c)
        return fused_feature # (b, n, 2c)