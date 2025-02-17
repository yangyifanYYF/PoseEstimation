import torch
import torch.nn as nn
import torch.nn.functional as F
from model.losses import ChamferDis, PoseDis, SmoothL1Dis, ChamferDis_wo_Batch
from utils.data_utils import generate_augmentation
from model.modules import ModifiedResnet, PointNet2MSG
from model.Net_modules import InstanceAdaptiveKeypointDetector, GeometricAwareFeatureAggregator, PoseSizeEstimator, NOCS_Predictor, Reconstructor, LocalGlobal, FeatureFusion, InstanceAdaptiveKeypointDetector1, InstanceAdaptiveKeypointDetector2, Reconstructor1, InstanceAdaptiveKeypointDetector3
from transformers import CLIPProcessor, CLIPModel

class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()
        self.cat_num = cfg.cat_num
        self.fuse_type = cfg.fuse_type
        self.last_module = cfg.last_module
        self.cfg = cfg
        if cfg.rgb_backbone == "resnet":
            self.rgb_extractor = ModifiedResnet()
        elif cfg.rgb_backbone == 'dino':
            # frozen dino
            self.rgb_extractor = torch.hub.load('facebookresearch/dinov2','dinov2_vits14')
            for param in self.rgb_extractor.parameters():
                param.requires_grad = False

            self.feature_mlp = nn.Sequential(
                nn.Conv1d(384, 128, 1),
            )
        
        self.pts_extractor = PointNet2MSG(radii_list=[[0.01, 0.02], [0.02,0.04], [0.04,0.08], [0.08,0.16]])
        
        self.IAKD = InstanceAdaptiveKeypointDetector(cfg.IAKD)
        self.IAKD1 = InstanceAdaptiveKeypointDetector1(cfg.IAKD)
        self.IAKD2 = InstanceAdaptiveKeypointDetector2(cfg.IAKD)
        self.IAKD3 = InstanceAdaptiveKeypointDetector3(cfg.IAKD)
        self.GAFA = GeometricAwareFeatureAggregator(cfg.GAFA)

        self.nocs_predictor = NOCS_Predictor(cfg.NOCS_Predictor)
        self.estimator = PoseSizeEstimator()

        self.reconstructor = Reconstructor(cfg.Reconstructor)
        self.reconstructor1 = Reconstructor1(cfg.Reconstructor)
        self.LocalGlobal = LocalGlobal(cfg.LG)
        self.FeatureFusion = FeatureFusion(cfg.FF)
        if self.cfg.clip:
            self.processor = CLIPProcessor.from_pretrained("/workspace/code/AG-Pose-main/model/clip-vit-base-patch16")
            self.model = CLIPModel.from_pretrained("/workspace/code/AG-Pose-main/model/clip-vit-base-patch16")
            self.synset_names = ['bowl', 'camera', 'can', 'laptop', 'mug', 'bottle']
            self.description = ['The bowl has a circular rim, a rounded concave interior, and a gently curved or flat base.', 
                                "The camera's shape can be described as a compact rectangular prism with a cylindrical lens protruding from the front.", 
                                "A can is a cylindrical object with a circular base and uniform height, typically featuring smooth surfaces and well-defined edges at the top and bottom.",
                                "A laptop is a rectangular, flat, and foldable device with a hinged design, typically consisting of a screen on one side and a keyboard on the other.",
                                "The mug has a cylindrical shape with a handle attached to the side, typically with a flat base and a slightly curved rim.", 
                                "The bottle typically has a cylindrical shape with a narrow neck, a wider body, and a flat or slightly rounded base."]
            self.clip_mlp1 = nn.Sequential(
                nn.Conv1d(512, 128, 1),
            )
            self.clip_mlp2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        
        
    def forward(self, inputs):
        end_points = {}

        rgb = inputs['rgb']
        pts = inputs['pts']
        choose = inputs['choose']
        cls = inputs['category_label'].reshape(-1)

        c = torch.mean(pts, 1, keepdim=True)  
        pts = pts - c
        
        b = pts.size(0)
        index = cls + torch.arange(b, dtype=torch.long).cuda() * self.cat_num
        
        # rgb feat
        if self.cfg.rgb_backbone == 'resnet':
            rgb_local = self.rgb_extractor(rgb) 
        elif self.cfg.rgb_backbone == 'dino':
            dino_feature = self.rgb_extractor .forward_features(rgb)["x_prenorm"][:, 1:]  
            f_dim = dino_feature.shape[-1]
            num_patches =int(dino_feature.shape[1]**0.5)
            dino_feature = dino_feature.reshape(b, num_patches, num_patches, f_dim).permute(0,3,1,2)
            dino_feature = F.interpolate(dino_feature, size=(num_patches * 14, num_patches * 14), mode='bilinear', align_corners=False) 
            dino_feature = dino_feature.reshape(b, f_dim, -1) 
            rgb_local = self.feature_mlp(dino_feature)
        elif self.cfg.rgb_backbone == 'clip':
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])
            
            device = rgb.device
            std = std.to(device)
            mean = mean.to(device)

            # 反归一化公式：input[c] = output[c] * std[c] + mean[c]
            rgb_original = rgb * std.view(3, 1, 1) + mean.view(3, 1, 1)
            rgb_original = rgb_original.clamp(0, 1)  # 确保值在 [0, 1] 范围

            inputs = self.processor(images=rgb_original, return_tensors="pt", do_rescale=False)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                image_features = image_features.unsqueeze(2).repeat(1, 1, pts.size(1))
            rgb_local = self.clip_mlp1(image_features) # (b, c, n)
        else:
            raise NotImplementedError
        
        d = rgb_local.size(1)
        rgb_local = rgb_local.view(b, d, -1)
        choose = choose.unsqueeze(1).repeat(1, d, 1)
        rgb_local = torch.gather(rgb_local, 2, choose).contiguous() # b, c, n

        if self.training:
            delta_r, delta_t, delta_s = generate_augmentation(b)
            pts = (pts - delta_t) / delta_s.unsqueeze(2) @ delta_r

        pts_local = self.pts_extractor(pts) # b, c, n
        
        if self.cfg.clip:
            text = ['A photo of a ' + self.synset_names[i.item()] + '. ' + self.description[j.item()] 
                    for i, j in zip(cls.flatten(), cls.flatten())]
            inputs = self.processor(text=text, return_tensors="pt", padding=True)
            # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
            # 将inputs中的所有张量移动到设备上
            inputs = {key: value.cuda() for key, value in inputs.items()}
            with torch.no_grad():
                text_feature = self.model.get_text_features(**inputs)
                text_feature = text_feature.unsqueeze(2).repeat(1, 1, pts.size(1))
            text_feature = self.clip_mlp2(text_feature.transpose(1,2)).transpose(1,2)
            fused_feature = self.FeatureFusion(rgb_local, pts_local, text_feature=text_feature) # (b, n, 2c)
            
        elif self.cfg.cls_token:
            fused_feature = self.FeatureFusion(rgb_local, pts_local, cls=cls) # (b, n, 2c)
        else:
            if self.fuse_type == 'concat':
                fused_feature = torch.cat((pts_local, rgb_local), dim=1).transpose(1, 2) # (b, n, 2c)
            elif self.fuse_type == 'self_attn':
                fused_feature = self.FeatureFusion(rgb_local, pts_local) # (b, n, 2c)
        
        if self.cfg.first_module == 'IAKD':
            kpt_3d, kpt_feature = self.IAKD(fused_feature, pts)
            
        elif self.cfg.first_module == 'IAKD1':
            batch_kpt_query, heat_map = self.IAKD1(fused_feature, cls, pts)
            kpt_3d = torch.bmm(heat_map, pts) 
            kpt_feature = torch.bmm(heat_map, fused_feature)
        elif self.cfg.first_module == 'IAKD2':
            kpt_3d, kpt_feature = self.IAKD2(fused_feature, pts)
        elif self.cfg.first_module == 'IAKD3':
            kpt_3d, kpt_feature = self.IAKD3(fused_feature, cls, pts)  
        
        if self.last_module == 'LG':
            kpt_feature = self.LocalGlobal(kpt_feature, kpt_3d.detach(), fused_feature, pts)
        elif self.last_module == 'GAFA':
            kpt_feature = self.GAFA(kpt_feature, kpt_3d.detach(), fused_feature, pts)

        if self.cfg.reconstructor == 'Reconstructor':
            recon_model, recon_delta = self.reconstructor(kpt_3d.transpose(1, 2), kpt_feature.transpose(1, 2))
        elif self.cfg.reconstructor == 'Reconstructor1':
            recon_model, recon_delta = self.reconstructor1(kpt_3d.transpose(1, 2), kpt_feature.transpose(1, 2))
        kpt_nocs = self.nocs_predictor(kpt_feature, index)
        r, t, s = self.estimator(kpt_3d, kpt_nocs.detach(), kpt_feature)

        if self.training:
            end_points['recon_delta'] = recon_delta
            # if self.cfg.first_module != 'IAKD2' and self.cfg.first_module != 'IAKD1':
            #     end_points['pred_heat_map'] = heat_map
            end_points['pred_kpt_3d'] =  \
            (kpt_3d @ delta_r.transpose(1, 2)) * delta_s.unsqueeze(2) + delta_t + c
            end_points['recon_model'] =  \
            (recon_model.transpose(1, 2) @ delta_r.transpose(1, 2)) * delta_s.unsqueeze(2) + delta_t + c
            end_points['pred_kpt_nocs'] = kpt_nocs
            end_points['pred_translation'] = delta_t.squeeze(1) + delta_s * torch.bmm(delta_r, t.unsqueeze(2)).squeeze(2) + c.squeeze(1)
            end_points['pred_rotation'] = delta_r @ r
            end_points['pred_size'] = s * delta_s

        else:
            end_points['pred_translation'] = t + c.squeeze(1)
            end_points['pred_rotation'] = r
            end_points['pred_size'] = s
            end_points['pred_kpt_3d'] =  kpt_3d + c
            end_points['kpt_nocs'] = kpt_nocs

        return end_points

class Loss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
    def forward(self, endpoints):
        
        pts = endpoints['pts']
        b = pts.shape[0] 
        
        recon_delta = endpoints['recon_delta']
        pred_kpt_3d = endpoints['pred_kpt_3d']
        recon_model = endpoints['recon_model']
        
        translation_gt = endpoints['translation_label']
        rotation_gt = endpoints['rotation_label']
        size_gt = endpoints['size_label']
        
        # pose 
        loss_pose = PoseDis(self.cfg.sym, endpoints['pred_rotation'], endpoints['pred_translation'], endpoints['pred_size'], rotation_gt, translation_gt, size_gt, endpoints['category_label'], endpoints['mug_handle_visibility'])
        # cd
        if self.cfg.chamfer_dis_k2p:
            loss_cd = self.chamfer_dis_k2p(pts, pred_kpt_3d)
        else:
            loss_cd = self.cd_dis_k2p(pts, pred_kpt_3d)
        # nocs
        kpt_nocs_gt = (pred_kpt_3d - translation_gt.unsqueeze(1)) / (torch.norm(size_gt, dim=1).view(b, 1, 1) + 1e-8) @ rotation_gt
        loss_nocs = SmoothL1Dis(endpoints['pred_kpt_nocs'], kpt_nocs_gt)
        # div
        if self.cfg.diversity_loss_3d1:
            loss_diversity = self.diversity_loss_3d1(pred_kpt_3d)
        else:
            loss_diversity = self.diversity_loss_3d(pred_kpt_3d)
        # reconstruction
        if self.cfg.obj_aware:
            # recon_with_mask
            loss_recon = self.ChamferDis_with_mask(pts, recon_model, endpoints['pc_mask'])
        else:
            loss_recon = ChamferDis(pts, recon_model)
        # regularization 
        loss_delta = recon_delta.norm(dim=2).mean()
        
        # 尝试帕累托优化
        if self.cfg.Pareto:
            # 计算每个损失的梯度
            loss_list = [loss_pose, loss_nocs, loss_cd, loss_diversity, loss_recon, loss_delta]
            gradients = []
            for loss in loss_list:
                loss.backward(retain_graph=True)  # 计算每个损失的梯度
                # 只保存非空梯度的副本
                gradients.append([param.grad.clone() for param in self.parameters() if param.grad is not None])

            # 确保梯度列表不为空，并计算每个目标函数的总范数
            grad_norms = []
            for grad in gradients:
                # 过滤掉空梯度列表
                non_empty_grads = [g for g in grad if g is not None]
                if non_empty_grads:  # 只有在梯度非空时才计算范数
                    grad_norm = torch.sqrt(torch.sum(torch.stack([torch.norm(grad_param)**2 for grad_param in non_empty_grads]))).item()
                    grad_norms.append(grad_norm)
                else:
                    grad_norms.append(0.0)  # 如果梯度为空，设置范数为0

            # 计算目标函数梯度之间的内积，用于平衡各个目标
            grad_inner_product = torch.zeros(len(grad_norms), len(grad_norms))  # 用于保存梯度的内积
            for i in range(len(grad_norms)):
                for j in range(i, len(grad_norms)):
                    # 检查梯度是否存在
                    if len(gradients[i]) > 0 and len(gradients[j]) > 0:  # 确保每个梯度列表有元素
                        grad_inner_product[i, j] = torch.sum(gradients[i][0] * gradients[j][0]).item()  # 计算梯度内积

            # 计算MGDA权重（基于梯度内积）
            grad_inner_product_inv = torch.inverse(grad_inner_product + 1e-6 * torch.eye(len(grad_inner_product)))  # 计算梯度内积的逆
            mgda_weights = torch.matmul(grad_inner_product_inv, torch.ones(len(grad_inner_product), 1))  # 求得每个目标的权重

            # 归一化权重
            mgda_weights = mgda_weights / torch.sum(mgda_weights)
            mgda_weights = mgda_weights.squeeze().tolist()

            # 计算总损失
            loss_all = sum(mgda_weights[i] * loss_list[i] for i in range(len(loss_list)))

            return {
                'loss_all': loss_all,
                'loss_pose': mgda_weights[0] * loss_pose,
                'loss_nocs': mgda_weights[1] * loss_nocs,
                'loss_cd': mgda_weights[2] * loss_cd,
                'loss_diversity': mgda_weights[3] * loss_diversity,
                'loss_recon': mgda_weights[4] * loss_recon,
                'loss_delta': mgda_weights[5] * loss_delta,
            }



        else:
            loss_all = self.cfg.pose*loss_pose + self.cfg.nocs*loss_nocs + self.cfg.cd*loss_cd + \
                self.cfg.diversity*loss_diversity + self.cfg.recon*loss_recon + self.cfg.delta*loss_delta
            return {
                'loss_all': loss_all,
                'loss_pose': self.cfg.pose*loss_pose,
                'loss_nocs': self.cfg.nocs*loss_nocs,
                'loss_cd': self.cfg.cd*loss_cd,
                'loss_diversity': self.cfg.diversity*loss_diversity,
                'loss_recon': self.cfg.recon*loss_recon,
                'loss_delta': self.cfg.delta*loss_delta,
            }
        
    def ChamferDis_with_mask(self, pts, recon_model, pc_mask):
        """
        calculate ChamferDis with valid pointcloud mask
        Args:
            pts: (b, n1, 3)
            recon_model: (b, n2, 3)
            pc_mask: (b, n1)

        Return:
            recon_loss
        """
        b = pts.shape[0]
        is_first = True
        
        for idx in range(b):
            pts_ = pts[idx] # (n1, 3)
            pts_ = pts_[pc_mask[idx] == True] 
            if pts_.shape[0] == 0:
                print('warning: no valid point')
                continue
            recon_model_ = recon_model[idx] 
            dis = ChamferDis_wo_Batch(pts_, recon_model_)
            
            if is_first:
                dis_all = dis
                is_first = False
            else:
                dis_all += dis
        return dis_all / b

    def cd_dis_k2p(self, pts, pred_kpt_3d):
        """_summary_

        Args:
            pts (_type_): (b, n, 3)
            pred_kpt_3d (_type_): (b, kpt_num, 3)
        """
        # (b, n, 1, 3)   -   (b, 1, kpt_num, 3)  = (b, n, kpt_num, 3) -> (b, n, kpt_num)
        dis = torch.norm(pts.unsqueeze(2) - pred_kpt_3d.unsqueeze(1), dim=3)
        # (b, kpt_num)
        dis = torch.min(dis, dim=1)[0]
        return torch.mean(dis)
    
    def chamfer_dis_k2p(self, pts, pred_kpt_3d):
        """改进版：双向 Chamfer 距离"""
        dis1 = torch.norm(pts.unsqueeze(2) - pred_kpt_3d.unsqueeze(1), dim=3)  # (b, n, kpt_num)
        min_dis1 = torch.min(dis1, dim=1)[0]  # (b, kpt_num) - 点云到关键点

        dis2 = torch.norm(pred_kpt_3d.unsqueeze(2) - pts.unsqueeze(1), dim=3)  # (b, kpt_num, n)
        min_dis2 = torch.min(dis2, dim=2)[0]  # (b, kpt_num) - 关键点到点云
        loss = (torch.mean(min_dis1) + torch.mean(min_dis2)) / 2
        
        # # 计算密度：统计每个关键点在点云中的 `k` 近邻个数 (b, kpt_num)
        # k_neighbors = torch.topk(-dis2, 10, dim=1)[0]  # 取负值，找到最近 10 个点
        # density = torch.mean(k_neighbors, dim=1)  # 计算均值，表示局部密度
        # density = 1.0 / (density + 1e-6)  # 取密度的反比，密度低 → 权重高
        # density = density / density.sum(dim=1, keepdim=True)  # 归一化权重 (b, kpt_num)
        # loss = torch.sum(loss * density, dim=1)  # 计算加权平均

        return loss  # 计算双向平均

    
    def diversity_loss_3d(self, data):
        """_summary_

        Args:
            data (_type_): (b, kpt_num, 3)
        """
        threshold = self.cfg.th
        b, kpt_num = data.shape[0], data.shape[1]
        
        dis_mat = data.unsqueeze(2) - data.unsqueeze(1)
        # (b, kpt_num, kpt_num)
        dis_mat = torch.norm(dis_mat, p=2, dim=3, keepdim=False)
        
        dis_mat = dis_mat + torch.eye(kpt_num, device=dis_mat.device).unsqueeze(0)
        dis_mat[dis_mat >= threshold] = threshold
    
        # dis=0 -> loss=1     dis=threshold -> loss=0   y= -x/threshold + 1
        dis_mat = -dis_mat / threshold + 1
        
        loss = torch.sum(dis_mat, dim=[1, 2])
        loss = loss / (kpt_num * (kpt_num-1))
        
        return loss.mean()
    
    def diversity_loss_3d1(self, data):
        threshold = self.cfg.th
        b, kpt_num = data.shape[0], data.shape[1]

        dis_mat = data.unsqueeze(2) - data.unsqueeze(1)  # (b, kpt_num, kpt_num, 3)
        dis_mat = torch.norm(dis_mat, p=2, dim=3)  # (b, kpt_num, kpt_num)

        # 避免自距离影响计算（用 clone() 避免 in-place 操作）
        dis_mat = dis_mat.clone()
        dis_mat.diagonal(dim1=1, dim2=2).fill_(float('inf'))

        # 限制最大距离
        dis_mat = torch.clamp(dis_mat, max=threshold)

        # 计算 loss
        loss = torch.mean(dis_mat)

        return loss