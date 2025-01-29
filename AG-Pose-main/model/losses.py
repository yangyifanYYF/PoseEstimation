import torch
import torch.nn.functional as F

def SmoothL1Dis(p1, p2, threshold=0.1):
    '''
    p1: b*n*3
    p2: b*n*3
    '''
    diff = torch.abs(p1 - p2)
    less = torch.pow(diff, 2) / (2.0 * threshold)
    higher = diff - threshold / 2.0
    dis = torch.where(diff > threshold, higher, less)
    dis = torch.mean(torch.sum(dis, dim=2))
    return dis

def ChamferDis(p1, p2):
    '''
    p1: b*n1*3
    p2: b*n2*3
    '''
    dis = torch.norm(p1.unsqueeze(2) - p2.unsqueeze(1), dim=3)
    dis1 = torch.min(dis, 2)[0]
    dis2 = torch.min(dis, 1)[0]
    dis = 0.5*dis1.mean(1) + 0.5*dis2.mean(1)
    return dis.mean()

def ChamferDis_wo_Batch(p1, p2):
    """
    Args:
        p1: (n1, 3)
        p2: (n2, 3)
    """
    dis = torch.norm(p1.unsqueeze(1) - p2.unsqueeze(0), dim=2) # (n1, n2)
    dis1 = torch.min(dis, 1)[0] # (n1, )
    dis2 = torch.min(dis, 0)[0] # (n2, )
    dis = 0.5*dis1.mean() + 0.5*dis2.mean()
    return dis

def PoseDis(sym, r1, t1, s1, r2, t2, s2, category_label, mug_handle_visibility):
    '''
    r1, r2: b*3*3
    t1, t2: b*3
    s1, s2: b*3
    '''
    
    if sym:
        # Remove redundant dimensions from category_label
        category_label = category_label.squeeze(-1)  # Shape: (b,)
        mug_handle_visibility = mug_handle_visibility.squeeze(-1)  # Shape: (b,)

        # Symmetry adjustment for specific categories
        symmetric_mask = (category_label == 0) | (category_label == 1) | (category_label == 3) | ((category_label == 5) & (mug_handle_visibility == 0))
        if symmetric_mask.any():
            y_axis = torch.tensor([0, 1, 0], device=r1.device, dtype=r1.dtype).view(1, 3, 1)
            r1[symmetric_mask] = r1[symmetric_mask] @ y_axis
            r2[symmetric_mask] = r2[symmetric_mask] @ y_axis
     
    dis_r = torch.mean(torch.norm(r1 - r2, dim=1))
    dis_t = torch.mean(torch.norm(t1 - t2, dim=1))
    dis_s = torch.mean(torch.norm(s1 - s2, dim=1))

    return dis_r + dis_t + dis_s

def UniChamferDis(p1, p2):
    '''
    p1: b, n1, 3
    p2: b, n2, 3
    '''
    # (b, n1, n2)
    dis = torch.norm(p1.unsqueeze(2) - p2.unsqueeze(1), dim=3)
    dis = torch.min(dis, 2)[0]

    return dis.mean()

def normalize_keypoints(keypoint_positions):
    """
    归一化关键点，使不同大小的物体可比
    """
    center = keypoint_positions.mean(dim=1, keepdim=True)  # (B, 1, 3)
    scale = keypoint_positions.std(dim=1, keepdim=True) + 1e-6  # (B, 1, 3) 避免除零
    return (keypoint_positions - center) / scale

def procrustes_align(A, B):
    """
    使用 Procrustes Analysis 进行关键点对齐
    A, B: (num_kpts, 3) 关键点矩阵
    """
    # 1. 去均值，消除平移
    A_mean = A.mean(dim=0, keepdim=True)
    B_mean = B.mean(dim=0, keepdim=True)
    A_centered = A - A_mean
    B_centered = B - B_mean

    # 2. 计算最优旋转矩阵 R
    U, _, Vt = torch.svd(A_centered.T @ B_centered)  # SVD 分解
    R = U @ Vt.T  # 旋转矩阵

    # 3. 变换 B，使其与 A 对齐
    B_aligned = B_centered @ R.T + A_mean

    return B_aligned


def geometric_consistency_loss(keypoint_positions, labels):
    """
    约束相同类别物体的关键点结构保持一致
    """
    batch_size, num_kpts, _ = keypoint_positions.shape
    loss = 0
    count = 0

    # 归一化关键点，使不同大小的物体可比
    keypoint_positions = normalize_keypoints(keypoint_positions)

    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            if labels[i] == labels[j]:  # 仅计算相同类别的样本
                aligned_kpts_j = procrustes_align(keypoint_positions[i], keypoint_positions[j])
                R_ij = keypoint_positions[i] @ aligned_kpts_j.T  # 计算几何一致性损失
                identity = torch.eye(num_kpts, device=keypoint_positions.device)
                loss += F.mse_loss(R_ij, identity)
                count += 1

    return loss / max(count, 1)  # 避免除零错误

