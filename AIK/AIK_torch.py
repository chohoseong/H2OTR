import torch
import AIK.AIK_config as cfg
from pytorch3d.transforms import matrix_to_axis_angle

def axangle2mat(axis, angle, is_normalized=False):
    if not is_normalized:
        n = torch.sqrt(torch.sum(axis**2, dim=-1))
        axis = axis / torch.norm(axis, dim=-1).unsqueeze(-1)
    
    c = torch.cos(angle); s = torch.sin(angle); C = 1-c
    
    x,y,z = axis[:,0], axis[:,1], axis[:,2]
    xs = x*s;   ys = y*s;   zs = z*s
    xC = x*C;   yC = y*C;   zC = z*C
    xyC = x*yC; yzC = y*zC; zxC = z*xC
    
    return torch.stack([x*xC+c,   xyC-zs,   zxC+ys,xyC+zs,   y*yC+c,   yzC-xs, zxC-ys,   yzC+xs,   z*zC+c ], dim=-1).reshape(-1, 3,3)
        

def adaptive_IK(T_, P_):
    T = T_.clone().unsqueeze(0).repeat(P_.shape[0], 1, 1)
    P = P_.clone()
    
    R = {}
    R_pa_k = {}
    q = {}
    
    q[0] = T[:, 0].unsqueeze(-1)

    P_0 = torch.stack([P[:, 1] - P[:, 0], P[:, 5] - P[:, 0],
                        P[:, 9] - P[:, 0], P[:, 13] - P[:, 0],
                        P[:, 17] - P[:, 0]], axis=-1)

    T_0 = torch.stack([T[:, 1] - T[:, 0], T[:, 5] - T[:, 0],
                        T[:, 9] - T[:, 0], T[:, 13] - T[:, 0],
                        T[:, 17] - T[:, 0]], axis=-1)

    H = torch.matmul(T_0, P_0.permute(0, 2, 1))
    U, S, V = torch.svd(H)
    R0 = torch.matmul(V, U.permute(0,2,1))
    det0 = torch.det(R0)    
    
    V_ = V.clone()
    indices = (torch.abs(det0 + 1) < 1e-6) + (torch.abs(S) < 1e-4).sum(-1).to(torch.bool)
    V_[indices, :, 2] = -V_[indices, :, 2]
    R0 = torch.matmul(V_, U.transpose(1, 2))
    
    R[0] = R0
    R[1] = R[0].clone()
    R[5] = R[0].clone()
    R[9] = R[0].clone()
    R[13] = R[0].clone()
    R[17] = R[0].clone()

    for k in cfg.kinematic_tree:
        pa = cfg.SNAP_PARENT[k]
        pa_pa = cfg.SNAP_PARENT[pa]
        q[pa] = torch.matmul(R[pa], (T[:, pa] - T[:, pa_pa]).unsqueeze(-1)) + q[pa_pa]
        delta_p_k = torch.matmul(torch.inverse(R[pa]), P[:, k].unsqueeze(-1) - q[pa])
        delta_p_k = delta_p_k[...,0]

        delta_t_k = T[:, k] - T[:, pa]
        
        temp_axis = torch.cross(delta_t_k, delta_p_k)
        axis = temp_axis / (temp_axis.pow(2).sum(-1).pow(1/2) + 1e-8 ).unsqueeze(-1)
        temp = (torch.norm(delta_t_k, dim=-1) + 1e-8) * (torch.norm(delta_p_k,dim=-1) + 1e-8)
        cos_alpha = torch.matmul(delta_t_k.unsqueeze(1), delta_p_k.unsqueeze(-1))[:,0,0] / temp

        eps = 1e-7
        alpha = torch.acos(cos_alpha-eps)

        D_sw = axangle2mat(axis=axis, angle=alpha, is_normalized=False)
        R_pa_k[k] = D_sw
        R[k] = torch.matmul(R[pa], R_pa_k[k]).to(torch.float32)
    
    pose_R = torch.zeros((P_.shape[0], 16, 3, 3)).to(T_.device)
    pose_R[:, 0] = R[0]
    for key in cfg.ID2ROT.keys():
        value = cfg.ID2ROT[key]
        pose_R[:, value] = R_pa_k[key]
    
    pose_R = matrix_to_axis_angle(pose_R)
    return pose_R.reshape(pose_R.shape[0], -1)