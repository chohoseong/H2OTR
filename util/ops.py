import torch
from pytorch3d.ops.knn import knn_points
import numpy as np

def get_NN(src_xyz, trg_xyz, k=1):
    src_lengths = torch.full(
        (src_xyz.shape[0],), src_xyz.shape[1], dtype=torch.int64, device=src_xyz.device
    ) 
    trg_lengths = torch.full(
        (trg_xyz.shape[0],), trg_xyz.shape[1], dtype=torch.int64, device=trg_xyz.device
    )
    src_nn = knn_points(src_xyz, trg_xyz, lengths1=src_lengths, lengths2=trg_lengths, K=k) 
    nn_dists = src_nn.dists 

    return nn_dists

def get_pseudo_cmap(nn_dists):
    nn_dists = torch.sqrt(nn_dists) / 10.0  # turn into center-meter
    cmap = 1.0 - 2 * (torch.sigmoid(nn_dists*2) -0.5)
    return cmap

def pixel2cam(pixel_coord, f, c, T_=None):
    x = (pixel_coord[..., 0] - c[0]) / f[0] * pixel_coord[..., 2]
    y = (pixel_coord[..., 1] - c[1]) / f[1] * pixel_coord[..., 2]
    z = pixel_coord[..., 2]
    try:
        cam_coord = np.concatenate((x[...,None], y[...,None], z[...,None]), -1)
    except:
        cam_coord = torch.cat((x[...,None], y[...,None], z[...,None]), -1)
        
    if T_ is not None: # MANO space와 scale과 wrist를 맞추고자
        ratio = torch.linalg.norm(T_[:,9] - T_[:,0], dim=-1) / torch.linalg.norm(cam_coord[:,:,9] - cam_coord[:,:,0], dim=-1)
        cam_coord = cam_coord * ratio[:,:,None,None]  # template, m
        cam_coord = cam_coord - cam_coord[:, :, :1] + T_[:,:1]
    return cam_coord

def rigid_transform_3D(A, B):
    batch, n, dim = A.shape
    centroid_A = torch.mean(A, dim = 1)
    centroid_B = torch.mean(B, dim = 1)
    H = torch.matmul((A - centroid_A[:,None]).permute(0,2,1), B - centroid_B[:,None]) / n
    U, s, V = torch.svd(H)
    R = torch.matmul(V, torch.transpose(U, 1,2))

    negative_det = torch.linalg.det(R) < 0
    s[negative_det, -1] = -s[negative_det, -1]
    V[negative_det, :, 2] = -V[negative_det, :, 2]
    R[negative_det] = torch.matmul(V[negative_det], torch.transpose(U[negative_det],1,2))

    varP = torch.var(A, unbiased=False, dim=1).sum(-1)
    c = 1/varP * torch.sum(s, dim=-1) 

    t = -torch.matmul(c[:,None,None]*R, centroid_A[...,None])[...,-1] + centroid_B
    return c, R, t