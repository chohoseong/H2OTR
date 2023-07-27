# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

import torch
from torch import nn

from .HOP import build_HandObejctPoseEstimator
from .IA import build_interaction_recognizer
from manopth.manolayer import ManoLayer
from AIK import AIK_torch as AIK
import AIK.AIK_config as AIK_config 
from pytorch3d.ops.knn import knn_points
from util.ops import get_NN, get_pseudo_cmap, pixel2cam, rigid_transform_3D

import numpy as np

def rigid_transform_3D_numpy(A, B):
    batch, n, dim = A.shape
    tmp_A = A.copy()
    tmp_B = B.copy()
    centroid_A = np.mean(tmp_A, axis = 1)
    centroid_B = np.mean(tmp_B, axis = 1)
    H = np.matmul((tmp_A - centroid_A[:,None]).transpose(0,2,1), tmp_B - centroid_B[:,None]) / n
    U, s, V = np.linalg.svd(H)
    R = np.matmul(V.transpose(0,2,1), U.transpose(0, 2, 1))

    negative_det = np.linalg.det(R) < 0
    s[negative_det, -1] = -s[negative_det, -1]
    V[negative_det, :, 2] = -V[negative_det, :, 2]
    R[negative_det] = np.matmul(V[negative_det].transpose(0,2,1), U[negative_det].transpose(0, 2, 1))

    varP = np.var(tmp_A, axis=1).sum(-1)
    c = 1/varP * np.sum(s, axis=-1) 

    t = -np.matmul(c[:,None,None]*R, centroid_A[...,None])[...,-1] + centroid_B
    return c, R, t

class H2OTR(nn.Module):
    def __init__(self, HOP, IA, _mano_root, cfg):
        """ Initializes the model.
        Parameters:
            HOP: torch module of the HandObjectPoseEstimator to be used. See HOP.py
            IA: torch module of the InteractionRecognizer. See IA.py
        """
        super().__init__()
        self.HOP = HOP
        self.IA = IA
        self.cfg = cfg
        self.idx2obj = {v:k for k, v in cfg.obj2idx.items()}
                
        self.mano_left = ManoLayer(flat_hand_mean=True,
                        side="left",
                        mano_root=_mano_root,
                        use_pca=False,
                        root_rot_mode='axisang',
                        joint_rot_mode='axisang')

        self.mano_right = ManoLayer(flat_hand_mean=True,
                        side="right",
                        mano_root=_mano_root,
                        use_pca=False,
                        root_rot_mode='axisang',
                        joint_rot_mode='axisang')
        
        self.MANO_LAYER = [self.mano_left, self.mano_right] if cfg.dataset == 'H2O' else [self.mano_right]


    def forward(self, samples, intrinsics, obj_vertices, obj_bbox):
        cam_fx, cam_fy, cam_cx, cam_cy, w, h = intrinsics
        num_frame = samples.shape[0]
        
        ### pose estimate ###        
        HOP_outputs = self.HOP(samples)
        
        out_logits,  pred_keypoints, pred_obj_keypoints = HOP_outputs['pred_logits'], HOP_outputs['pred_keypoints'], HOP_outputs['pred_obj_keypoints']

        # labels, left_hand_uvd, right_hand_uvd, obj_uvd = self.query_select(out_logits,  pred_keypoints, pred_obj_keypoints)
        labels, hand_kp, obj_kp = self.query_select(out_logits,  pred_keypoints, pred_obj_keypoints)

        #### denormalize ###
        target_sizes = torch.tensor([w, h])[None,None].to(out_logits.device)
        hand_kp[...,:2] *=  target_sizes.unsqueeze(1); hand_kp[...,2] *= 1000
        obj_kp[...,:2] *=  target_sizes; obj_kp[...,2] *= 1000
        keypoints = torch.cat([hand_kp, obj_kp.unsqueeze(1)], dim=1)
        
        #### inverse kinematic ###
        T_keypoints_left, T_keypoints_right = AIK_config.T_keypoints_left.cuda(), AIK_config.T_keypoints_right.cuda()
        T_ = torch.stack([T_keypoints_left, T_keypoints_right]) if hand_kp.shape[1] == 2 else T_keypoints_right[None]
        hand_cam_align = pixel2cam(hand_kp, (cam_fx,cam_fy), (cam_cx,cam_cy), T_)

        pose_params = [AIK.adaptive_IK(t, hand_cam_align[:,i]) for i, t in enumerate(T_)]            
        pose_params = torch.cat(pose_params, dim=-1)
        
        mano_results = [mano_layer(pose_params[:,48*i:48*(i+1)]) for i, mano_layer in enumerate(self.MANO_LAYER)]
        hand_verts = torch.stack([m[0] for m in mano_results], dim=1)
        j3d_recon = torch.stack([m[1] for m in mano_results], dim=1)        
        hand_cam = pixel2cam(hand_kp, (cam_fx,cam_fy), (cam_cx,cam_cy))
        hand_verts = hand_verts - j3d_recon[:,:,:1] + hand_cam[:,:,:1]
        
        #### object 6D pose ###
        obj_label = labels[:,-1,:self.cfg.hand_idx[0]].sum(0).argmax(-1).item()
        obj_cam = pixel2cam(obj_kp,  (cam_fx, cam_fy), (cam_cx, cam_cy))
        _, R, t = rigid_transform_3D(obj_bbox[obj_label].unsqueeze(0)*1000, obj_cam)
        _, R2, _ = rigid_transform_3D_numpy(obj_bbox[obj_label].unsqueeze(0).detach().cpu().numpy()*1000, obj_cam.detach().cpu().numpy())
        obj_verts = torch.tensor(obj_vertices[obj_label], dtype=torch.float32)[None].repeat(R.shape[0], 1, 1).cuda()
        obj_verts = torch.matmul(R, obj_verts.permute(0,2,1)*1000).permute(0,2,1) + t[:,None]
        
        #### contact ####
        obj_nn_dist_affordance = get_NN(obj_verts.to(torch.float32), hand_verts.reshape(num_frame,-1,3).to(torch.float32))
        hand_nn_dist_affordance = torch.stack([get_NN(hand_verts[:,idx].to(torch.float32), obj_verts.to(torch.float32)) for idx in range(hand_verts.shape[1])], dim=1)
        obj_cmap_affordance = get_pseudo_cmap(obj_nn_dist_affordance)
        hand_cmap_affordance = torch.stack([get_pseudo_cmap(hand_nn_dist_affordance[:,idx]) for idx in range(hand_verts.shape[1])], dim=1)
        
        hand_info = torch.cat([hand_verts/1000, hand_cmap_affordance], dim=-1).to(torch.float32)
        obj_info = torch.cat([obj_verts/1000, obj_cmap_affordance], dim=-1).to(torch.float32)
                        
        action_logits = self.IA(hand_info, obj_info, labels[:,-1, 1:self.cfg.hand_idx[0]])
                
        outputs = {'keypoints' : keypoints,
                   'action_logits': action_logits,
                   'obj_name' : self.idx2obj[obj_label],
                   'obj_rot' : R,
                   'obj_trans' : t,
                   'hand_info' : hand_info
                   }
        return outputs
    

    def query_select(self, out_logits, pred_keypoints, pred_obj_keypoints):
        prob = out_logits.sigmoid()
        B, num_queries, num_classes = prob.shape
        # query index select
        best_score = torch.zeros(B).to(out_logits.device)
        obj_idx = torch.zeros(B).to(out_logits.device).to(torch.long)
        for i in range(1, self.cfg.hand_idx[0]):
            score, idx = torch.max(prob[:,:,i], dim=-1)
            obj_idx[best_score < score] = idx[best_score < score]
            best_score[best_score < score] = score[best_score < score]

        hand_idx = []
        for i in self.cfg.hand_idx:
            hand_idx.append(torch.argmax(prob[:,:,i], dim=-1)) 
        hand_idx = torch.stack(hand_idx, dim=-1)   
        keep = torch.cat([hand_idx, obj_idx[:,None]], dim=-1)
        hand_kp = torch.gather(pred_keypoints, 1, hand_idx.unsqueeze(-1).repeat(1,1,63)).reshape(B, -1 ,21, 3)
        obj_kp = torch.gather(pred_obj_keypoints, 1, obj_idx.unsqueeze(1).unsqueeze(1).repeat(1,1,63)).reshape(B, 21, 3)
        labels = torch.gather(out_logits, 1, keep.unsqueeze(2).repeat(1,1,num_classes)).softmax(dim=-1)
        return labels, hand_kp, obj_kp

def build(args, cfg):
    HOP = build_HandObejctPoseEstimator(args, cfg)
    IA = build_interaction_recognizer(args, cfg)
    
    _mano_root = 'mano/models'
    
    model = H2OTR(
        HOP,
        IA,
        _mano_root=_mano_root,
        cfg=cfg)

    return model
