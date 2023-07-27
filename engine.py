import os
import torch
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
import trimesh
from util.vis import visualize, visualize_obj, put_text_on_image
from util.ops import get_NN, get_pseudo_cmap

def demo(model, samples, intrinsics, orig_img, obj_vertices, obj_bbox, cfg):
    
    model.eval()
    idx2action = {v:k for k, v in cfg.action2idx.items()}
    with torch.no_grad():
        outputs = model(samples, intrinsics, obj_vertices, obj_bbox)               
        # model output
        pose, action_logits, obj_name, obj_rot, obj_trans, hand_info = outputs['keypoints'], outputs['action_logits'], outputs['obj_name'],\
                                                                        outputs['obj_rot'], outputs['obj_trans'], outputs['hand_info']
                                                                        
        action_class = action_logits[0].argmax()

        if cfg.dataset=='H2O':
            GT_obj_mesh = trimesh.load(f'{cfg.object_model_path}/{obj_name}/{obj_name}.obj')
        else:
            GT_obj_mesh = trimesh.load(f'{cfg.object_model_path}/{obj_name}_model/{obj_name}_model.ply')
            
        obj_verts = torch.tensor(GT_obj_mesh.vertices, dtype=torch.float32)[None].repeat(samples.shape[0], 1, 1).to(samples.device)
        obj_verts = torch.matmul(obj_rot.to(torch.float32), obj_verts.permute(0, 2, 1)*1000).permute(0, 2, 1) + obj_trans[:,None]
        hand_verts = hand_info[...,:-1]*1000
        
        obj_nn_dist_affordance = get_NN(obj_verts.to(torch.float32), hand_verts.reshape(samples.shape[0],-1,3).to(torch.float32))
        hand_nn_dist_affordance = torch.stack([get_NN(hand_verts[:,idx].to(torch.float32), obj_verts.to(torch.float32)) for idx in range(hand_verts.shape[1])], dim=1)
        obj_cmap_affordance = get_pseudo_cmap(obj_nn_dist_affordance)
        hand_cmap_affordance = torch.stack([get_pseudo_cmap(hand_nn_dist_affordance[:,idx]) for idx in range(hand_verts.shape[1])], dim=1)
        
        cmap = plt.cm.get_cmap('plasma')
        pose = pose.detach().cpu().numpy()
        obj_cmap_affordance = obj_cmap_affordance.detach().cpu().numpy()
        hand_cmap_affordance = hand_cmap_affordance.detach().cpu().numpy()
        obj_verts = obj_verts.detach().cpu().numpy()
        hand_verts = hand_verts.detach().cpu().numpy()

        for idx, img in enumerate(tqdm(orig_img)):
            img = np.array(img)
            
            obj_v_color = (cmap(obj_cmap_affordance[idx])[:,0,:-1] * 255).astype(np.int64)
            hand_v_color = [(cmap(hand_cmap_affordance[idx, h])[:,0,:-1] * 255).astype(np.int64) for h in range(hand_info.shape[1])]

            obj_mesh = trimesh.Trimesh(vertices=obj_verts[idx], vertex_colors=obj_v_color, faces = GT_obj_mesh.faces)
            hand_mesh = [trimesh.Trimesh(vertices=hand_verts[idx,i], faces=(mano_layer.th_faces).detach().cpu().numpy(), vertex_colors=hand_v_color[i]) 
                        for i, mano_layer in enumerate(model.MANO_LAYER)]

            if cfg.dataset == 'H2O':
                img = visualize(img, pose[idx, 0].astype(np.int32), 'left')
                img = visualize(img, pose[idx, 1].astype(np.int32), 'right')
                img = visualize_obj(img, pose[idx, 2].astype(np.int32))
                img = put_text_on_image(img, idx2action[action_class.item()])
                
                if not os.path.exists('./results/H2O/2Dpose'):
                    os.makedirs('./results/H2O/2Dpose')
                if not os.path.exists('./results/H2O/3Dpose'):
                    os.makedirs('./results/H2O/3Dpose')
                    
                cv2.imwrite(f'./results/H2O/2Dpose/{idx}.png', img[...,::-1])
                trimesh.exchange.export.export_mesh(hand_mesh[0],f'./results/H2O/3Dpose/{idx}_left.obj')
                trimesh.exchange.export.export_mesh(hand_mesh[1],f'./results/H2O/3Dpose/{idx}_right.obj')
                trimesh.exchange.export.export_mesh(obj_mesh,f'./results/H2O/3Dpose/{idx}_obj.obj')

            else :
                img = visualize(img, pose[idx, 0].astype(np.int32), 'right')
                img = visualize_obj(img, pose[idx, 1].astype(np.int32))
                img = put_text_on_image(img, idx2action[action_class.item()])
                
                if not os.path.exists('./results/FPHA/2Dpose'):
                    os.makedirs('./results/FPHA/2Dpose')
                if not os.path.exists('./results/FPHA/3Dpose'):
                    os.makedirs('./results/FPHA/3Dpose')
                    
                cv2.imwrite(f'./results/FPHA/2Dpose/{idx}.png', img[...,::-1])
                trimesh.exchange.export.export_mesh(hand_mesh[0],f'./results/FPHA/3Dpose/{idx}_right.obj')
                trimesh.exchange.export.export_mesh(obj_mesh,f'./results/FPHA/3Dpose/{idx}_obj.obj')