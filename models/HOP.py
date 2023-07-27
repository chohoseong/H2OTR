# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

import torch
import torch.nn.functional as F
from torch import nn
import math

from util.misc import (NestedTensor, nested_tensor_from_tensor_list,inverse_sigmoid)

from .backbone import build_backbone
from .deformable_transformer import build_deforamble_transformer
import copy


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class HOP(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels, 
                 aux_loss=True, with_box_refine=False, cfg=None):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries. 
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.hidden_dim = transformer.d_model
        self.cls_embed = nn.Linear(self.hidden_dim, num_classes)
        self.keypoint_embed = MLP(self.hidden_dim, self.hidden_dim, 63, 3) 
        self.obj_keypoint_embed = MLP(self.hidden_dim, self.hidden_dim, 63, 3) 
        self.num_feature_levels = num_feature_levels
        self.cfg = cfg
        
        self.query_embed = nn.Embedding(num_queries, self.hidden_dim*2) 
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):   
                input_proj_list.append(nn.Sequential(                 
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, self.hidden_dim),
                ))
                in_channels = self.hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.cls_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.keypoint_embed.layers[-1].weight.data, 0)      
        nn.init.constant_(self.keypoint_embed.layers[-1].bias.data, 0)        
        nn.init.constant_(self.obj_keypoint_embed.layers[-1].weight.data, 0)  
        nn.init.constant_(self.obj_keypoint_embed.layers[-1].bias.data, 0)    
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
    
        num_pred = transformer.decoder.num_layers
        if with_box_refine:
            self.cls_embed = _get_clones(self.cls_embed, num_pred)
            self.keypoint_embed = _get_clones(self.keypoint_embed, num_pred)         
            self.obj_keypoint_embed = _get_clones(self.obj_keypoint_embed, num_pred) 

            self.transformer.decoder.cls_embed = self.cls_embed
            self.transformer.decoder.keypoint_embed = self.keypoint_embed
            self.transformer.decoder.obj_keypoint_embed = self.obj_keypoint_embed
        else:
            self.cls_embed = nn.ModuleList([self.cls_embed for _ in range(num_pred)])
            self.keypoint_embed = nn.ModuleList([self.keypoint_embed for _ in range(num_pred)])
            self.obj_keypoint_embed = nn.ModuleList([self.obj_keypoint_embed for _ in range(num_pred)])

            self.transformer.decoder.keypoint_embed = None
            self.transformer.decoder.obj_keypoint_embed = None
            
            
    def forward(self, samples: NestedTensor):
        ### backbone ###
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)  
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose() 
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors) # 
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)
        
        query_embeds = self.query_embed.weight 
        hs, init_reference, inter_references = self.transformer(srcs, masks, pos, query_embeds)
        # hs : result include intermeditate feature (num_decoder_layer, B, num_queries, hidden_dim)
        
        dataset = 'H2O' if len(self.cfg.hand_idx) == 2 else 'FPHA'
        outputs_classes = []
        outputs_keypoints = [] 
        outputs_obj_keypoints = [] 
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
                reference = inverse_sigmoid(reference)
            else:
                reference = inter_references[lvl - 1]
                if dataset == 'H2O':
                    reference = inverse_sigmoid(reference)
                else:
                    reference = inverse_sigmoid((reference+ 0.5)/2)

            outputs_class = self.cls_embed[lvl](hs[lvl])
            key = self.keypoint_embed[lvl](hs[lvl]) 
            obj_key = self.obj_keypoint_embed[lvl](hs[lvl]) 
                
            if reference.shape[-1] == 42:
                ref_x = reference[...,0::2].mean(-1).unsqueeze(-1)
                ref_y = reference[...,1::2].mean(-1).unsqueeze(-1)

                key = key.reshape(key.shape[0], key.shape[1], 21, 3)
                key[..., :2] += torch.cat([ref_x, ref_y], dim=-1)[:,:,None,:] 
                key = key.reshape(key.shape[0], key.shape[1], -1)

                obj_key = obj_key.reshape(obj_key.shape[0], obj_key.shape[1], 21, 3)
                obj_key[..., :2] += torch.cat([ref_x, ref_y], dim=-1)[:,:,None,:] 
                obj_key = obj_key.reshape(obj_key.shape[0], obj_key.shape[1], -1)

            else:
                assert reference.shape[-1] == 2
                key = key.reshape(key.shape[0], key.shape[1], 21, 3)
                key[..., :2] += reference[:,:,None,:] 
                key = key.reshape(key.shape[0], key.shape[1], -1)
                
                obj_key = obj_key.reshape(obj_key.shape[0], obj_key.shape[1], 21, 3)
                obj_key[..., :2] += reference[:,:,None,:] 
                obj_key = obj_key.reshape(obj_key.shape[0], obj_key.shape[1], -1)

            if dataset == 'H2O':
                outputs_keypoint = key.sigmoid() 
                outputs_obj_keypoint = obj_key.sigmoid() 
            else:
                outputs_keypoint = key.sigmoid()*2 - 0.5
                outputs_obj_keypoint = obj_key.sigmoid()*2 - 0.5
            outputs_classes.append(outputs_class)
            outputs_keypoints.append(outputs_keypoint) 
            outputs_obj_keypoints.append(outputs_obj_keypoint) 
        outputs_class = torch.stack(outputs_classes)
        outputs_keypoints = torch.stack(outputs_keypoints) 
        outputs_obj_keypoints = torch.stack(outputs_obj_keypoints) 

        out = {'pred_logits': outputs_class[-1], 'pred_keypoints': outputs_keypoints[-1], 'pred_obj_keypoints': outputs_obj_keypoints[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_keypoints, outputs_obj_keypoints)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_keypoints, outputs_obj_keypoints):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_keypoints': b,  'pred_obj_keypoints': c}
                for a, b, c, in zip(outputs_class[:-1], outputs_keypoints[:-1], outputs_obj_keypoints[:-1])]

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def build_HandObejctPoseEstimator(args, cfg):
    num_classes = cfg.num_obj_classes
    backbone = build_backbone(args)
    transformer = build_deforamble_transformer(args, cfg)
    
    model = HOP(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=True,
        with_box_refine=args.with_box_refine,
        cfg=cfg)

    return model
