import os
import argparse
from pathlib import Path
import numpy as np
import torch
from engine import demo
from models import build_model

from PIL import Image
import torchvision.transforms as T
import pickle
from cfg import Config

def get_args_parser():
    parser = argparse.ArgumentParser('H2OTR', add_help=False)

 
    parser.add_argument('--with_box_refine', default=True, action='store_true')

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    
    # InteractionRecognizer
    parser.add_argument('--IR_enc_layers', default=4, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--IR_nheads', default=3, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pool', default='cls', type=str, choices=('cls', 'mean'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--IR_dim_head', default=64, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--IR_dropout', default=0.2, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--scale_dim', default=4, type=int,
                        help="Dropout applied in the transformer")
    
    # dataset parameters
    parser.add_argument('--dataset_file', default='H2O', type=str, choices=('H2O', 'FPHA'))
    parser.add_argument('--data_path', default='', type=str)

    parser.add_argument('--output_dir', default='./results',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--pretrained_model', default='./weights/H2O.pth')
    parser.add_argument('--img_size', default=(540, 960), type=tuple)
    parser.add_argument('--vid_id', default=20, type=int)

    return parser

def get_image(path):
    return Image.open(path).convert('RGB')

def make_transforms(img_size):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return T.Compose([
            T.Resize(img_size),
            normalize,
        ])

def sampling(total_frame_num, sample_frame_num):
    idxs = np.arange(0, sample_frame_num) * total_frame_num/sample_frame_num
    if total_frame_num >= sample_frame_num:
        idxs = np.unique(np.trunc(idxs))
    else:
        idxs = np.trunc(idxs)
    return list(idxs.astype(np.int32))    


def load_h2o(data_path, transform, vid_id, cfg):
    with open(os.path.join(data_path, 'action_labels/action_test.txt')) as f:
        action_path_list = f.readlines()
    
    action_path_list = action_path_list[1:]
    _id, file_path, start_act, end_act, _, _ = action_path_list[vid_id].split(' ')
    
    vid_path = os.path.join(data_path, file_path)
    
    img_ids = np.arange(int(start_act), int(end_act)+1)
    img_ids = img_ids[sampling(len(img_ids), cfg.num_frames)]
    
    vid = []
    orig_img = []
    for i in img_ids:
        file_path = os.path.join(vid_path, 'cam4/rgb', f'{i}'.zfill(6) + '.png')
        img = get_image(file_path)
        orig_img.append(img)
        img = transform(img)
        vid.append(img)
    vid = torch.stack(vid)
    
    return vid, orig_img

def load_fpha(data_path, transform, vid_id, cfg):
    obj_trans_root = os.path.join(data_path, 'Object_6D_pose_annotation_v1')
    
    with open(os.path.join(data_path, 'data_split_action_recognition.txt'), 'r') as f:
        vid_info = f.read().split('\n')
        train_vid = [v.split(' ')[0] for v in vid_info[1:601]]
        test_vid = [v.split(' ')[0] for v in vid_info[602:-1]]
    
    action_path_list=[]
    for i, vid_path in enumerate(test_vid):
        if not os.path.exists(os.path.join(obj_trans_root, vid_path)):
            continue
        vid = os.path.join(data_path, 'Video_files', vid_path,  'color')
        action_path_list.append(vid)
    
    vid = []
    orig_img = []
    
    for file_name in sorted(os.listdir(action_path_list[vid_id])):
        file_path = os.path.join(action_path_list[vid_id], file_name)
        img = get_image(file_path)
        orig_img.append(img)
        img = transform(img)
        vid.append(img)
    vid = torch.stack(vid)
    
    return vid[sampling(len(vid), cfg.num_frames)], np.array(orig_img)[sampling(len(vid), cfg.num_frames)]

def main(args):
    device = torch.device(args.device)
    cfg = Config(args)

    data_path = os.path.join(args.data_path, args.dataset_file)

    ## load img ##
    if args.dataset_file == 'H2O':
        vid, orig_img = load_h2o(data_path, make_transforms(args.img_size), args.vid_id, cfg)
    else:
        vid, orig_img = load_fpha(data_path, make_transforms(args.img_size), args.vid_id, cfg)

    ## load object ##
    idx2obj = {v:k for k, v in cfg.obj2idx.items()}
    obj_vertices = {}
    obj_bbox = {}
    for i in range(1,cfg.hand_idx[0]):
        with open(os.path.join('config', args.dataset_file, f'{idx2obj[i]}_vertices.pkl'), 'rb') as f:
            vertices = pickle.load(f)
            obj_vertices[i] = torch.from_numpy(vertices).to(torch.float32).to(device)
        with open(os.path.join('config', args.dataset_file, f'{idx2obj[i]}_bbox.pkl'), 'rb') as f:
            bbox = pickle.load(f)
            obj_bbox[i] = torch.from_numpy(bbox).to(torch.float32).to(device)
            
    ## load model ##
    model = build_model(args, cfg)
    model.to(device)
    
    checkpoint = torch.load(args.pretrained_model, map_location='cpu')
    model.load_state_dict(checkpoint)
    
    ## demo ##
    demo(model, vid.to(device), cfg.cam_param, orig_img, obj_vertices, obj_bbox, cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('H2OTR demo script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
