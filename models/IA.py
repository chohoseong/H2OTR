import torch
from torch import nn
from einops import repeat
from models.module import Attention, PreNorm, FeedForward

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class InteractionRecognizer(nn.Module):
    def __init__(self, num_classes=37, num_frames=64, dim = 308, depth = 4, heads = 3, pool = 'cls', dim_head = 64, dropout = 0.2,
                 scale_dim = 4, sample_points = 2000, dataset=None):

        super().__init__()
        
        self.linear = nn.Linear(sample_points, 100)

        self.left_linear = nn.Linear(778*(3+1), 100)
        self.right_linear = nn.Linear(778*(3+1), 100)
        self.obj_linear = nn.Linear(sample_points*(3+1), 100)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames+1, dim))

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        self.num_frames = num_frames
        self.dataset = dataset
    
    def forward(self, x, y, label):
        if self.dataset == 'H2O':
            x_1 = self.left_linear(x[:,0].reshape(self.num_frames, -1))
            x_2 = self.right_linear(x[:,1].reshape(self.num_frames, -1))
            y = self.obj_linear(y.reshape(self.num_frames, -1))
            x = torch.cat([x_1,x_2,y,label],dim=-1)
        else:
            x = self.right_linear(x.reshape(self.num_frames, -1))
            y = self.obj_linear(y.reshape(self.num_frames, -1))
            x = torch.cat([x,y,label], dim=-1)

        x = x.unsqueeze(0)
        t = x.shape[1]

        cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=1)
        x = torch.cat((cls_temporal_tokens, x), dim=1)
        x += self.pos_embedding

        x = self.temporal_transformer(x)
        
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        return self.mlp_head(x)
    
def build_interaction_recognizer(args, cfg):
    
    model = InteractionRecognizer(
            num_classes=cfg.num_action_classes, 
            num_frames=cfg.num_frames, 
            dim = cfg.IR_dim, 
            depth = args.IR_enc_layers, 
            heads = args.IR_nheads, 
            pool = args.pool, 
            dim_head = args.IR_dim_head, 
            dropout = args.IR_dropout,
            scale_dim = args.scale_dim, 
            dataset = args.dataset_file)

    return model