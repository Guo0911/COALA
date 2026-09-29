import torch

from torch import nn

from .utils import MLP, Attention

def get_adapter(
    name:str,
    input_dim:int,
    output_dim:int,
    num_heads:int = 4,
    num_depths:int = 2
):
    if name == "conv-branchformer":
        return ConvBranchformerAudioEncoder(input_dim=input_dim, output_dim=output_dim, num_heads=num_heads, depth=num_depths)
    else:
        raise NotImplementedError

class ConvBranchformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()

        self.size = dim
        self.norm1 = norm_layer(dim)

        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(input_dim=dim, hidden_dim=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.conv1d = nn.Conv1d(dim, dim*2, kernel_size=1)
        self.glu_act = nn.GLU()
        self.cnn_norm1 = norm_layer(dim)
        self.cnn_norm2 = norm_layer(dim)

        # depth-wise conv
        self.dep_cnn1 = nn.Conv1d(dim, dim, kernel_size=3, groups=dim, padding=1)
        self.dep_cnn2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.conv1d_2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.cnn_drop = nn.Dropout(0.2)
        self.dropout = nn.Dropout(0.2)

        # attention-based pooling for two branches
        self.pooling_proj1 = torch.nn.Linear(dim, 1)
        self.pooling_proj2 = torch.nn.Linear(dim, 1)

        # linear projections for calculating merging weights
        self.weight_proj1 = torch.nn.Linear(dim, 1)
        self.weight_proj2 = torch.nn.Linear(dim, 1)

        # linear projection after weighted average
        self.merge_proj = torch.nn.Linear(dim, dim)

        self.mlp2 = MLP(input_dim=dim, hidden_dim=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # self atten block
        x_att = x + self.drop_path(self.attn(self.norm1(x)))
        x_att = x_att + self.drop_path(self.mlp(self.norm2(x_att)))

        # conv block
        x_cnn = self.conv1d(self.cnn_norm1(x).transpose(1,2)).transpose(1,2) #x_cnn [B, 50, dim*2]
        x_cnn = self.glu_act(x_cnn) #x_cnn [B, 50, dim]

        # depth-wise conv
        x_cnn = self.dep_cnn1(x_cnn.transpose(1,2))
        x_cnn = self.dep_cnn2(x_cnn).transpose(1,2)
        x_cnn = torch.nn.functional.relu(self.cnn_norm2(x_cnn))
        x_cnn = self.cnn_drop(self.conv1d_2(x_cnn.transpose(1,2)).transpose(1,2))
        x_cnn = x + x_cnn 

        # branch1 for atten_out attention pooling
        score1 = (
            self.pooling_proj1(x_att).transpose(1, 2) / self.size**0.5
        ) # (batch, 1, time)
        score1 = torch.softmax(score1, dim=-1)
        pooled1 = torch.matmul(score1, x_att).squeeze(1) # (batch, size)
        weight1 = self.weight_proj1(pooled1) # (batch, 1)

        # branch2 for cnn_out attention pooling
        score2 = (
            self.pooling_proj2(x_cnn).transpose(1, 2) / self.size**0.5
        ) # (batch, 1, time)
        score2 = torch.softmax(score2, dim=-1)
        pooled2 = torch.matmul(score2, x_cnn).squeeze(1) # (batch, size)
        weight2 = self.weight_proj2(pooled2) # (batch, 1)

        # normalize weights of two branches
        merge_weights = torch.softmax(
            torch.cat([weight1, weight2], dim=-1), dim=-1
        ) # (batch, 2)
        merge_weights = merge_weights.unsqueeze(-1).unsqueeze(-1) # (batch, 2, 1, 1)
        w1, w2 = merge_weights[:, 0], merge_weights[:, 1] # (batch, 1, 1)

        # merge and proj
        x = x + self.dropout(
            self.mlp2(w1 * x_att + w2 * x_cnn)
        )
        
        return x

class ConvBranchformerAudioEncoder(nn.Module):
    def __init__(self, input_dim=84, output_dim=24, num_heads=3, depth=3):
        super().__init__()

        self.proj = MLP(
            input_dim=input_dim,
            hidden_dim=output_dim,
            output_dim=output_dim,
            act_layer=nn.GELU,
            drop=0.1
        )
        self.encoder = nn.ModuleList([
            ConvBranchformerBlock(dim=output_dim, num_heads=num_heads) for i in range(depth)
        ])

    def forward(self, x):
        x = self.proj(x)
        for block in self.encoder:
            x = block(x)

        return x
