import math

import torch
from torch import nn
from torch.nn import functional as F

from .layers import CMCosConv, CMEdgeConv


class encoder_mlp(nn.Module):
    def __init__(self, dim=32, dropout=0.3):
        super(encoder_mlp, self).__init__()
        self.input_mlp = nn.Sequential(
            nn.Linear(2, dim), 
            nn.BatchNorm1d(dim),
            nn.PReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return self.input_mlp(x)


class encoder_self_attn(nn.Module):
    def __init__(self, hidden_dim=512, out_dim=32, dropout=0.3):
        super(encoder_self_attn, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_q = nn.Linear(2, hidden_dim, bias=False)
        self.input_k = nn.Linear(2, hidden_dim, bias=False)
        self.input_v = nn.Linear(2, hidden_dim, bias=False)
        self.input_dropout = nn.Dropout(p=dropout)
        self.input_mlp = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, out_dim),
            # nn.BatchNorm1d(32),
            # nn.PReLU(),
            # nn.Dropout(p=dropout),
        )

    def forward(self, x, batch_size):
        q = self.input_q(x).view(batch_size, -1, self.hidden_dim)
        k = self.input_k(x).view(batch_size, -1, self.hidden_dim)
        v = self.input_v(x).view(batch_size, -1, self.hidden_dim)

        attn = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.hidden_dim)
        attn = self.input_dropout(F.softmax(attn, dim=-1))
        # attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        # print(5, out.size())
        out = out.view(-1, out.size(-1))
        out = self.input_mlp(out)
        return out


class encoder_multi_head(nn.Module):
    def __init__(self, head=8, hidden_dim=512, out_dim=32, dropout=0.3):
        super(encoder_multi_head, self).__init__()
        self.hidden_dim = hidden_dim
        self.d_k = hidden_dim // head
        self.head = head
        self.input_q = nn.Linear(2, hidden_dim, bias=False)
        self.input_k = nn.Linear(2, hidden_dim, bias=False)
        self.input_v = nn.Linear(2, hidden_dim, bias=False)
        self.input_dropout = nn.Dropout(p=dropout)
        self.input_mlp = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, out_dim),
            # nn.BatchNorm1d(32),
            # nn.PReLU(),
        )

    def forward(self, x, batch_size):
        # print(x.size())
        q = self.input_q(x).view(batch_size, -1, self.head, self.d_k).transpose(1,2)
        k = self.input_k(x).view(batch_size, -1, self.head, self.d_k).transpose(1,2)
        v = self.input_v(x).view(batch_size, -1, self.head, self.d_k).transpose(1,2)
        # print(q.size())

        attn = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.d_k)
        attn = self.input_dropout(F.softmax(attn, dim=-1))
        # attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        # print(5, out.size())
        out = out.transpose(1,2).contiguous().view(-1, self.hidden_dim)
        out = self.input_mlp(out)
        return out

    


class gcn_self_attn(nn.Module):
    def __init__(self,
                 input_channels=2,
                 conv_channels=2,
                 nb_conv_layers=1,
                 dropout=0.0,
                 visual_dim=2048,
                 conv_type="cos",
                 conv_activation=None):
        super().__init__()
        if conv_type == "cos":
            convClass = CMCosConv
        elif conv_type == "edge":
            convClass = lambda in_channels, out_channels, visual_dim: CMEdgeConv(in_channels=in_channels,
                                                                     out_channels=out_channels,
                                                                     visual_dim=visual_dim)
        else:
            raise ValueError("invalid CM convolutions argument")

        print('single head self attn.')
        self.self_attn = encoder_self_attn(dropout=0.3)
        # print('multi head self attn.')
        # self.self_attn = encoder_multi_head(head=8, dropout=0.3)

        self.node_projection = nn.Linear(32, 1)

        self.gcn_body = nn.ModuleList()
        self.gcn_face = nn.ModuleList()
        self.gcn_body.append(
            convClass(32, 32, 2048), 
        )
        self.gcn_face.append(
            convClass(32, 32, 512), 
        )
        # self.conv_dropout = nn.Dropout(p=dropout)
        self.conv_activation = lambda x: x
        self.gcn_projection_body = nn.Linear(32, 1)
        self.gcn_projection_face = nn.Linear(32, 1)

    def forward(self, batch_graph_body, batch_graph_face, batch_vec):
        if batch_vec is None:
            sz_b = 1
        else:
            sz_b = batch_vec[-1] + 1

        out = self.self_attn(batch_graph_body.x, sz_b)

        batch_graph_body.x = out
        batch_graph_face.x = out

        nodes_scores = self.node_projection(out).squeeze()

        batch_graph_body.x = self.gcn_body[0](batch_graph_body)
        batch_graph_face.x = self.gcn_face[0](batch_graph_face)

        gcn_scores_body = self.gcn_projection_body(batch_graph_body.x).squeeze()
        gcn_scores_face = self.gcn_projection_face(batch_graph_face.x).squeeze()

        return nodes_scores + gcn_scores_body + gcn_scores_face




class V3_GCN_SELF_ATTN(torch.nn.Module):

    def __init__(self,
                 input_channels=2,
                 conv_channels=2,
                 nb_conv_layers=1,
                 dropout=0.0,
                 visual_dim=2048,
                 conv_type="cos",
                 conv_activation=None):
        super().__init__()
        if conv_type == "cos":
            convClass = CMCosConv
        elif conv_type == "edge":
            convClass = lambda in_channels, out_channels, visual_dim: CMEdgeConv(in_channels=in_channels,
                                                                     out_channels=out_channels,
                                                                     visual_dim=visual_dim)
        else:
            raise ValueError("invalid CM convolutions argument")

        self.d_k = 512
        self.d_v = 512
        self.input_q = nn.Linear(2, self.d_k, bias=False)
        self.input_k = nn.Linear(2, self.d_k, bias=False)
        self.input_v = nn.Linear(2, self.d_v, bias=False)
        self.input_dropout = nn.Dropout(p=0.3)
        # self.input_linear = nn.Linear(512, 32)
        # self.layer_norm = nn.LayerNorm(32)

        self.input_mlp = nn.Sequential(
            # nn.BatchNorm1d(512),
            nn.LayerNorm(512),
            nn.PReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512,32),
            # nn.BatchNorm1d(32),
            # nn.PReLU(),
        )

        self.node_projection = nn.Linear(32, 1)

        self.gcn_body = nn.ModuleList()
        self.gcn_face = nn.ModuleList()
        self.gcn_body.append(
            convClass(32, 32, 2048), 
        )
        self.gcn_face.append(
            convClass(32, 32, 512), 
        )
        # self.conv_dropout = nn.Dropout(p=dropout)
        self.conv_activation = lambda x: x
        self.gcn_projection_body = nn.Linear(32, 1)
        self.gcn_projection_face = nn.Linear(32, 1)

    def forward(self, batch_graph_body, batch_graph_face, batch_vec):
        # print(1, batch_graph_body.x.size(), batch_graph_face.x.size())
        # print(batch_graph_body.batch.size())
        # print(batch_graph_body.batch)
        # sz_b = batch_graph_body.batch[-1] + 1
        # sz_b_face = batch_graph_face.batch[-1] + 1
        # assert sz_b == sz_b_face, (sz_b, sz_b_face)
        # print(2, sz_b, sz_b_face)

        if batch_vec is None:
            sz_b = 1
        else:
            sz_b = batch_vec[-1] + 1

        q = self.input_q(batch_graph_body.x).view(sz_b, -1, self.d_k)
        k = self.input_k(batch_graph_body.x).view(sz_b, -1, self.d_k)
        v = self.input_v(batch_graph_body.x).view(sz_b, -1, self.d_v)
        # print(3, q.size(), k.size(), v.size())

        # q = self.input_q(batch_graph_body.x)
        # k = self.input_k(batch_graph_body.x)
        # v = self.input_v(batch_graph_body.x)

        attn = torch.matmul(q, k.transpose(1,2)) / math.sqrt(self.d_k)
        # attn = torch.matmul(q, k.transpose(0,1)) / math.sqrt(self.d_k)
        # attn = torch.matmul(q, k.transpose(1,2)) 
        attn = self.input_dropout(F.softmax(attn, dim=-1))
        # print(4, attn.size())
        out = torch.matmul(attn, v)
        # print(5, out.size())
        out = out.view(-1, out.size(-1))
        # print(6, out.size())
        # out = self.input_linear(out)
        # out = self.layer_norm(out)

        out = self.input_mlp(out)

        batch_graph_body.x = out
        batch_graph_face.x = out

        nodes_scores = self.node_projection(out).squeeze()

        batch_graph_body.x = self.gcn_body[0](batch_graph_body)
        batch_graph_face.x = self.gcn_face[0](batch_graph_face)

        gcn_scores_body = self.gcn_projection_body(batch_graph_body.x).squeeze()
        gcn_scores_face = self.gcn_projection_face(batch_graph_face.x).squeeze()

        return nodes_scores + gcn_scores_body + gcn_scores_face



class V3_GCN_SELF_ATTN2(torch.nn.Module):

    def __init__(self,
                 input_channels=2,
                 conv_channels=2,
                 nb_conv_layers=1,
                 dropout=0.0,
                 visual_dim=2048,
                 conv_type="cos",
                 conv_activation=None):
        super().__init__()
        if conv_type == "cos":
            convClass = CMCosConv
        elif conv_type == "edge":
            convClass = lambda in_channels, out_channels, visual_dim: CMEdgeConv(in_channels=in_channels,
                                                                     out_channels=out_channels,
                                                                     visual_dim=visual_dim)
        else:
            raise ValueError("invalid CM convolutions argument")

        self.d_k = 32
        self.d_v = 32
        self.input_q = nn.Linear(2, self.d_k, bias=False)
        self.input_k = nn.Linear(2, self.d_k, bias=False)
        self.input_v = nn.Linear(2, self.d_v, bias=False)
        self.input_dropout = nn.Dropout(p=0.3)
        self.input_linear = nn.Linear(32, 32)
        # self.layer_norm = nn.LayerNorm(32)

        # self.input_mlp = nn.Sequential(
        #     nn.BatchNorm1d(32),
        #     nn.PReLU(),
        #     nn.Dropout(p=dropout),
        #     nn.Linear(32,32),
        #     # nn.BatchNorm1d(32),
        #     # nn.PReLU(),
        # )

        self.node_projection = nn.Linear(32, 1)

        self.gcn_body = nn.ModuleList()
        self.gcn_face = nn.ModuleList()
        self.gcn_body.append(
            convClass(32, 32, 2048), 
        )
        self.gcn_face.append(
            convClass(32, 32, 512), 
        )
        # self.conv_dropout = nn.Dropout(p=dropout)
        self.conv_activation = lambda x: x
        self.gcn_projection_body = nn.Linear(32, 1)
        self.gcn_projection_face = nn.Linear(32, 1)

        self.output_q = nn.Linear(2, self.d_k, bias=False)
        self.output_k = nn.Linear(2, self.d_k, bias=False)
        self.output_v = nn.Linear(2, self.d_v, bias=False)

    def forward(self, batch_graph_body, batch_graph_face, batch_vec):
        # print(1, batch_graph_body.x.size(), batch_graph_face.x.size())
        # print(batch_graph_body.batch.size())
        # print(batch_graph_body.batch)
        # sz_b = batch_graph_body.batch[-1] + 1
        # sz_b_face = batch_graph_face.batch[-1] + 1
        # assert sz_b == sz_b_face, (sz_b, sz_b_face)
        # print(2, sz_b, sz_b_face)

        if batch_vec is None:
            sz_b = 1
        else:
            sz_b = batch_vec[-1] + 1

        q = self.input_q(batch_graph_body.x).view(sz_b, -1, self.d_k)
        k = self.input_k(batch_graph_body.x).view(sz_b, -1, self.d_k)
        v = self.input_v(batch_graph_body.x).view(sz_b, -1, self.d_v)
        # print(3, q.size(), k.size(), v.size())

        # q = self.input_q(batch_graph_body.x)
        # k = self.input_k(batch_graph_body.x)
        # v = self.input_v(batch_graph_body.x)

        attn = torch.matmul(q, k.transpose(1,2)) / math.sqrt(self.d_k)
        # attn = torch.matmul(q, k.transpose(0,1)) / math.sqrt(self.d_k)
        # attn = torch.matmul(q, k.transpose(1,2)) 
        attn = self.input_dropout(F.softmax(attn, dim=-1))
        # print(4, attn.size())
        out = torch.matmul(attn, v)
        # print(5, out.size())
        out = out.view(-1, out.size(-1))
        # print(6, out.size())
        # out = self.input_linear(out)
        # out = self.layer_norm(out)

        # out = self.input_mlp(out)

        batch_graph_body.x = out
        batch_graph_face.x = out

        nodes_scores = self.node_projection(out).squeeze()

        batch_graph_body.x = self.gcn_body[0](batch_graph_body)
        batch_graph_face.x = self.gcn_face[0](batch_graph_face)

        gcn_scores_body = self.gcn_projection_body(batch_graph_body.x).squeeze()
        gcn_scores_face = self.gcn_projection_face(batch_graph_face.x).squeeze()

        return nodes_scores + gcn_scores_body + gcn_scores_face

