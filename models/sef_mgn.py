import torch
from torch import nn

from .layers import CMCosConv, CMEdgeConv


class SEF_MGN_9(torch.nn.Module):

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

        print('\n SEF_MGN_9 ... \n')
        self.input_mlp = nn.Sequential(
            nn.Linear(9, 32), 
            nn.BatchNorm1d(32),
            nn.PReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(32,32),
            # nn.BatchNorm1d(32),
            # nn.PReLU(),
        )
        self.node_projection = nn.Linear(32, 1)

        self.gcn_0 = nn.ModuleList()
        self.gcn_b1  = nn.ModuleList()
        self.gcn_b2  = nn.ModuleList()
        self.gcn_b3  = nn.ModuleList()
        self.gcn_b21 = nn.ModuleList()
        self.gcn_b22 = nn.ModuleList()
        self.gcn_b31 = nn.ModuleList()
        self.gcn_b32 = nn.ModuleList()
        self.gcn_b33 = nn.ModuleList()
        
        self.gcn_0.append(
            convClass(32, 32, visual_dim), 
        )
        self.gcn_b1.append(
            convClass(32, 32, visual_dim // 8), 
        )
        self.gcn_b2.append(
            convClass(32, 32, visual_dim // 8), 
        )
        self.gcn_b3.append(
            convClass(32, 32, visual_dim // 8), 
        )
        self.gcn_b21.append(
            convClass(32, 32, visual_dim // 8), 
        )
        self.gcn_b22.append(
            convClass(32, 32, visual_dim // 8), 
        )
        self.gcn_b31.append(
            convClass(32, 32, visual_dim // 8), 
        )
        self.gcn_b32.append(
            convClass(32, 32, visual_dim // 8), 
        )
        self.gcn_b33.append(
            convClass(32, 32, visual_dim // 8), 
        )
        
        self.conv_dropout = nn.Dropout(p=dropout)

        self.gcn_projection_0 = nn.Linear(32, 1)
        self.gcn_projection_b1  = nn.Linear(32, 1)
        self.gcn_projection_b2  = nn.Linear(32, 1)
        self.gcn_projection_b3  = nn.Linear(32, 1)
        self.gcn_projection_b21 = nn.Linear(32, 1)
        self.gcn_projection_b22 = nn.Linear(32, 1)
        self.gcn_projection_b31 = nn.Linear(32, 1)
        self.gcn_projection_b32 = nn.Linear(32, 1)
        self.gcn_projection_b33 = nn.Linear(32, 1)
        

    def forward(self, batch_graphs):
        out = self.input_mlp(batch_graphs[0].x)
        for i in range(len(batch_graphs)):
            batch_graphs[i].x = out

        nodes_scores = self.node_projection(out).squeeze()
        
        batch_graphs[0].x = self.gcn_0[0](batch_graphs[0])
        batch_graphs[1].x = self.gcn_b1[0](batch_graphs[1])
        batch_graphs[2].x = self.gcn_b2[0](batch_graphs[2])
        batch_graphs[3].x = self.gcn_b3[0](batch_graphs[3])
        batch_graphs[4].x = self.gcn_b21[0](batch_graphs[4])
        batch_graphs[5].x = self.gcn_b22[0](batch_graphs[5])
        batch_graphs[6].x = self.gcn_b31[0](batch_graphs[6])
        batch_graphs[7].x = self.gcn_b32[0](batch_graphs[7])
        batch_graphs[8].x = self.gcn_b33[0](batch_graphs[8])

        gcn_scores_0 = self.gcn_projection_0(batch_graphs[0].x).squeeze()
        
        gcn_scores_b1  = self.gcn_projection_b1(batch_graphs[1].x).squeeze()
        gcn_scores_b2  = self.gcn_projection_b2(batch_graphs[2].x).squeeze()
        gcn_scores_b3  = self.gcn_projection_b3(batch_graphs[3].x).squeeze()
        gcn_scores_b21 = self.gcn_projection_b21(batch_graphs[4].x).squeeze()
        gcn_scores_b22 = self.gcn_projection_b22(batch_graphs[5].x).squeeze()
        gcn_scores_b31 = self.gcn_projection_b31(batch_graphs[6].x).squeeze()
        gcn_scores_b32 = self.gcn_projection_b32(batch_graphs[7].x).squeeze()
        gcn_scores_b33 = self.gcn_projection_b33(batch_graphs[8].x).squeeze()
        gcn_scores_1 = (gcn_scores_b1 + gcn_scores_b2 + gcn_scores_b3 + gcn_scores_b21 + gcn_scores_b22 + gcn_scores_b31 + gcn_scores_b32 + gcn_scores_b33) / 8

        return nodes_scores + gcn_scores_0 + gcn_scores_1


class SEF_MGN_2(torch.nn.Module):

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

        print('\n SEF_MGN_2 ... \n')
        self.input_mlp = nn.Sequential(
            nn.Linear(9, 32), 
            nn.BatchNorm1d(32),
            nn.PReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(32,32),
            # nn.BatchNorm1d(32),
            # nn.PReLU(),
        )
        self.node_projection = nn.Linear(32, 1)

        self.gcn_0 = nn.ModuleList()
        self.gcn_b21 = nn.ModuleList()
        
        self.gcn_0.append(
            convClass(32, 32, visual_dim), 
        )
        self.gcn_b21.append(
            convClass(32, 32, visual_dim // 8), 
        )
        
        self.conv_dropout = nn.Dropout(p=dropout)

        self.gcn_projection_0 = nn.Linear(32, 1)
        self.gcn_projection_b21 = nn.Linear(32, 1)
        

    def forward(self, batch_graphs):
        out = self.input_mlp(batch_graphs[0].x)
        batch_graphs[0].x = out
        batch_graphs[4].x = out

        nodes_scores = self.node_projection(out).squeeze()
        
        batch_graphs[0].x = self.gcn_0[0](batch_graphs[0])
        batch_graphs[4].x = self.gcn_b21[0](batch_graphs[4])

        gcn_scores_0 = self.gcn_projection_0(batch_graphs[0].x).squeeze()
        
        gcn_scores_b21 = self.gcn_projection_b21(batch_graphs[4].x).squeeze()
        
        gcn_scores_1 = gcn_scores_b21

        return nodes_scores + gcn_scores_0 + gcn_scores_1
