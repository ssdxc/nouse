import torch
from torch import nn

from .layers import CMCosConv, CMEdgeConv


class V3_GCN(torch.nn.Module):

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

        self.input_mlp = nn.Sequential(
            nn.Linear(2, 32), 
            # nn.BatchNorm1d(32),
            # nn.ReLU(inplace=True),
            nn.PReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(32,32)
        )
        self.node_projection = nn.Linear(32, 1)

        self.gcn_body = nn.ModuleList()
        self.gcn_face = nn.ModuleList()
        self.gcn_body.append(
            convClass(32, 32, visual_dim), 
        )
        self.gcn_face.append(
            convClass(32, 32, 512), 
        )
        self.conv_dropout = nn.Dropout(p=dropout)
        self.conv_activation = lambda x: x
        self.gcn_projection_body = nn.Linear(32, 1)
        self.gcn_projection_face = nn.Linear(32, 1)

    def forward(self, batch_graph_body, batch_graph_face):
        out = self.input_mlp(batch_graph_body.x)
        batch_graph_body.x = out
        batch_graph_face.x = out

        nodes_scores = self.node_projection(out).squeeze()

        batch_graph_body.x = self.gcn_body[0](batch_graph_body)
        batch_graph_face.x = self.gcn_face[0](batch_graph_face)

        gcn_scores_body = self.gcn_projection_body(batch_graph_body.x).squeeze()
        gcn_scores_face = self.gcn_projection_face(batch_graph_face.x).squeeze()

        return nodes_scores + gcn_scores_body + gcn_scores_face
        # return nodes_scores
        # return nodes_scores + gcn_scores_body
        # return gcn_scores_body + gcn_scores_face


class V3_GCN2(torch.nn.Module):

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

        print('\ngcn2 ... \n')
        self.input_mlp = nn.Sequential(
            nn.Linear(2, 32), 
            nn.BatchNorm1d(32),
            nn.PReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(32,32),
            # nn.BatchNorm1d(32),
            # nn.PReLU(),
        )
        self.node_projection = nn.Linear(32, 1)

        self.gcn_body = nn.ModuleList()
        self.gcn_face = nn.ModuleList()
        self.gcn_body.append(
            convClass(32, 32, visual_dim), 
        )
        self.gcn_face.append(
            convClass(32, 32, 512), 
        )
        self.conv_dropout = nn.Dropout(p=dropout)
        self.gcn_projection_body = nn.Linear(32, 1)
        self.gcn_projection_face = nn.Linear(32, 1)

    def forward(self, batch_graph_body, batch_graph_face):
        out = self.input_mlp(batch_graph_body.x)
        batch_graph_body.x = out
        batch_graph_face.x = out

        nodes_scores = self.node_projection(out).squeeze()

        batch_graph_body.x = self.gcn_body[0](batch_graph_body)
        batch_graph_face.x = self.gcn_face[0](batch_graph_face)

        gcn_scores_body = self.gcn_projection_body(batch_graph_body.x).squeeze()
        gcn_scores_face = self.gcn_projection_face(batch_graph_face.x).squeeze()

        return nodes_scores + gcn_scores_body + gcn_scores_face
        # return 1*nodes_scores + 1.5*gcn_scores_body + 1*gcn_scores_face
        # return nodes_scores
        # return nodes_scores + gcn_scores_body
        # return nodes_scores + gcn_scores_face
        # return gcn_scores_body + gcn_scores_face


class V3_GCN_MLP(torch.nn.Module):

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

        print('\n linear ... \n')
        self.input_mlp = nn.Sequential(
            nn.Linear(2, 32), 
            nn.BatchNorm1d(32),
            nn.PReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(32,32),
            # nn.BatchNorm1d(32),
            # nn.PReLU(),
        )
        self.node_projection = nn.Linear(32, 1)

    def forward(self, batch_graph_body, batch_graph_face):
        out = self.input_mlp(batch_graph_body.x)
        nodes_scores = self.node_projection(out).squeeze()
        return nodes_scores


class V3_GCN_BODY_D(torch.nn.Module):

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

        print('\n linear + body ... \n')
        self.input_mlp = nn.Sequential(
            nn.Linear(2, 32), 
            nn.BatchNorm1d(32),
            nn.PReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(32,32),
            # nn.BatchNorm1d(32),
            # nn.PReLU(),
        )
        self.node_projection = nn.Linear(32, 1)

        self.gcn_body = nn.ModuleList()
        self.gcn_body.append(
            convClass(32, 32, visual_dim), 
        )
        self.conv_dropout = nn.Dropout(p=dropout)
        self.conv_activation = lambda x: x
        self.gcn_projection_body = nn.Linear(32, 1)

    def forward(self, batch_graph_body, batch_graph_face):
        out = self.input_mlp(batch_graph_body.x)
        batch_graph_body.x = out

        nodes_scores = self.node_projection(out).squeeze()

        batch_graph_body.x = self.gcn_body[0](batch_graph_body)

        gcn_scores_body = self.gcn_projection_body(batch_graph_body.x).squeeze()

        return nodes_scores + gcn_scores_body


class V3_GCN_BODY_S(torch.nn.Module):

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

        self.input_mlp = nn.Sequential(
            nn.Linear(2, 32), 
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(32,32)
        )

        self.gcn_body = nn.ModuleList()
        self.gcn_body.append(
            convClass(32, 32, 2048), 
        )
        self.conv_dropout = nn.Dropout(p=dropout)
        self.conv_activation = lambda x: x
        self.gcn_projection_body = nn.Linear(32, 1)

    def forward(self, batch_graph_body, batch_graph_face):
        out = self.input_mlp(batch_graph_body.x)
        batch_graph_body.x = out

        batch_graph_body.x = self.gcn_body[0](batch_graph_body)

        gcn_scores_body = self.gcn_projection_body(batch_graph_body.x).squeeze()

        return gcn_scores_body


class V3_GCN_FACE_D(torch.nn.Module):

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

        self.input_mlp = nn.Sequential(
            nn.Linear(2, 32), 
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(32,32)
        )
        self.node_projection = nn.Linear(32, 1)

        self.gcn_face = nn.ModuleList()
        self.gcn_face.append(
            convClass(32, 32, 512), 
        )
        self.conv_dropout = nn.Dropout(p=dropout)
        self.conv_activation = lambda x: x
        self.gcn_projection_face = nn.Linear(32, 1)

    def forward(self, batch_graph_body, batch_graph_face):
        out = self.input_mlp(batch_graph_face.x)
        batch_graph_face.x = out

        nodes_scores = self.node_projection(out).squeeze()

        batch_graph_face.x = self.gcn_face[0](batch_graph_face)

        gcn_scores_face = self.gcn_projection_face(batch_graph_face.x).squeeze()

        return nodes_scores + gcn_scores_face


class V3_GCN_FACE_S(torch.nn.Module):

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

        self.input_mlp = nn.Sequential(
            nn.Linear(2, 32), 
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(32,32)
        )

        self.gcn_face = nn.ModuleList()
        self.gcn_face.append(
            convClass(32, 32, 512), 
        )
        self.conv_dropout = nn.Dropout(p=dropout)
        self.conv_activation = lambda x: x
        self.gcn_projection_face = nn.Linear(32, 1)

    def forward(self, batch_graph_body, batch_graph_face):
        out = self.input_mlp(batch_graph_face.x)
        batch_graph_face.x = out

        batch_graph_face.x = self.gcn_face[0](batch_graph_face)

        gcn_scores_face = self.gcn_projection_face(batch_graph_face.x).squeeze()

        return gcn_scores_face


class V3_GCN_BODY_FACE(torch.nn.Module):

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

        self.input_mlp = nn.Sequential(
            nn.Linear(2, 32), 
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(32,32)
        )

        self.gcn_body = nn.ModuleList()
        self.gcn_face = nn.ModuleList()
        self.gcn_body.append(
            convClass(32, 32, 2048), 
        )
        self.gcn_face.append(
            convClass(32, 32, 512), 
        )
        self.conv_dropout = nn.Dropout(p=dropout)
        self.conv_activation = lambda x: x
        self.gcn_projection_body = nn.Linear(32, 1)
        self.gcn_projection_face = nn.Linear(32, 1)

    def forward(self, batch_graph_body, batch_graph_face):
        out = self.input_mlp(batch_graph_body.x)
        batch_graph_body.x = out
        batch_graph_face.x = out

        batch_graph_body.x = self.gcn_body[0](batch_graph_body)
        batch_graph_face.x = self.gcn_face[0](batch_graph_face)

        gcn_scores_body = self.gcn_projection_body(batch_graph_body.x).squeeze()
        gcn_scores_face = self.gcn_projection_face(batch_graph_face.x).squeeze()

        return gcn_scores_body + gcn_scores_face