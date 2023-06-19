import torch
from torch import nn

from .layers import CMCosConv, CMEdgeConv


class SEF(torch.nn.Module):

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

        print('\n SEF ... \n')
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

class SEF_indep(torch.nn.Module):

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

        print('\n SEF ... \n')
        self.input_mlp_body = nn.Sequential(
            nn.Linear(2, 32), 
            nn.BatchNorm1d(32),
            nn.PReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(32,32),
            # nn.BatchNorm1d(32),
            # nn.PReLU(),
        )
        self.input_mlp_face = nn.Sequential(
            nn.Linear(2, 32), 
            nn.BatchNorm1d(32),
            nn.PReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(32,32),
            # nn.BatchNorm1d(32),
            # nn.PReLU(),
        )
        self.input_mlp_linear = nn.Sequential(
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
        out = self.input_mlp_linear(batch_graph_body.x)
        batch_graph_body.x = self.input_mlp_body(batch_graph_body.x)
        batch_graph_face.x = self.input_mlp_face(batch_graph_face.x)

        nodes_scores = self.node_projection(out).squeeze()

        batch_graph_body.x = self.gcn_body[0](batch_graph_body)
        batch_graph_face.x = self.gcn_face[0](batch_graph_face)

        gcn_scores_body = self.gcn_projection_body(batch_graph_body.x).squeeze()
        gcn_scores_face = self.gcn_projection_face(batch_graph_face.x).squeeze()

        return nodes_scores + gcn_scores_body + gcn_scores_face


class Linear(torch.nn.Module):

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

        print('\n Linear ... \n')
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


class Body(torch.nn.Module):

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

        print('\n Body ... \n')
        self.input_mlp = nn.Sequential(
            nn.Linear(2, 32), 
            nn.BatchNorm1d(32),
            nn.PReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(32,32),
            # nn.BatchNorm1d(32),
            # nn.PReLU(),
        )

        self.gcn_body = nn.ModuleList()
        self.gcn_body.append(
            convClass(32, 32, visual_dim), 
        )
        self.gcn_projection_body = nn.Linear(32, 1)

    def forward(self, batch_graph_body, batch_graph_face):
        out = self.input_mlp(batch_graph_body.x)
        batch_graph_body.x = out

        batch_graph_body.x = self.gcn_body[0](batch_graph_body)

        gcn_scores_body = self.gcn_projection_body(batch_graph_body.x).squeeze()

        return gcn_scores_body


class Face(torch.nn.Module):

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

        print('\n Face ... \n')
        self.input_mlp = nn.Sequential(
            nn.Linear(2, 32), 
            nn.BatchNorm1d(32),
            nn.PReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(32,32),
            # nn.BatchNorm1d(32),
            # nn.PReLU(),
        )

        self.gcn_face = nn.ModuleList()
        self.gcn_face.append(
            convClass(32, 32, 512), 
        )
        self.gcn_projection_face = nn.Linear(32, 1)

    def forward(self, batch_graph_body, batch_graph_face):
        out = self.input_mlp(batch_graph_face.x)
        batch_graph_face.x = out

        batch_graph_face.x = self.gcn_face[0](batch_graph_face)

        gcn_scores_face = self.gcn_projection_face(batch_graph_face.x).squeeze()

        return gcn_scores_face



class Linear_Body(torch.nn.Module):

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

        print('\n Linear Body ... \n')
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
        self.gcn_projection_body = nn.Linear(32, 1)

    def forward(self, batch_graph_body, batch_graph_face):
        out = self.input_mlp(batch_graph_body.x)
        batch_graph_body.x = out

        nodes_scores = self.node_projection(out).squeeze()

        batch_graph_body.x = self.gcn_body[0](batch_graph_body)

        gcn_scores_body = self.gcn_projection_body(batch_graph_body.x).squeeze()

        return nodes_scores + gcn_scores_body



class Linear_Face(torch.nn.Module):

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

        print('\n Linear Face ... \n')
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

        self.gcn_face = nn.ModuleList()
        self.gcn_face.append(
            convClass(32, 32, 512), 
        )

        self.gcn_projection_face = nn.Linear(32, 1)

    def forward(self, batch_graph_body, batch_graph_face):
        out = self.input_mlp(batch_graph_face.x)
        batch_graph_face.x = out

        nodes_scores = self.node_projection(out).squeeze()

        batch_graph_face.x = self.gcn_face[0](batch_graph_face)

        gcn_scores_face = self.gcn_projection_face(batch_graph_face.x).squeeze()

        return nodes_scores + gcn_scores_face


class Body_Face(torch.nn.Module):

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

        print('\n Body_Face ... \n')
        self.input_mlp = nn.Sequential(
            nn.Linear(2, 32), 
            nn.BatchNorm1d(32),
            nn.PReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(32,32),
            # nn.BatchNorm1d(32),
            # nn.PReLU(),
        )

        self.gcn_body = nn.ModuleList()
        self.gcn_face = nn.ModuleList()
        self.gcn_body.append(
            convClass(32, 32, visual_dim), 
        )
        self.gcn_face.append(
            convClass(32, 32, 512), 
        )

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


class GCN1(torch.nn.Module):

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

        print('\n GCN1 ... \n')
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


class GCN2(torch.nn.Module):

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

        print('\n GCN2 ... \n')
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
        self.gcn_body.append(
            convClass(32, 32, visual_dim),  
        )
        self.gcn_face.append(
            convClass(32, 32, 512), 
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

        batch_graph_body.x = self.gcn_body[1](batch_graph_body)
        batch_graph_face.x = self.gcn_face[1](batch_graph_face)

        gcn_scores_body = self.gcn_projection_body(batch_graph_body.x).squeeze()
        gcn_scores_face = self.gcn_projection_face(batch_graph_face.x).squeeze()

        return nodes_scores + gcn_scores_body + gcn_scores_face



class GCN3(torch.nn.Module):

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

        print('\n GCN3 ... \n')
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
        self.gcn_body.append(
            convClass(32, 32, visual_dim),  
        )
        self.gcn_body.append(
            convClass(32, 32, visual_dim),  
        )
        self.gcn_face.append(
            convClass(32, 32, 512), 
        )
        self.gcn_face.append(
            convClass(32, 32, 512), 
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

        batch_graph_body.x = self.gcn_body[1](batch_graph_body)
        batch_graph_face.x = self.gcn_face[1](batch_graph_face)

        batch_graph_body.x = self.gcn_body[2](batch_graph_body)
        batch_graph_face.x = self.gcn_face[2](batch_graph_face)

        gcn_scores_body = self.gcn_projection_body(batch_graph_body.x).squeeze()
        gcn_scores_face = self.gcn_projection_face(batch_graph_face.x).squeeze()

        return nodes_scores + gcn_scores_body + gcn_scores_face


class GCN4(torch.nn.Module):

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

        print('\n GCN4 ... \n')
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
        self.gcn_body.append(
            convClass(32, 32, visual_dim),  
        )
        self.gcn_body.append(
            convClass(32, 32, visual_dim),  
        )
        self.gcn_body.append(
            convClass(32, 32, visual_dim),  
        )
        self.gcn_face.append(
            convClass(32, 32, 512), 
        )
        self.gcn_face.append(
            convClass(32, 32, 512), 
        )
        self.gcn_face.append(
            convClass(32, 32, 512), 
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

        batch_graph_body.x = self.gcn_body[1](batch_graph_body)
        batch_graph_face.x = self.gcn_face[1](batch_graph_face)

        batch_graph_body.x = self.gcn_body[2](batch_graph_body)
        batch_graph_face.x = self.gcn_face[2](batch_graph_face)

        batch_graph_body.x = self.gcn_body[3](batch_graph_body)
        batch_graph_face.x = self.gcn_face[3](batch_graph_face)

        gcn_scores_body = self.gcn_projection_body(batch_graph_body.x).squeeze()
        gcn_scores_face = self.gcn_projection_face(batch_graph_face.x).squeeze()

        return nodes_scores + gcn_scores_body + gcn_scores_face