from __future__ import absolute_import

from .v3_gcn import V3_GCN, V3_GCN_MLP, V3_GCN_BODY_D, V3_GCN2, V3_GCN_BODY_S, V3_GCN_FACE_D, V3_GCN_FACE_S, V3_GCN_BODY_FACE
from .v3_gcn_self_attn import V3_GCN_SELF_ATTN, gcn_self_attn
from .ablation import SEF, Linear, Body, Face, Linear_Body, Linear_Face, Body_Face, GCN1, GCN2, GCN3, GCN4, SEF_indep
from .sef_mgn import SEF_MGN_9, SEF_MGN_2

__model_factory = {
    'v3_gcn': V3_GCN,
    'v3_gcn2': V3_GCN2,
    'v3_gcn_mlp': V3_GCN_MLP,
    'v3_gcn_body_d': V3_GCN_BODY_D,
    'v3_gcn_body_s': V3_GCN_BODY_S,
    'v3_gcn_face_d': V3_GCN_FACE_D,
    'v3_gcn_face_s': V3_GCN_FACE_S,
    'v3_gcn_self_attn': V3_GCN_SELF_ATTN,
    'v3_gcn_body_face': V3_GCN_BODY_FACE,
    'gcn_self_attn': gcn_self_attn, 
    'sef': SEF,
    'linear': Linear,
    'body': Body,
    'face': Face,
    'linear_body': Linear_Body,
    'linear_face': Linear_Face,
    'body_face': Body_Face, 
    'gcn1': GCN1, 
    'gcn2': GCN2, 
    'gcn3': GCN3, 
    'gcn4': GCN4, 
    'sef_mgn_9': SEF_MGN_9,
    'sef_mgn_2': SEF_MGN_2,
    'sef_indep': SEF_indep
}

def show_avai_models():
    """Displays available models.

    Examples::
        >>> from torchreid import models
        >>> models.show_avai_models()
    """
    print(list(__model_factory.keys()))

def build_model(name, **kwargs):
    avai_models = list(__model_factory.keys())
    if name == 'gat':
        return __model_factory[name]()
    if name not in avai_models:
        raise KeyError(
            'Unknown model: {}. Must be one of {}'.format(name, avai_models)
        )
    return __model_factory[name](**kwargs)