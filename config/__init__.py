from .defaults import _C as cfg

from data.transforms import build_transforms

def model_kwargs(cfg):
    model_config = {
        'input_channels': [int(cfg.MODEL.IN_DIM)] + int(cfg.MODEL.NB_LAYERS)*[int(cfg.MODEL.INPUT_CHANNELS)],
        'conv_channels': cfg.MODEL.CONV_CHANNELS,
        'dropout': cfg.TRAIN.DROPOUT, 
        'conv_activation': None,
        'nb_conv_layers': cfg.MODEL.NB_CONV_LAYERS,
        'visual_dim': cfg.DATA.VISUAL_DIM, 
        'conv_type': cfg.MODEL.CONV_TYPE
    }
    if cfg.MODEL.CONV_ACTIVATION:
        model_config['conv_activation'] = cfg.MODEL.CONV_ACTIVATION
        print('Conv Activation:', cfg.MODEL.CONV_ACTIVATION)
    return model_config

def data_kwargs(cfg):
    data_config =  {
        'root': cfg.DATA.ROOT,
        'k': cfg.DATA.TOP_K,
        'nodes_num': cfg.DATA.NODES_NUM,
        'transform': None,
        'pre_transform': None
    }
    
    try:
        data_config['transform'] = build_transforms(cfg.DATA.TRANSFORM)
    except:
        print('Transform: None !')
    else:
        print('Transform:', cfg.DATA.TRANSFORM)
    
    try:
        data_config['pre_transform'] = build_transforms(cfg.DATA.PRE_TRANSFORM)
    except:
        print('pre_Transform: None !')
    else:
        print('pre_Transform:', cfg.DATA.PRE_TRANSFORM)

    return data_config