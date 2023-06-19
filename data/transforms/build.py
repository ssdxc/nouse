from torch_geometric.transforms import Compose

from .transforms import NormalizeNodes

__transforms = {
    'NormalizeNodes': NormalizeNodes()
}

def build_transforms(graphTransforms):
    if len(graphTransforms) == 0:
        return None

    avai_transforms = list(__transforms.keys())
    for name in graphTransforms:
        if name not in avai_transforms:
            raise ValueError(
                'Invalid transform name. Received "{}", '
                'but expected to be one of {}'.format(name, avai_transforms)
            )

    if len(graphTransforms) == 1:
        transform =  __transforms[graphTransforms[0]]
    else:
        raise ValueError
        
    return transform