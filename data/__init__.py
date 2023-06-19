from __future__ import print_function, absolute_import

from .datasets import VC_Clothes_Body
from .datasets import VC_Clothes
from .datasets import VC_Clothes_v3
from .datasets import VC_Clothes_v3_v1
from .datasets import Campus4K
from .datasets import Campus4K_v3
from .datasets import Market_v3
from .datasets import DukeMTMC_v3
from .datasets import MSMT17_v3
from .datasets import Campus4K_IMGREID_V3
from .datasets import DukeMTMC_v3_mgn

__datasets = {
    'vc_clothes_body': VC_Clothes_Body,
    'vc_clothes': VC_Clothes,
    'vc_clothes_v3': VC_Clothes_v3,
    'vc_clothes_v3_v1': VC_Clothes_v3_v1,
    'campus4k': Campus4K,
    'campus4k_v3': Campus4K_v3,
    'market_v3': Market_v3,
    'dukemtmc_v3': DukeMTMC_v3,
    'msmt17_v3': MSMT17_v3,
    'campus4k_imgreid': Campus4K_IMGREID_V3,
    'campus4k_imgreid_v3': Campus4K_IMGREID_V3,
    'dukemtmc_v3_mgn': DukeMTMC_v3_mgn,
}

def init_dataset(name, **kwargs):
    avai_datasets = list(__datasets.keys())
    if name not in avai_datasets:
        raise ValueError(
            'Invalid dataset name. Received "{}", '
            'but expected to be one of {}'.format(name, avai_datasets)
        )
    return __datasets[name](**kwargs)