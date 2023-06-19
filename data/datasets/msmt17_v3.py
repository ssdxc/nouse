import os
import os.path as osp
import numpy as np
import json

from tqdm import tqdm
from scipy.spatial import distance
from sklearn import preprocessing
from collections import Counter
from utils import compute_distance_matrix

from ..build_graph_v3 import GraphDataset_v3
from utils import mkdir_if_missing

import sys 

class MSMT17_v3(object):

    def __init__(self, root='', nodes_num=300, k=100, transform=None, pre_transform=None):
        self.root = root
        self.k = k
        self.nodes_num = nodes_num
        self.transform = transform
        self.pre_transform = pre_transform
        self.graph_train_dir = osp.join(root, 'train')
        self.graph_test_dir = osp.join(root, 'test')

        self.init_evaluation_info()

        self.graph_train = self.process_dir(self.graph_train_dir)
        self.graph_test = self.process_dir(self.graph_test_dir)

    def check_before_run(self, required_files):
        for fpath in required_files:
            if not osp.exists(fpath):
                raise RuntimeError('"{}" is not found'.format(fpath))

    def process_dir(self, graph_dir):
        graph_data = GraphDataset_v3(graph_dir, self.nodes_num, self.k, self.transform, self.pre_transform)
        return graph_data

    def init_evaluation_info(self):
        self.q_pids = np.load(osp.join(self.graph_test_dir, 'query_pids.npy'))
        self.q_camids = np.load(osp.join(self.graph_test_dir, 'query_camids.npy'))
        self.g_pids = np.load(osp.join(self.graph_test_dir, 'gallery_pids.npy'))
        self.g_camids = np.load(osp.join(self.graph_test_dir, 'gallery_camids.npy'))
