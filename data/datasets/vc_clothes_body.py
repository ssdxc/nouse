import os
import os.path as osp
import numpy as np
import json

from tqdm import tqdm
from scipy.spatial import distance
from sklearn import preprocessing
from collections import Counter
from utils import compute_distance_matrix

from ..build_graph import GraphDataset
from utils import mkdir_if_missing

import sys 

class VC_Clothes_Body(object):

    def __init__(self, root='', nodes_num=300, k=100, transform=None, pre_transform=None):
        self.root = root
        self.k = k
        self.nodes_num = nodes_num
        self.transform = transform
        self.pre_transform = pre_transform
        self.reid_train_dir = osp.join(root, 'reid_data', 'train')
        self.reid_query_dir = osp.join(root, 'reid_data', 'query')
        self.reid_gallery_dir = osp.join(root, 'reid_data', 'gallery')
        self.graph_train_dir = osp.join(root, 'graph_data', 'train')
        self.graph_test_dir = osp.join(root, 'graph_data', 'test')

        self.init_evaluation_info()

        self.graph_train = self.process_dir(self.graph_train_dir, self.reid_train_dir, self.reid_train_dir, is_train=True)
        self.graph_test = self.process_dir(self.graph_test_dir, self.reid_query_dir, self.reid_gallery_dir)

    def check_before_run(self, required_files):
        for fpath in required_files:
            if not osp.exists(fpath):
                raise RuntimeError('"{}" is not found'.format(fpath))

    def process_dir(self, graph_dir, query_dir, gallery_dir, is_train=False):
        graph_raw_dir = osp.join(graph_dir, 'raw')
        imglist_txt_path = osp.join(graph_dir, 'imglist.txt')
        if osp.exists(graph_raw_dir) and osp.exists(imglist_txt_path):
            graph_data = GraphDataset(graph_dir, self.nodes_num, self.k, self.transform, self.pre_transform)
        else:
            print('processing reid data ... ')
            # query
            q_imgnames = np.load(osp.join(query_dir, 'imgnames.npy'))
            q_pids = np.load(osp.join(query_dir, 'pids.npy'))
            q_camids = np.load(osp.join(query_dir, 'camids.npy'))
            q_feats = np.load(osp.join(query_dir, 'feats.npy'))
            # gallery
            g_imgnames = np.load(osp.join(gallery_dir, 'imgnames.npy'))
            g_pids = np.load(osp.join(gallery_dir, 'pids.npy'))
            g_camids = np.load(osp.join(gallery_dir, 'camids.npy'))
            g_feats = np.load(osp.join(gallery_dir, 'feats.npy'))

            mkdir_if_missing(graph_raw_dir)
            lines = []
            # with open(imglist_txt_path, 'w') as f:
            #     lines = ["%s 1 \n"%s for s in q_imgnames]
            #     f.write(''.join(lines))

            print("computing distance ... ")
            q_feats, g_feats = q_feats.cuda(), g_feats.cuda()
            distmat = compute_distance_matrix(q_feats, g_feats, metric='euclidean').cpu().numpy()
            # distmat = distance.cdist(q_feats, g_feats, 'euclidean').astype(np.float32)
            indices = np.argsort(distmat, axis=1)
            simmat = 1 - distmat

            match_sum = []
            for i, imgname in enumerate(tqdm(q_imgnames)):
                _idxs = indices[i][:self.nodes_num]
                _imgnames = g_imgnames[_idxs]
                _sims = simmat[i][_idxs]
                _q_pid = q_pids[i]
                _g_pids = g_pids[_idxs]
                _q_feat = q_feats[i]
                _g_feats = g_feats[_idxs]
                _matches = np.zeros((self.nodes_num,))
                _matches[_q_pid==_g_pids] = 1
                assert _matches.sum() >= 0 and _matches.sum() <= self.nodes_num, (self.nodes_num, _matches.sum())
                match_sum.append(_matches.sum())
                if is_train:
                    if _matches.sum() == self.nodes_num or _matches.sum() == 0:
                        print("skip", imgname, "... matches_sum nodes_num: \n", _matches.sum(), self.nodes_num)
                        continue
                lines.append('%s 1 \n'%imgname)
                np.savez(
                    osp.join(graph_dir, 'raw', '%s_n%d.npz'%(imgname.split('.')[0],self.nodes_num)),
                    imgidxs = _idxs,
                    matches = _matches,
                    scores = _sims.reshape(-1,1),
                    feats = _g_feats
                )
            
            with open(imglist_txt_path, 'w') as f:
                f.write(''.join(lines))
            match_sum = np.array(match_sum)
            print('statistic: ', Counter(match_sum))
            print('mean:', match_sum.mean(), ', min:', match_sum.min(), ', max:', match_sum.max())
            graph_data = GraphDataset(graph_dir, self.nodes_num, self.k, self.transform, self.pre_transform)

        return graph_data

    def init_evaluation_info(self):
        self.q_pids = np.load(osp.join(self.reid_query_dir, 'pids.npy'))
        self.q_camids = np.load(osp.join(self.reid_query_dir, 'camids.npy'))
        self.g_pids = np.load(osp.join(self.reid_gallery_dir, 'pids.npy'))
        self.g_camids = np.load(osp.join(self.reid_gallery_dir, 'camids.npy'))
