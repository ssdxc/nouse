import numpy as np
import os.path as osp
from tqdm import tqdm

import torch
from torch_geometric.nn import knn_graph
from torch_geometric.data import Data, Dataset

class GraphDataset(Dataset):
    """
    data: (imgidxs, matches, scores, feats_0)
    """
    
    def __init__(self, root, nodes_num=300, k=100, transform=None, pre_transform=None):
        self.root = root
        self.k = k
        self.nodes_num = nodes_num
        imglist_path = osp.join(self.root, 'imglist.txt')
        self.imgnames = self.load_txt(imglist_path)
        super(GraphDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        self._raw_file_names = ['%s_n%d.npz'%(imgname.split('.')[0],self.nodes_num) for imgname in self.imgnames]
        return self._raw_file_names

    @property
    def processed_file_names(self):
        self._processed_file_names = ['%s_n%d_k%d.pt'%(imgname.split('.')[0],self.nodes_num,self.k) for imgname in self.imgnames]
        return self._processed_file_names

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        for i, imgname in enumerate(tqdm(self.imgnames)):
            npz_data = np.load(osp.join(self.root, 'raw', self._raw_file_names[i]))
            candi_imgidxs = torch.from_numpy(npz_data['imgidxs']).float()
            candi_matches = torch.from_numpy(npz_data['matches']).float()
            candi_scores = torch.from_numpy(npz_data['scores']).float()
            # print(candi_scores.shape)
            candi_feats = torch.from_numpy(npz_data['feats']).float()
            graph = Data(
                x=candi_scores,
                edge_index=knn_graph(candi_feats, self.k, loop=True, flow="target_to_source"),
                y=candi_matches
            )
            graph.imgidxs = candi_imgidxs
            graph.feats = candi_feats
            graph.qid = torch.tensor([i])

            if self.pre_transform is not None:
                graph = self.pre_transform(graph)

            torch.save(graph, osp.join(self.processed_dir, '%s_n%d_k%d.pt'%(imgname.split('.')[0],self.nodes_num,self.k)))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, self._processed_file_names[idx]))
        return data

    def load_txt(self, txt_path):
        imgnames = []
        with open(txt_path, 'r') as f:
            for line in f:
                item = line.strip()
                imgnames.append(item)
        return imgnames

