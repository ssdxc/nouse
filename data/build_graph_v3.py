import os
import numpy as np
import os.path as osp
from tqdm import tqdm

import torch
from torch_geometric.nn import knn_graph
from torch_geometric.data import Data, Dataset

os.environ['OMP_NUM_THREADS'] = '8'

class GraphDataset_v3(Dataset):
    """
    data: (imgidxs, matches, scores, feats_0)
    """
    
    def __init__(self, root, nodes_num=300, k=100, transform=None, pre_transform=None):
        self.root = root
        self.k = k
        self.nodes_num = nodes_num
        imglist_path = osp.join(self.root, 'imglist.txt')
        self.imgnames = self.load_txt(imglist_path)
        super(GraphDataset_v3, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        self._raw_file_names = ['%s_n%d.npz'%(imgname.split('.')[0],self.nodes_num) for imgname in self.imgnames]
        return self._raw_file_names

    @property
    def processed_file_names(self):
        # check for existance
        self._processed_file_names_body = ['%s_n%d_k%d_body.pt'%(imgname.split('.')[0],self.nodes_num,self.k) for imgname in self.imgnames]
        self._processed_file_names_face = ['%s_n%d_k%d_face.pt'%(imgname.split('.')[0],self.nodes_num,self.k) for imgname in self.imgnames]
        return self._processed_file_names_body + self._processed_file_names_face

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
            candi_feats_body = torch.from_numpy(npz_data['feats_body']).float()
            candi_feats_face = torch.from_numpy(npz_data['feats_face']).float()

            graph_body = Data(
                x=candi_scores,
                edge_index=knn_graph(candi_feats_body, self.k, loop=True, flow="target_to_source"),
                y=candi_matches
            )
            graph_face = Data(
                x=candi_scores,
                edge_index=knn_graph(candi_feats_face, self.k, loop=True, flow="target_to_source"),
                y=candi_matches
            )
            graph_body.imgidxs = candi_imgidxs
            graph_body.feats = candi_feats_body
            graph_body.qid = torch.tensor([i])
            graph_face.imgidxs = candi_imgidxs
            graph_face.feats = candi_feats_face
            graph_face.qid = torch.tensor([i])

            if self.pre_transform is not None:
                graph_body = self.pre_transform(graph_body)
                graph_face = self.pre_transform(graph_face)

            torch.save(graph_body, osp.join(self.processed_dir, '%s_n%d_k%d_body.pt'%(imgname.split('.')[0],self.nodes_num,self.k)))
            torch.save(graph_face, osp.join(self.processed_dir, '%s_n%d_k%d_face.pt'%(imgname.split('.')[0],self.nodes_num,self.k)))


    def len(self):
        return int(len(self.processed_file_names) / 2)

    def get(self, idx):
        data_body = torch.load(osp.join(self.processed_dir, self._processed_file_names_body[idx]))
        data_face = torch.load(osp.join(self.processed_dir, self._processed_file_names_face[idx]))
        return data_body, data_face

    def load_txt(self, txt_path):
        imgnames = []
        with open(txt_path, 'r') as f:
            for line in f:
                item = line.strip()
                imgnames.append(item)
        return imgnames

