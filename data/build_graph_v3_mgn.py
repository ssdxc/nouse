import os
import numpy as np
import os.path as osp
from tqdm import tqdm

import torch
from torch_geometric.nn import knn_graph
from torch_geometric.data import Data, Dataset

os.environ['OMP_NUM_THREADS'] = '8'

class GraphDataset_v3_mgn(Dataset):
    """
    data: (imgidxs, matches, scores, feats_0)
    """
    
    def __init__(self, root, nodes_num=300, k=100, transform=None, pre_transform=None):
        self.root = root
        self.k = k
        self.nodes_num = nodes_num
        imglist_path = osp.join(self.root, 'imglist.txt')
        self.imgnames = self.load_txt(imglist_path)
        super(GraphDataset_v3_mgn, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        self._raw_file_names = ['%s_n%d.npz'%(imgname.split('.')[0],self.nodes_num) for imgname in self.imgnames]
        return self._raw_file_names

    @property
    def processed_file_names(self):
        # check for existance
        self._processed_file_names = ['%s_n%d_k%d_body.pt'%(imgname.split('.')[0],self.nodes_num,self.k) for imgname in self.imgnames]
        self._processed_file_names_b1  = ['%s_n%d_k%d_body_b1.pt'%(imgname.split('.')[0],self.nodes_num,self.k) for imgname in self.imgnames]
        self._processed_file_names_b2  = ['%s_n%d_k%d_body_b2.pt'%(imgname.split('.')[0],self.nodes_num,self.k) for imgname in self.imgnames]
        self._processed_file_names_b3  = ['%s_n%d_k%d_body_b3.pt'%(imgname.split('.')[0],self.nodes_num,self.k) for imgname in self.imgnames]
        self._processed_file_names_b21 = ['%s_n%d_k%d_body_b21.pt'%(imgname.split('.')[0],self.nodes_num,self.k) for imgname in self.imgnames]
        self._processed_file_names_b22 = ['%s_n%d_k%d_body_b22.pt'%(imgname.split('.')[0],self.nodes_num,self.k) for imgname in self.imgnames]
        self._processed_file_names_b31 = ['%s_n%d_k%d_body_b31.pt'%(imgname.split('.')[0],self.nodes_num,self.k) for imgname in self.imgnames]
        self._processed_file_names_b32 = ['%s_n%d_k%d_body_b32.pt'%(imgname.split('.')[0],self.nodes_num,self.k) for imgname in self.imgnames]
        self._processed_file_names_b33 = ['%s_n%d_k%d_body_b33.pt'%(imgname.split('.')[0],self.nodes_num,self.k) for imgname in self.imgnames]

        return self._processed_file_names + self._processed_file_names_b1 + self._processed_file_names_b2 + self._processed_file_names_b3 + self._processed_file_names_b21 + self._processed_file_names_b22 + self._processed_file_names_b31 + self._processed_file_names_b32 + self._processed_file_names_b33

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
            candi_feats_b1  = torch.from_numpy(npz_data['feats_b1']).float()
            candi_feats_b2  = torch.from_numpy(npz_data['feats_b2']).float()
            candi_feats_b3  = torch.from_numpy(npz_data['feats_b3']).float()
            candi_feats_b21 = torch.from_numpy(npz_data['feats_b21']).float()
            candi_feats_b22 = torch.from_numpy(npz_data['feats_b22']).float()
            candi_feats_b31 = torch.from_numpy(npz_data['feats_b31']).float()
            candi_feats_b32 = torch.from_numpy(npz_data['feats_b32']).float()
            candi_feats_b33 = torch.from_numpy(npz_data['feats_b33']).float()
            
            graph = Data(
                x=candi_scores,
                edge_index=knn_graph(candi_feats, self.k, loop=True, flow="target_to_source"),
                y=candi_matches
            )
            graph_b1 = Data(
                x=candi_scores,
                edge_index=knn_graph(candi_feats_b1, self.k, loop=True, flow="target_to_source"),
                y=candi_matches
            )
            graph_b2 = Data(
                x=candi_scores,
                edge_index=knn_graph(candi_feats_b2, self.k, loop=True, flow="target_to_source"),
                y=candi_matches
            )
            graph_b3 = Data(
                x=candi_scores,
                edge_index=knn_graph(candi_feats_b3, self.k, loop=True, flow="target_to_source"),
                y=candi_matches
            )
            graph_b21 = Data(
                x=candi_scores,
                edge_index=knn_graph(candi_feats_b21, self.k, loop=True, flow="target_to_source"),
                y=candi_matches
            )
            graph_b22 = Data(
                x=candi_scores,
                edge_index=knn_graph(candi_feats_b22, self.k, loop=True, flow="target_to_source"),
                y=candi_matches
            )
            graph_b31 = Data(
                x=candi_scores,
                edge_index=knn_graph(candi_feats_b31, self.k, loop=True, flow="target_to_source"),
                y=candi_matches
            )
            graph_b32 = Data(
                x=candi_scores,
                edge_index=knn_graph(candi_feats_b32, self.k, loop=True, flow="target_to_source"),
                y=candi_matches
            )
            graph_b33 = Data(
                x=candi_scores,
                edge_index=knn_graph(candi_feats_b33, self.k, loop=True, flow="target_to_source"),
                y=candi_matches
            )
            

            graph.imgidxs, graph.feats, graph.qid = candi_imgidxs, candi_feats, torch.tensor([i])
            graph_b1.imgidxs,  graph_b1.feats,  graph_b1.qid  = candi_imgidxs, candi_feats_b1, torch.tensor([i])
            graph_b2.imgidxs,  graph_b2.feats,  graph_b2.qid  = candi_imgidxs, candi_feats_b2, torch.tensor([i])
            graph_b3.imgidxs,  graph_b3.feats,  graph_b3.qid  = candi_imgidxs, candi_feats_b3, torch.tensor([i])
            graph_b21.imgidxs, graph_b21.feats, graph_b21.qid = candi_imgidxs, candi_feats_b21, torch.tensor([i])
            graph_b22.imgidxs, graph_b22.feats, graph_b22.qid = candi_imgidxs, candi_feats_b22, torch.tensor([i])
            graph_b31.imgidxs, graph_b31.feats, graph_b31.qid = candi_imgidxs, candi_feats_b31, torch.tensor([i])
            graph_b32.imgidxs, graph_b32.feats, graph_b32.qid = candi_imgidxs, candi_feats_b32, torch.tensor([i])
            graph_b33.imgidxs, graph_b33.feats, graph_b33.qid = candi_imgidxs, candi_feats_b33, torch.tensor([i])

            if self.pre_transform is not None:
                graph = self.pre_transform(graph)
                graph_b1  = self.pre_transform(graph_b1)
                graph_b2  = self.pre_transform(graph_b2)
                graph_b3  = self.pre_transform(graph_b3)
                graph_b21 = self.pre_transform(graph_b21)
                graph_b22 = self.pre_transform(graph_b22)
                graph_b31 = self.pre_transform(graph_b31)
                graph_b32 = self.pre_transform(graph_b32)
                graph_b33 = self.pre_transform(graph_b33)

            torch.save(graph, osp.join(self.processed_dir, '%s_n%d_k%d_body.pt'%(imgname.split('.')[0],self.nodes_num,self.k)))
            torch.save(graph_b1 , osp.join(self.processed_dir, '%s_n%d_k%d_body_b1.pt'%(imgname.split('.')[0],self.nodes_num,self.k)))
            torch.save(graph_b2 , osp.join(self.processed_dir, '%s_n%d_k%d_body_b2.pt'%(imgname.split('.')[0],self.nodes_num,self.k)))
            torch.save(graph_b3 , osp.join(self.processed_dir, '%s_n%d_k%d_body_b3.pt'%(imgname.split('.')[0],self.nodes_num,self.k)))
            torch.save(graph_b21, osp.join(self.processed_dir, '%s_n%d_k%d_body_b21.pt'%(imgname.split('.')[0],self.nodes_num,self.k)))
            torch.save(graph_b22, osp.join(self.processed_dir, '%s_n%d_k%d_body_b22.pt'%(imgname.split('.')[0],self.nodes_num,self.k)))
            torch.save(graph_b31, osp.join(self.processed_dir, '%s_n%d_k%d_body_b31.pt'%(imgname.split('.')[0],self.nodes_num,self.k)))
            torch.save(graph_b32, osp.join(self.processed_dir, '%s_n%d_k%d_body_b32.pt'%(imgname.split('.')[0],self.nodes_num,self.k)))
            torch.save(graph_b33, osp.join(self.processed_dir, '%s_n%d_k%d_body_b33.pt'%(imgname.split('.')[0],self.nodes_num,self.k)))
            


    def len(self):
        return int(len(self.processed_file_names) / 9)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, self._processed_file_names[idx]))
        data_b1  = torch.load(osp.join(self.processed_dir, self._processed_file_names_b1[idx]))
        data_b2  = torch.load(osp.join(self.processed_dir, self._processed_file_names_b2[idx]))
        data_b3  = torch.load(osp.join(self.processed_dir, self._processed_file_names_b3[idx]))
        data_b21 = torch.load(osp.join(self.processed_dir, self._processed_file_names_b21[idx]))
        data_b22 = torch.load(osp.join(self.processed_dir, self._processed_file_names_b22[idx]))
        data_b31 = torch.load(osp.join(self.processed_dir, self._processed_file_names_b31[idx]))
        data_b32 = torch.load(osp.join(self.processed_dir, self._processed_file_names_b32[idx]))
        data_b33 = torch.load(osp.join(self.processed_dir, self._processed_file_names_b33[idx]))
        
        return [data, data_b1, data_b2, data_b3, data_b21, data_b22, data_b31, data_b32, data_b33]

    def load_txt(self, txt_path):
        imgnames = []
        with open(txt_path, 'r') as f:
            for line in f:
                item = line.strip()
                imgnames.append(item)
        return imgnames

