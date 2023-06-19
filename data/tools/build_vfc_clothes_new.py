import sys
import os
import os.path as osp
import numpy as np
from tqdm import tqdm
from sklearn import preprocessing
from scipy.spatial import distance
from collections import Counter

import torch

sys.path.append('.')
from utils.tools import mkdir_if_missing, read_json
from utils.distance import compute_distance_matrix


def main():
    root = '/data/xieqk/data/VC-Clothes'
    body_feats_root = '/data/xieqk/data/VC-Clothes/detections/merge'
    train_feats_body = osp.join(body_feats_root, 't_feats_body.npz')
    query_feats_body = osp.join(body_feats_root, 'q_feats_body.npz')
    gallery_feats_body = osp.join(body_feats_root, 'g_feats_body.npz')
    train_dir_face = osp.join(root, 'detections', 'retina_arcface_0.8', 'train')
    query_dir_face = osp.join(root, 'detections', 'retina_arcface_0.8', 'query')
    gallery_dir_face = osp.join(root, 'detections', 'retina_arcface_0.8', 'gallery')

    save_root = '/data/xieqk/data/VC-Clothes/exp/new_50_50_0.8/'
    train_save_dir = osp.join(save_root, 'train')
    test_save_dir = osp.join(save_root, 'test')

    make_graph_raw(train_save_dir, train_feats_body, train_dir_face, train_feats_body, train_dir_face, is_train=True)
    make_graph_raw(test_save_dir, query_feats_body, query_dir_face, gallery_feats_body, gallery_dir_face)


def make_graph_raw(save_dir, q_npz_path, q_face_dir, g_npz_path, g_face_dir, is_train=False):
    print('loading query from {} & {}'.format(q_npz_path, q_face_dir))
    q_npz_body = np.load(q_npz_path)
    q_imgnames = [osp.basename(x) for x in q_npz_body['paths']]
    q_feats_body = q_npz_body['feats']
    q_pids, q_camids = q_npz_body['pids'], q_npz_body['camids']
    q_feats_face, q_face_flags = get_face_feats(q_imgnames, q_face_dir)
    print('loading gallery from {} & {}'.format(g_npz_path, g_face_dir))
    g_npz_body = np.load(g_npz_path)
    g_imgnames = [osp.basename(x) for x in g_npz_body['paths']]
    g_feats_body = g_npz_body['feats']
    g_pids, g_camids = g_npz_body['pids'], g_npz_body['camids']
    g_feats_face, g_face_flags = get_face_feats(g_imgnames, g_face_dir)


def get_face_feats(imgnames, dir_face):
    feats_face, face_flags = [], []
    for imgname in tqdm(imgnames):
        imgname_base = osp.splitext(imgname)[0]
        json_name = imgname_base + '.json'
        json_path = osp.join(dir_face, json_name)
        if osp.exists(json_path):
            face_data_dict = read_json(json_path)
            feat_face = face_data_dict['feats'][0]
            face_flag = 1
        else:
            feat_face = np.random.random((512,))
            face_flag = 0

        feats_face.append(feat_face)
        face_flags.append(face_flag)
    return np.array(feats_face), np.array(face_flags)