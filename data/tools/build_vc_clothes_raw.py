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
    root = '/data4/xieqk/datasets/VC-Clothes'
    body_feats_root = '/data2/xieqk/Ti-Six/data/xieqk/code/xieqk/merge'
    train_feats_body = osp.join(body_feats_root, 't_feats_body.npz')
    query_feats_body = osp.join(body_feats_root, 'q_feats_body.npz')
    gallery_feats_body = osp.join(body_feats_root, 'g_feats_body.npz')
    train_dir_face = osp.join(root, 'face_det', 'train')
    query_dir_face = osp.join(root, 'face_det', 'query')
    gallery_dir_face = osp.join(root, 'face_det', 'gallery')

    save_root = '/data4/xieqk/code/gitee/megcn/datasets/vc_clothes/graph_data_norm/'
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

    # Torch
    q_body_torch = torch.from_numpy(q_feats_body).cuda()
    g_body_torch = torch.from_numpy(g_feats_body).cuda()
    q_face_torch = torch.from_numpy(q_feats_face).cuda()
    g_face_torch = torch.from_numpy(g_feats_face).cuda()
    print('Torch: computing distance body ... ', q_body_torch.size(), g_body_torch.size())
    distmat_body_torch = compute_distance_matrix(
        q_body_torch, g_body_torch, 'euclidean')
    print('Torch: computing distance face ... ', q_face_torch.size(), g_face_torch.size())
    distmat_face_torch = compute_distance_matrix(
        q_face_torch, g_face_torch, 'euclidean')
    distmat_body = distmat_body_torch.cpu().numpy()
    distmat_face = distmat_face_torch.cpu().numpy()
    print('distmat shape: body {}, face {}'.format(distmat_body.shape, distmat_face.shape))

    # # Numpy
    # print('Numpy: computing distance body ... ', q_feats_body.shape, g_feats_body.shape)
    # distmat_body = distance.cdist(q_feats_body, g_feats_body, 'euclidean')
    # print('Numpy: computing distance face ... ', q_feats_face.shape, g_feats_face.shape)
    # distmat_face = distance.cdist(q_feats_face, g_feats_face, 'euclidean')
    # print('distmat shape: body {}, face {}'.format(distmat_body.shape, distmat_face.shape))

    simmat_body = 1 - distmat_body
    simmat_face = 1 - distmat_face

    for i, q_flag in enumerate(q_face_flags):
        for j, g_flag in enumerate(g_face_flags):
            if q_flag == 0 or g_flag == 0:
                simmat_face[i, j] = 0

    indices = np.argsort(distmat_body, axis=1)

    node_num = 200
    imglist = []
    match_sum = []
    g_imgnames = np.array(g_imgnames)
    save_dir_raw = osp.join(save_dir, 'raw')
    mkdir_if_missing(save_dir_raw)
    for i, imgname in enumerate(tqdm(q_imgnames)):
        _idxs = indices[i][:node_num]
        _g_imgnames = g_imgnames[_idxs]
        _g_feats = g_feats_body[_idxs]
        _q_pid = q_pids[i]
        _g_pids = g_pids[_idxs]
        _matches = np.zeros((node_num,))
        _matches[_q_pid == _g_pids] = 1
        assert _matches.sum() >= 0 and _matches.sum() <= node_num, (node_num, _matches.sum())
        match_sum.append(_matches.sum())
        if is_train:
            if _matches.sum() == node_num or _matches.sum() == 0:
                print('skip {}, matches_sum: {}, node_num: {}'.format(imgname, _matches.sum(), node_num))
                continue
        _sims_body = simmat_body[i][_idxs]
        _sims_face = simmat_face[i][_idxs]
        _sims = np.vstack((_sims_body, _sims_face)).T
        imglist.append('{}\n'.format(imgname))
        np.savez(
            osp.join(save_dir_raw, '{}_n{}.npz'.format(imgname.split('.')[0], node_num)),
            imgidxs = _idxs, 
            matches = _matches,
            scores = _sims,
            feats = _g_feats
        )
    
    with open(osp.join(save_dir, 'imglist.txt'), 'w') as f:
        f.write(''.join(imglist))
    
    np.save(osp.join(save_dir, 'query_pids.npy'), q_pids)
    np.save(osp.join(save_dir, 'query_camids.npy'), q_camids)
    np.save(osp.join(save_dir, 'gallery_pids.npy'), g_pids)
    np.save(osp.join(save_dir, 'gallery_camids.npy'), g_camids)
    
    match_sum = np.array(match_sum)
    print('statistic: ', Counter(match_sum))
    print('mean: {}, min: {}, max: {}'.format(match_sum.mean(), match_sum.min(), match_sum.max()))


def get_face_feats(imgnames, dir_face):
    feats_face, face_flags = [], []
    for imgname in tqdm(imgnames):
        imgname_base = osp.splitext(imgname)[0]
        json_name = imgname_base + '.json'
        face_data_dict = read_json(osp.join(dir_face, json_name))

        feat_face, face_flag = feats_select(face_data_dict)

        feats_face.append(feat_face)
        face_flags.append(face_flag)
    return np.array(feats_face), np.array(face_flags)

def feats_select(face_dict):
    bboxs, face_feats = face_dict['bbox'], face_dict['feats']
    if len(bboxs) > 0:
        scores = np.array(bboxs)[:, -1]
        idx = np.argsort(scores)[-1]
        feat_face = face_feats[idx]
        return feat_face, 1
    feat_face = np.random.rand(512)
    feat_face = preprocessing.normalize([feat_face], 'l2').reshape((-1, ))
    # print(np.sum(feat_face*feat_face))
    return feat_face, 0

def feats_select_v2(face_dict):
    bboxs, face_feats = face_dict['bbox'], face_dict['feats']
    if len(bboxs) > 0:
        scores = np.array(bboxs)[:, -1]
        idx = np.argsort(scores)[-1]
        # print(bboxs, scores, idx)
        if scores[idx] > 0.98:
            feat_face = np.array(face_feats[idx])
            # print(np.sum(feat_face*feat_face))
            return feat_face, 1
    feat_face = np.random.rand(512)
    feat_face = preprocessing.normalize([feat_face], 'l2').reshape((-1, ))
    # print(np.sum(feat_face*feat_face))
    return feat_face, 0

if __name__ == '__main__':
    main()
