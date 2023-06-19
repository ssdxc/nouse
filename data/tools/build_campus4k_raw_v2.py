import sys
import os
import os.path as osp
import numpy as np
from tqdm import tqdm
from sklearn import preprocessing
from scipy.spatial import distance
from collections import Counter
from glob import glob

import torch

sys.path.append('.')
from utils.tools import mkdir_if_missing, read_json
from utils.distance import compute_distance_matrix


def main():
    root = '/data3/xieqk/code/gitee/megcn/datasets/campus4k/reid_data'
    train_feats_body = osp.join(root, 'body', 'resnet50_campus4k_softmax', 'train_result.npz')
    query_feats_body = osp.join(root, 'body', 'resnet50_campus4k_softmax', 'query_result.npz')
    gallery_feats_body = osp.join(root, 'body', 'resnet50_campus4k_softmax', 'gallery_result.npz')
    face_root = '/data3/xieqk/datasets/Campus4K/campus4k-reid/detections/retina_arcface_0.95'
    train_dir_face = osp.join(face_root, 'train')
    query_dir_face = osp.join(face_root, 'query')
    gallery_dir_face = osp.join(face_root, 'gallery')

    save_root = '/data/xieqk/data/campus4k/exp/megcn/v2_insightface_n50/'
    # save_root = '/data4/xieqk/code/gitee/megcn/datasets/campus4k/graph_data_v2/'
    # save_root = '/data4/xieqk/code/gitee/megcn/datasets/campus4k_body/graph_data/'
    train_save_dir = osp.join(save_root, 'train')
    test_save_dir = osp.join(save_root, 'test')

    make_graph_raw(train_save_dir, train_feats_body, train_dir_face, train_feats_body, train_dir_face, is_train=True)
    make_graph_raw(test_save_dir, query_feats_body, query_dir_face, gallery_feats_body, gallery_dir_face)


def make_graph_raw(save_dir, q_npz_path, q_face_dir, g_npz_path, g_face_dir, is_train=False):
    print('loading query from {} & {}'.format(q_npz_path, q_face_dir))
    q_npz_body = np.load(q_npz_path)
    q_feats_body = q_npz_body['feats']
    q_pids, q_tids, q_camids = q_npz_body['pids'], q_npz_body['tids'], q_npz_body['camids']
    q_vidnames = ['{:04d}_{:04d}'.format(q_pids[i], q_tids[i]) for i in range(len(q_pids))]
    q_feats_face, q_face_flags = get_face_feats(q_vidnames, q_face_dir)
    print('loading gallery from {} & {}'.format(g_npz_path, g_face_dir))
    g_npz_body = np.load(g_npz_path)
    g_feats_body = g_npz_body['feats']
    g_pids, g_tids,  g_camids = g_npz_body['pids'], g_npz_body['tids'], g_npz_body['camids']
    g_vidnames = ['{:04d}_{:04d}'.format(g_pids[i], g_tids[i]) for i in range(len(g_pids))]
    g_feats_face, g_face_flags = get_face_feats(g_vidnames, g_face_dir)

    # Torch
    q_body_torch = torch.from_numpy(q_feats_body).cuda()
    g_body_torch = torch.from_numpy(g_feats_body).cuda()
    q_face_torch = torch.from_numpy(q_feats_face).cuda()
    g_face_torch = torch.from_numpy(g_feats_face).cuda()
    print('computing distance body ... ', q_body_torch.size(), g_body_torch.size())
    distmat_body_torch = compute_distance_matrix(q_body_torch, g_body_torch, 'euclidean')
    print('computing distance face ... ', q_face_torch.size(), g_face_torch.size())
    distmat_face_torch = compute_distance_matrix(q_face_torch, g_face_torch, 'euclidean')
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

    node_num = 20
    imglist = []
    match_sum = []
    g_vidnames = np.array(g_vidnames)
    save_dir_raw = osp.join(save_dir, 'raw')
    mkdir_if_missing(save_dir_raw)
    for i, vidname in enumerate(tqdm(q_vidnames)):
        _idxs = indices[i][:node_num]
        _g_vidnames = g_vidnames[_idxs]
        _g_feats = g_feats_body[_idxs]
        _g_feats_face = g_feats_face[_idxs]
        _q_pid = q_pids[i]
        _g_pids = g_pids[_idxs]
        _matches = np.zeros((node_num,))
        _matches[_q_pid == _g_pids] = 1
        assert _matches.sum() >= 0 and _matches.sum() <= node_num, (node_num, _matches.sum())
        match_sum.append(_matches.sum())
        if is_train:
            if _matches.sum() == node_num or _matches.sum() == 0:
                print('skip {}, matches_sum: {}, node_num: {}'.format(vidname, _matches.sum(), node_num))
                continue
        _sims_body = simmat_body[i][_idxs]
        _sims_face = simmat_face[i][_idxs]
        _sims = np.vstack((_sims_body, _sims_face)).T
        imglist.append('{}\n'.format(vidname))
        np.savez(
            osp.join(save_dir_raw, '{}_n{}.npz'.format(vidname, node_num)),
            imgidxs = _idxs, 
            matches = _matches,
            scores = _sims,
            # scores = _sims_body.reshape(-1,1),
            feats_body = _g_feats,
            feats_face = _g_feats_face
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

def bbox_select(bboxs, h, w, lambda_x, lambda_y, size):
    xc_mean, yc_mean = 0.478, 0.148
    xc_std, yc_std = 0.174, 0.122
    for idx, bbox in enumerate(bboxs):
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        xc = (x1+x2)/(2*w)
        yc = (y1+y2)/(2*h)
        if abs(xc-xc_mean) < lambda_x*xc_std and abs(yc-yc_mean) < lambda_y*yc_std:
            if (x2-x1)*(y2-y1) > size*size:
                return idx
    return None

def get_feat_select_and_size(data, size=32, lambda_x=1.5, lambda_y=1.5):
    if data:
        feat_per_img = []
        for img_name, img_info in data.items():
            height, width = img_info['height'], img_info['width']
            bboxs, points, feats = img_info['bbox'], img_info['points'], img_info['feats']
            idx = bbox_select(bboxs, height, width, lambda_x, lambda_y, size)
            if idx is None:
                continue
            else:
                face_feat = feats[idx]
                feat_per_img.append(face_feat)
        feat_per_img = np.array(feat_per_img)
        if feat_per_img.shape[0] <= 1:
            return None
        else:
            feat = feat_per_img.mean(axis=0)
            feat = preprocessing.normalize([feat], 'l2').reshape((-1,))
            return feat
    else:
        return None

def get_face_feats(vidnames, dir_face):
    feats_face, face_flags = [], []
    for vidname in tqdm(vidnames):
        pid_str, tid_str = vidname.split('_')
        json_dir = osp.join(dir_face, pid_str, tid_str)
        face_files = sorted(glob(osp.join(json_dir, '*.json')))
        tid_feats = []
        for json_file in face_files:
            face_data = read_json(json_file)
            feat = face_select_v2(face_data)
            if feat == None:
                continue
            tid_feats.append(feat)

        if len(tid_feats) == 0:
            feat_face = np.random.rand(512)
            feat_face = preprocessing.normalize([feat_face], 'l2').reshape((-1, ))
            feats_face.append(feat_face)
            face_flags.append(0)
        else:
            tid_feats = np.array(tid_feats)
            tid_feat = tid_feats.mean(axis=0)
            tid_feat = preprocessing.normalize([tid_feat], 'l2').reshape((-1,))
            feats_face.append(tid_feat)
            face_flags.append(1)
 
    return np.array(feats_face), np.array(face_flags)

def face_select_v2(face_data, size=32, lambda_x=1.5, lambda_y=1.5):
    bboxs = face_data['bboxs']
    xc_mean, yc_mean = 0.478, 0.148
    xc_std, yc_std = 0.174, 0.122
    h, w = face_data['height'], face_data['width']
    for idx, bbox in enumerate(bboxs):
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        xc = (x1+x2)/(2*w)
        yc = (y1+y2)/(2*h)
        if abs(xc-xc_mean) < lambda_x*xc_std and abs(yc-yc_mean) < lambda_y*yc_std:
            if (x2-x1)*(y2-y1) > size*size and bbox[4] > 0.92:
                return face_data['feats'][idx]
    return None

def feats_select(face_dict):
    bboxs, face_feats = face_dict['bbox'], face_dict['feats']
    if len(bboxs) > 0:
        scores = np.array(bboxs)[:, -1]
        idx = np.argsort(scores)[-1]
        feat_face = np.array(face_feats[idx])
        return feat_face, 1
    feat_face = np.random.rand(512)
    # feat_face = preprocessing.normalize([feat_face], 'l2').reshape((-1, ))
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
