import sys
import os
import os.path as osp
import numpy as np
from tqdm import tqdm
from sklearn import preprocessing
from scipy.spatial import distance
from collections import Counter

import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

sys.path.append('.')
from utils.tools import mkdir_if_missing, read_json
from utils.distance import compute_distance_matrix


def main():
    root = '/data2/xieqk/data/vc-clothes'
    reid_model = 'bot512'
    face_model = 'retina_arcface_0.01'
    train_feats_body = osp.join(root, 'features', 'fast-reid', reid_model, 't_feats_body.npz')
    query_feats_body = osp.join(root, 'features', 'fast-reid', reid_model, 'q_feats_body.npz')
    gallery_feats_body = osp.join(root, 'features', 'fast-reid', reid_model, 'g_feats_body.npz')
    train_dir_face = osp.join(root, 'features', 'face', face_model, 'train')
    query_dir_face = osp.join(root, 'features', 'face', face_model, 'query')
    gallery_dir_face = osp.join(root, 'features', 'face', face_model, 'gallery')

    save_root = '/data2/xieqk/data/vc-clothes/exp/bot512_insightface_n300/'
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
    q_feats_face = get_face_head_feats(q_imgnames, q_face_dir)
    print('loading gallery from {} & {}'.format(g_npz_path, g_face_dir))
    g_npz_body = np.load(g_npz_path)
    g_imgnames = [osp.basename(x) for x in g_npz_body['paths']]
    g_feats_body = g_npz_body['feats']
    g_pids, g_camids = g_npz_body['pids'], g_npz_body['camids']
    g_feats_face = get_face_head_feats(g_imgnames, g_face_dir)

    # # Torch
    # q_body_torch = torch.from_numpy(q_feats_body).cuda()
    # g_body_torch = torch.from_numpy(g_feats_body).cuda()
    # q_face_torch = torch.from_numpy(q_feats_face).cuda()
    # g_face_torch = torch.from_numpy(g_feats_face).cuda()
    # print('Torch: computing distance body ... ', q_body_torch.size(), g_body_torch.size())
    # distmat_body_torch = compute_distance_matrix(
    #     q_body_torch, g_body_torch, 'euclidean')
    # # distmat_body_torch = compute_distance_matrix(
    # #     q_body_torch, g_body_torch, 'cosine')
    # print('Torch: computing distance face ... ', q_face_torch.size(), g_face_torch.size())
    # distmat_face_torch = compute_distance_matrix(
    #     q_face_torch, g_face_torch, 'euclidean')
    # # distmat_face_torch = compute_distance_matrix(
    # #     q_face_torch, g_face_torch, 'cosine')
    # distmat_body = distmat_body_torch.cpu().numpy()
    # distmat_face = distmat_face_torch.cpu().numpy()
    # print('distmat shape: body {}\n'.format(distmat_body.shape), distmat_body)
    # print('distmat shape: face {}\n'.format(distmat_face.shape), distmat_face)

    # Numpy
    print('Numpy: computing distance body ... ', q_feats_body.shape, g_feats_body.shape)
    distmat_body = distance.cdist(q_feats_body, g_feats_body, 'euclidean')
    # distmat_body = distance.cdist(q_feats_body, g_feats_body, 'cosine')
    print('Numpy: computing distance face ... ', q_feats_face.shape, g_feats_face.shape)
    distmat_face = distance.cdist(q_feats_face, g_feats_face, 'euclidean')
    # distmat_face = distance.cdist(q_feats_face, g_feats_face, 'cosine')
    print('distmat shape: body {}\n'.format(distmat_body.shape), distmat_body)
    print('distmat shape: face {}\n'.format(distmat_face.shape), distmat_face)

    simmat_body = 1 - distmat_body
    simmat_face = 1 - distmat_face

    # for i, q_flag in enumerate(q_face_flags):
    #     for j, g_flag in enumerate(g_face_flags):
    #         if q_flag == 0 or g_flag == 0:
    #             simmat_face[i, j] = 0

    indices_body = np.argsort(distmat_body, axis=1)
    indices_face = np.argsort(distmat_face, axis=1)

    num_full = 300
    num_face = 100

    imglist = []
    match_sum = []
    dup_num = []
    g_imgnames = np.array(g_imgnames)
    save_dir_raw = osp.join(save_dir, 'raw')
    mkdir_if_missing(save_dir_raw)
    for i, imgname in enumerate(tqdm(q_imgnames)):
        _idxs_full = indices_body[i][:num_full]
        _idxs_face = indices_face[i][:num_face]
        _idxs = []
        for idx in _idxs_face:
            _idxs.append(idx)
        dup = 0
        for idx in _idxs_full:
            if len(_idxs) == num_full:
                break
            elif idx not in _idxs:
                _idxs.append(idx)
            else:
                dup += 1
        dup_num.append(dup)
        assert len(_idxs) == num_full, len(_idxs)
        _idxs = np.array(_idxs)

        _g_imgnames = g_imgnames[_idxs]
        _g_feats_body = g_feats_body[_idxs]
        _g_feats_face = g_feats_face[_idxs]
        _q_pid = q_pids[i]
        _g_pids = g_pids[_idxs]
        _matches = np.zeros((num_full,))
        _matches[_q_pid == _g_pids] = 1
        assert _matches.sum() >= 0 and _matches.sum() <= num_full, (num_full, _matches.sum())
        match_sum.append(_matches.sum())
        if is_train:
            if _matches.sum() == num_full or _matches.sum() == 0:
                print('skip {}, matches_sum: {}, num_full: {}'.format(imgname, _matches.sum(), num_full))
                continue

        _sims_body = simmat_body[i][_idxs]
        _sims_face = simmat_face[i][_idxs]
        _sims = np.vstack((_sims_body, _sims_face)).T
        imglist.append('{}\n'.format(imgname))
        np.savez(
            osp.join(save_dir_raw, '{}_n{}.npz'.format(imgname.split('.')[0], num_full)),
            imgidxs = _idxs, 
            matches = _matches,
            scores = _sims,
            feats_body = _g_feats_body,
            feats_face = _g_feats_face
        )
    
    with open(osp.join(save_dir, 'imglist.txt'), 'w') as f:
        f.write(''.join(imglist))
    
    np.save(osp.join(save_dir, 'query_pids.npy'), q_pids)
    np.save(osp.join(save_dir, 'query_camids.npy'), q_camids)
    np.save(osp.join(save_dir, 'gallery_pids.npy'), g_pids)
    np.save(osp.join(save_dir, 'gallery_camids.npy'), g_camids)
    
    match_sum = np.array(match_sum)
    print('match_sum statistic: ', Counter(match_sum))
    print('mean: {}, min: {}, max: {}'.format(match_sum.mean(), match_sum.min(), match_sum.max()))

    dup_num = np.array(dup_num)
    print('dup_num statistic: ', Counter(dup_num))
    print('mean: {}, min: {}, max: {}'.format(dup_num.mean(), dup_num.min(), dup_num.max()))


def get_face_head_feats(imgnames, dir_face):
    feats = []
    for imgname in tqdm(imgnames):
        imgname_base = osp.splitext(imgname)[0]
        json_name_face = imgname_base + '.json'
        npy_name_head = imgname_base + '.npy'
        json_path_face = osp.join(dir_face, json_name_face)
        if osp.exists(json_path_face):
            face_dict = read_json(json_path_face)
            # feat = face_dict['feats'][0]
            # feat = np.array(feat)
            feat = face_select(face_dict)
            if feat is None:
                feat = np.random.random((512,))
                feat = preprocessing.normalize([feat], 'l2').reshape((-1,))
            else:
                feat = np.array(feat)
        else:
            feat = np.random.random((512,))
            feat = preprocessing.normalize([feat], 'l2').reshape((-1, ))
        
        assert abs(np.sum(feat*feat)-1) <0.01, np.sum(feat*feat)
        feats.append(feat)
    return np.array(feats)

def face_select(face_dict):
    h = face_dict["height"]
    bboxs = face_dict["bboxs"]
    for i, bbox in enumerate(bboxs):
        score = bbox[4]
        if score < 0.93:
            continue
        y_min, y_max = bbox[1], bbox[3]
        yc = (y_min+y_max)/2
        if yc <= 0.2*h:
            return face_dict["feats"][i]
    return None


if __name__ == '__main__':
    main()
