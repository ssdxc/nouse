import sys
import os
import os.path as osp
import numpy as np
from tqdm import tqdm
from sklearn import preprocessing
from scipy.spatial import distance
from collections import Counter

import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

sys.path.append('.')
from utils.tools import mkdir_if_missing, read_json
from utils.distance import compute_distance_matrix
from metrics.rank import evaluate_rank


def main():
    root = '/home/xieqk/xieqk/code/gitee/megcn/exp/fast-reid/features/dukemtmc-reid/mgn_wo_norm'
    train_feats_body = osp.join(root, 't_feats_body_wo_norm.npz')
    query_feats_body = osp.join(root, 'q_feats_body_wo_norm.npz')
    gallery_feats_body = osp.join(root, 'g_feats_body_wo_norm.npz')

    save_root = '/home/xieqk/xieqk/code/gitee/megcn/exp/graph_data/dukemtmc-reid/mgn_n300/'
    train_save_dir = osp.join(save_root, 'train')
    test_save_dir = osp.join(save_root, 'test')

    make_graph_raw(train_save_dir, train_feats_body, train_feats_body, is_train=True)
    make_graph_raw(test_save_dir, query_feats_body, gallery_feats_body)


def make_graph_raw(save_dir, q_npz_path, g_npz_path, is_train=False):
    print('loading query from {}'.format(q_npz_path))
    q_npz_body = np.load(q_npz_path)
    q_imgnames = [osp.basename(x) for x in q_npz_body['paths']]
    qf = q_npz_body['feats']
    q_pids, q_camids = q_npz_body['pids'], q_npz_body['camids']
    qf_b1, qf_b2, qf_b3 = qf[:, :256].copy(), qf[:, 256:(256*2)].copy(), qf[:, (256*2):(256*3)].copy()
    qf_b21, qf_b22 = qf[:, (256*3):(256*4)].copy(), qf[:, (256*4):(256*5)].copy()
    qf_b31, qf_b32, qf_b33 = qf[:, (256*5):(256*6)].copy(), qf[:, (256*6):(256*7)].copy(), qf[:, (256*7):(256*8)].copy()
    
    print('loading gallery from {}'.format(g_npz_path))
    g_npz_body = np.load(g_npz_path)
    g_imgnames = [osp.basename(x) for x in g_npz_body['paths']]
    gf = g_npz_body['feats']
    g_pids, g_camids = g_npz_body['pids'], g_npz_body['camids']
    gf_b1, gf_b2, gf_b3 = gf[:, :256].copy(), gf[:, 256:(256*2)].copy(), gf[:, (256*2):(256*3)].copy()
    gf_b21, gf_b22 = gf[:, (256*3):(256*4)].copy(), gf[:, (256*4):(256*5)].copy()
    gf_b31, gf_b32, gf_b33 = gf[:, (256*5):(256*6)].copy(), gf[:, (256*6):(256*7)].copy(), gf[:, (256*7):(256*8)].copy()

    qf = preprocessing.normalize(qf, norm='l2')
    qf_b1  = preprocessing.normalize(qf_b1 , norm='l2')
    qf_b2  = preprocessing.normalize(qf_b2 , norm='l2')
    qf_b3  = preprocessing.normalize(qf_b3 , norm='l2')
    qf_b21 = preprocessing.normalize(qf_b21, norm='l2')
    qf_b22 = preprocessing.normalize(qf_b22, norm='l2')
    qf_b31 = preprocessing.normalize(qf_b31, norm='l2')
    qf_b32 = preprocessing.normalize(qf_b32, norm='l2')
    qf_b33 = preprocessing.normalize(qf_b33, norm='l2')
    gf = preprocessing.normalize(gf, norm='l2')
    gf_b1  = preprocessing.normalize(gf_b1 , norm='l2')
    gf_b2  = preprocessing.normalize(gf_b2 , norm='l2')
    gf_b3  = preprocessing.normalize(gf_b3 , norm='l2')
    gf_b21 = preprocessing.normalize(gf_b21, norm='l2')
    gf_b22 = preprocessing.normalize(gf_b22, norm='l2')
    gf_b31 = preprocessing.normalize(gf_b31, norm='l2')
    gf_b32 = preprocessing.normalize(gf_b32, norm='l2')
    gf_b33 = preprocessing.normalize(gf_b33, norm='l2')

    distmat = eval_reid_cuda(qf, gf, q_pids, g_pids, q_camids, g_camids, metric='euclidean')
    distmat_b1 = eval_reid_cuda(qf_b1, gf_b1, q_pids, g_pids, q_camids, g_camids, metric='euclidean')
    distmat_b2 = eval_reid_cuda(qf_b2, gf_b2, q_pids, g_pids, q_camids, g_camids, metric='euclidean')
    distmat_b3 = eval_reid_cuda(qf_b3, gf_b3, q_pids, g_pids, q_camids, g_camids, metric='euclidean')
    distmat_b21 = eval_reid_cuda(qf_b21, gf_b21, q_pids, g_pids, q_camids, g_camids, metric='euclidean')
    distmat_b22 = eval_reid_cuda(qf_b22, gf_b22, q_pids, g_pids, q_camids, g_camids, metric='euclidean')
    distmat_b31 = eval_reid_cuda(qf_b31, gf_b31, q_pids, g_pids, q_camids, g_camids, metric='euclidean')
    distmat_b32 = eval_reid_cuda(qf_b32, gf_b32, q_pids, g_pids, q_camids, g_camids, metric='euclidean')
    distmat_b33 = eval_reid_cuda(qf_b33, gf_b33, q_pids, g_pids, q_camids, g_camids, metric='euclidean')

    indices = np.argsort(distmat)
    # indices_b1 = np.argsort(distmat_b1)
    # indices_b2 = np.argsort(distmat_b2)
    # indices_b3 = np.argsort(distmat_b3)
    # indices_b21 = np.argsort(distmat_b21)
    # indices_b22 = np.argsort(distmat_b22)
    # indices_b31 = np.argsort(distmat_b31)
    # indices_b32 = np.argsort(distmat_b32)
    # indices_b33 = np.argsort(distmat_b33)
    
    simmat = 1 - distmat
    simmat_b1 = 1 - distmat_b1
    simmat_b2 = 1 - distmat_b2
    simmat_b3 = 1 - distmat_b3
    simmat_b21 = 1 - distmat_b21
    simmat_b22 = 1 - distmat_b22
    simmat_b31 = 1 - distmat_b31
    simmat_b32 = 1 - distmat_b32
    simmat_b33 = 1 - distmat_b33

    num = 300

    imglist = []
    match_sum = []
    dup_num = []
    g_imgnames = np.array(g_imgnames)
    save_dir_raw = osp.join(save_dir, 'raw')
    mkdir_if_missing(save_dir_raw)
    for i, imgname in enumerate(tqdm(q_imgnames)):
        _idxs = indices[i][:num]

        _g_imgnames = g_imgnames[_idxs]
        _g_feats = gf[_idxs]
        _g_feats_b1 = gf_b1[_idxs]
        _g_feats_b2 = gf_b2[_idxs]
        _g_feats_b3 = gf_b3[_idxs]
        _g_feats_b21 = gf_b21[_idxs]
        _g_feats_b22 = gf_b22[_idxs]
        _g_feats_b31 = gf_b31[_idxs]
        _g_feats_b32 = gf_b32[_idxs]
        _g_feats_b33 = gf_b33[_idxs]

        _q_pid = q_pids[i]
        _g_pids = g_pids[_idxs]
        _matches = np.zeros((num,))
        _matches[_q_pid == _g_pids] = 1
        assert _matches.sum() >= 0 and _matches.sum() <= num, (num, _matches.sum())
        match_sum.append(_matches.sum())
        if is_train:
            if _matches.sum() == num or _matches.sum() == 0:
                print('skip {}, matches_sum: {}, num: {}'.format(imgname, _matches.sum(), num))
                continue

        _sims = simmat[i][_idxs]
        _sims_b1  = simmat_b1[i][_idxs]
        _sims_b2  = simmat_b2[i][_idxs]
        _sims_b3  = simmat_b3[i][_idxs]
        _sims_b21 = simmat_b21[i][_idxs]
        _sims_b22 = simmat_b22[i][_idxs]
        _sims_b31 = simmat_b31[i][_idxs]
        _sims_b32 = simmat_b32[i][_idxs]
        _sims_b33 = simmat_b33[i][_idxs]
        _combine_sims = np.vstack((
            _sims,
            _sims_b1,
            _sims_b2,
            _sims_b3,
            _sims_b21,
            _sims_b22,
            _sims_b31,
            _sims_b32,
            _sims_b33
        )).T
        imglist.append('{}\n'.format(imgname))
        np.savez(
            osp.join(save_dir_raw, '{}_n{}.npz'.format(imgname.split('.')[0], num)),
            imgidxs = _idxs, 
            matches = _matches,
            scores = _combine_sims,
            feats = _g_feats,
            feats_b1  = _g_feats_b1 ,
            feats_b2  = _g_feats_b2 ,
            feats_b3  = _g_feats_b3 ,
            feats_b21 = _g_feats_b21,
            feats_b22 = _g_feats_b22,
            feats_b31 = _g_feats_b31,
            feats_b32 = _g_feats_b32,
            feats_b33 = _g_feats_b33,
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

    # dup_num = np.array(dup_num)
    # print('dup_num statistic: ', Counter(dup_num))
    # print('mean: {}, min: {}, max: {}'.format(dup_num.mean(), dup_num.min(), dup_num.max()))


def get_face_head_feats(imgnames, dir_face):
    feats = []
    for imgname in tqdm(imgnames):
        imgname_base = osp.splitext(imgname)[0]
        json_name_face = imgname_base + '.json'
        npy_name_head = imgname_base + '.npy'
        json_path_face = osp.join(dir_face, json_name_face)
        if osp.exists(json_path_face):
            face_dict = read_json(json_path_face)
            feat = face_dict['feats'][0]
            feat = np.array(feat)
        else:
            feat = np.random.random((512,))
            feat = preprocessing.normalize([feat], 'l2').reshape((-1, ))
        
        assert abs(np.sum(feat*feat)-1) <0.01, (feat.shape, np.sum(feat*feat))
        feats.append(feat)
    return np.array(feats)


def eval_reid(qf, gf, q_pids, g_pids, q_camids, g_camids, metric='cosine'):
    print('Numpy: computing distance body ... ', qf.shape, gf.shape)
    distmat = distance.cdist(qf, gf, metric)
    # print(distmat.dtype)
    all_cmc, mAP = evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids)
    print('mAP: {} / rank: {} {} {} {}'.format(mAP, all_cmc[0], all_cmc[5-1], all_cmc[10-1], all_cmc[20-1]))
    return distmat

def eval_reid_cuda(qf, gf, q_pids, g_pids, q_camids, g_camids, metric='cosine'):
    qf_torch = torch.from_numpy(qf.astype(np.float64)).cuda()
    gf_torch = torch.from_numpy(gf.astype(np.float64)).cuda()
    print('Pytorch: computing distance body ... ', qf_torch.size(), gf_torch.size())
    # distmat_torch = compute_distance_matrix(qf_torch, gf_torch, 'euclidean')
    distmat_torch = compute_distance_matrix(qf_torch, gf_torch, metric)
    distmat = distmat_torch.cpu().numpy()
    # print(distmat.dtype)
    all_cmc, mAP = evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids)
    print('mAP: {} / rank: {} {} {} {}'.format(mAP, all_cmc[0], all_cmc[5-1], all_cmc[10-1], all_cmc[20-1]))
    return distmat


if __name__ == '__main__':
    main()
