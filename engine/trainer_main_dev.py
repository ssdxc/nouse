import datetime
import json
import os
import time
import os.path as osp
import numpy as np
from collections import Counter
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader

from utils import mkdir_if_missing, AverageMeter, MetricMeter
from metrics import evaluate_rank


class trainer_main_dev(object):

    def __init__(self, cfg, model, optimizer, scheduler, warmup_epoch, warmup_scheduler, loss, dataset):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = model.to(self.device)
        self.model.train()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_epoch = warmup_epoch
        self.warmup_scheduler = warmup_scheduler
        self.loss = loss
        self.dataset = dataset
        self.train_loader = DataLoader(
            self.dataset.graph_train, 
            batch_size=cfg.TRAIN.BATCH_SIZE, 
            shuffle=True, 
            num_workers=4
        )
        self.test_loader = DataLoader(
            self.dataset.graph_test, 
            batch_size=1, 
            shuffle=False, 
            num_workers=4
        )
        self.test_data = self.dataset.graph_test
        self.checkpoint_dir = osp.join(cfg.CHECKPOINTS_DIR, 'checkpoints')
        self.tensorboard_dir = osp.join(cfg.CHECKPOINTS_DIR, 'tensorboard')
        self.writer = SummaryWriter(self.tensorboard_dir)
        mkdir_if_missing(self.checkpoint_dir)
        mkdir_if_missing(self.tensorboard_dir)
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.eval_freq = cfg.TEST.EVAL_FREQ
        self.print_freq = cfg.TRAIN.PRINT_FREQ

    def save_model(self, epoch, mAP, rank1):
        mdoel_dict = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
            'mAP': mAP,
            'rank1': rank1
        }
        save_path = osp.join(self.checkpoint_dir, "model_epoch_{}.pth.tar".format(epoch))
        torch.save(mdoel_dict, save_path)
        if (epoch + 1) == self.max_epoch:
            os.rename(save_path, osp.join(osp.dirname(save_path), 'model_final_epoch_{}.pth.tar'.format(epoch)))

    def forward(self, batch_graph_body, batch_graph_face):
        """
        batch_graph: batch of graphs (torch_geometric.data.Data object)
        return: scores, labels and batch vector for the batch of graphs
        """
        for k in batch_graph_body.keys:
            batch_graph_body[k] = batch_graph_body[k].to(self.device)
        for k in batch_graph_face.keys:
            batch_graph_face[k] = batch_graph_face[k].to(self.device)
            # => move all the tensors in batch_graph to device
        scores = self.model(batch_graph_body, batch_graph_face)
        # print(batch_graph_body.y.cpu().tolist(), batch_graph_body.y)
        assert batch_graph_body.y.cpu().tolist() == batch_graph_face.y.cpu().tolist()
        assert batch_graph_body.batch.cpu().tolist() == batch_graph_face.batch.cpu().tolist()
        # returns 1-D tensors:
        return scores, batch_graph_body.y, batch_graph_body.batch

    def run(self):
        time_start = time.time()
        print('=> Start training')

        best_mAP, best_rank1, best_epoch = 0, 0, 0
        mAP, rank1 = self.test()
        self.writer.add_scalar('Test/mAP', mAP, 0)
        self.writer.add_scalar('Test/rank1', rank1, 0)

        for epoch in range(1, self.max_epoch + 1):
            self.train(epoch)
            if epoch <= self.warmup_epoch:
                self.warmup_scheduler.step()
            else:
                self.scheduler.step()

            if epoch % self.eval_freq == 0 and epoch != self.max_epoch:
                mAP, rank1 = self.test()
                if mAP > best_mAP:
                    best_mAP = mAP
                    best_rank1 = rank1
                    best_epoch = epoch
                self.writer.add_scalar('Test/mAP', mAP, epoch)
                self.writer.add_scalar('Test/rank1', rank1, epoch)
                self.save_model(epoch, mAP, rank1)
        
        if self.max_epoch > 0:
            assert epoch == self.max_epoch, 'current epoch: {}'.format(epoch)
            print('=> Final test')
            mAP, rank1 = self.test()
            if mAP > best_mAP:
                best_mAP = mAP
                best_rank1 = rank1
                best_epoch = epoch
            self.writer.add_scalar('Test/mAP', mAP, epoch)
            self.writer.add_scalar('Test/rank1', rank1, epoch)
            self.save_model(self.max_epoch, mAP, rank1)

        elapsed = round(time.time() - time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print('Elapsed {}'.format(elapsed))
        print('Best result at epoch {}: mAP {}, Rank-1 {}'.format(best_epoch, best_mAP, best_rank1))
        self.writer.close()

    def get_model_names(self, names=None):
        names_real = list(self._models.keys())
        if names is not None:
            if not isinstance(names, list):
                names = [names]
            for name in names:
                assert name in names_real
            return names
        else:
            return names_real
    
    def get_current_lr(self, names=None):
        names = self.get_model_names(names)
        name = names[0]
        return self._optims[name].param_groups[-1]['lr']

    def train(self, epoch):
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        self.model.train()
        num_batches = len(self.train_loader)
        end = time.time()
        for batch_idx, (batch_graph_body, batch_graph_face) in enumerate(self.train_loader):
            # print(batch_graph_body, batch_graph_face)
            data_time.update(time.time() - end)
            self.optimizer.zero_grad()
            scores, labels, batch_vec = self.forward(batch_graph_body, batch_graph_face)
            loss = self.loss(scores, labels, batch_vec)
            loss.backward()
            self.optimizer.step()
            batch_time.update(time.time() - end)
            losses.update({'loss': loss.item()})

            if (batch_idx+1) % self.print_freq == 0:
                nb_this_epoch = num_batches - (batch_idx + 1)
                nb_future_epochs = (self.max_epoch - epoch) * num_batches
                eta_seconds = batch_time.avg * (nb_this_epoch + nb_future_epochs)
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(
                    'epoch: [{0}/{1}][{2}/{3}]  '
                    'time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'data {data_time.val:.3f} ({data_time.avg:.3f})  '
                    'eta {eta}  '
                    '{losses}, lr {lr:.2e}'.format(
                        epoch,
                        self.max_epoch,
                        batch_idx + 1,
                        num_batches,
                        batch_time=batch_time,
                        data_time=data_time,
                        eta=eta_str,
                        losses=losses,
                        lr=self.optimizer.param_groups[0]['lr']
                    )
                )
                # mAP, rank1 = self.test()
                # self.model.train()
            
            n_iter = epoch * num_batches + batch_idx
            self.writer.add_scalar('Train/time', batch_time.avg, n_iter)
            self.writer.add_scalar('Train/time_iter', batch_time.val, n_iter)
            self.writer.add_scalar('Train/data', data_time.avg, n_iter)
            self.writer.add_scalar('Train/data_iter', data_time.val, n_iter)
            for name, meter in losses.meters.items():
                self.writer.add_scalar('Train/' + name, meter.avg, n_iter)
                self.writer.add_scalar('Train/' + name + '_iter', meter.val, n_iter)
            
            end = time.time()

    def test(self):
        print('##### Evaluating #####')
        q_num, g_num = len(self.dataset.q_pids), len(self.dataset.g_pids)
        score_mat = np.full((q_num, g_num), -np.inf)
        
        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(len(self.test_data))):
                query_body, query_face = self.test_data[i]
                qidx_body, qidx_face = query_body.qid.tolist()[0], query_face.qid.tolist()[0]
                # print(qidx_body, qidx_face)
                assert qidx_body == qidx_face, (qidx_body, qidx_face)
                for k in query_body.keys:
                    # print(k)
                    query_body[k] = query_body[k].to(self.device)
                for k in query_face.keys:
                    query_face[k] = query_face[k].to(self.device)
                scores = self.model(query_body, query_face).tolist()
                assert query_body.imgidxs.tolist() == query_face.imgidxs.tolist()
                for score, imgidx in zip(scores, query_body.imgidxs.tolist()):
                    score_mat[qidx_body, int(imgidx)] = score
        
        distmat = -score_mat
        print('Done, dist matrix:', distmat.shape)
        print('Computing CMC and mAP ...')
        cmc, mAP = evaluate_rank(
            distmat,
            self.dataset.q_pids,
            self.dataset.g_pids,
            self.dataset.q_camids,
            self.dataset.g_camids,
            use_metric_cuhk03=False
        )

        print("Results ----------")
        print("mAP: {:.1%}".format(np.mean(mAP)))
        print("CMC curve")
        for r in [1,5,10,20]:
            print("Rank-{:<3}: {:.1%}".format(r, cmc[r-1]))
        print("------------------")

        return mAP, cmc[0]

    # def test(self):
    #     print('##### Evaluating #####')
    #     q_num, g_num = len(self.dataset.q_pids), len(self.dataset.g_pids)
    #     score_mat = np.full((q_num, g_num), -np.inf)
        
    #     self.model.eval()
    #     with torch.no_grad():
    #         for (query_body, query_face) in tqdm(self.test_loader):
    #             scores, _, _ = self.forward(query_body, query_face)
    #             scores = scores.tolist()
    #             qidx_body, qidx_face = query_body.qid.tolist()[0], query_face.qid.tolist()[0]
    #             # print(qidx_body, qidx_face)
    #             assert qidx_body == qidx_face, (qidx_body, qidx_face)
    #             assert query_body.imgidxs.tolist() == query_face.imgidxs.tolist()
    #             for score, imgidx in zip(scores, query_body.imgidxs.tolist()):
    #                 score_mat[qidx_body, int(imgidx)] = score
        
    #     distmat = -score_mat
    #     print('Done, dist matrix:', distmat.shape)
    #     print('Computing CMC and mAP ...')
    #     cmc, mAP = evaluate_rank(
    #         distmat,
    #         self.dataset.q_pids,
    #         self.dataset.g_pids,
    #         self.dataset.q_camids,
    #         self.dataset.g_camids,
    #         use_metric_cuhk03=False
    #     )

    #     print("Results ----------")
    #     print("mAP: {:.1%}".format(np.mean(mAP)))
    #     print("CMC curve")
    #     for r in [1,5,10,20]:
    #         print("Rank-{:<3}: {:.1%}".format(r, cmc[r-1]))
    #     print("------------------")

    #     return mAP, cmc[0]

        
        









