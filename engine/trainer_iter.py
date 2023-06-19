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


class Trainer_iter(object):

    def __init__(self, cfg, model, optimizer, loss, dataset):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = model.to(self.device)
        self.model.train()
        self.optimizer = optimizer
        self.loss = loss
        self.dataset = dataset
        self.train_loader = DataLoader(
            self.dataset.graph_train, 
            batch_size=cfg.TRAIN.BATCH_SIZE, 
            shuffle=True, 
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

    def forward(self, batch_graph):
        """
        batch_graph: batch of graphs (torch_geometric.data.Data object)
        return: scores, labels and batch vector for the batch of graphs
        """
        for k in batch_graph.keys:
            batch_graph[k] = batch_graph[k].to(self.device)
            # => move all the tensors in batch_graph to device
        scores = self.model(batch_graph)
        # returns 1-D tensors:
        return scores, batch_graph.y, batch_graph.batch

    def run(self):
        time_start = time.time()
        print('=> Start training')

        best_mAP, best_rank1, best_epoch = 0, 0, 0
        mAP, rank1 = self.test()

        for epoch in range(1, self.max_epoch + 1):
            self.train(epoch)

            if epoch % self.eval_freq == 0 and epoch != self.max_epoch:
                mAP, rank1 = self.test()
                if mAP > best_mAP:
                    best_mAP = mAP
                    best_rank1 = rank1
                    best_epoch = epoch
                # self.writer.add_scalar('Test/mAP', mAP, epoch)
                # self.writer.add_scalar('Test/rank1', rank1, epoch)
                self.save_model(epoch, mAP, rank1)
        
        if self.max_epoch > 0:
            assert epoch == self.max_epoch, 'current epoch: {}'.format(epoch)
            print('=> Final test')
            mAP, rank1 = self.test()
            if mAP > best_mAP:
                best_mAP = mAP
                best_rank1 = rank1
                best_epoch = epoch
            # self.writer.add_scalar('Test/mAP', mAP, epoch)
            # self.writer.add_scalar('Test/rank1', rank1, epoch)
            self.save_model(self.max_epoch, mAP, rank1)

        elapsed = round(time.time() - time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print('Elapsed {}'.format(elapsed))
        print('Best result at epoch {}: mAP {}, Rank-1 {}'.format(best_epoch, best_mAP, best_rank1))
        self.writer.close()

    def train(self, epoch):
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        self.model.train()
        num_batches = len(self.train_loader)
        end = time.time()
        for batch_idx, batch_graph in enumerate(self.train_loader):
            data_time.update(time.time() - end)
            self.optimizer.zero_grad()
            scores, labels, batch_vec = self.forward(batch_graph)
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
                    '{losses}'.format(
                        epoch,
                        self.max_epoch,
                        batch_idx + 1,
                        num_batches,
                        batch_time=batch_time,
                        data_time=data_time,
                        eta=eta_str,
                        losses=losses
                    )
                )
            mAP, rank1 = self.test()
            
            # n_iter = epoch * num_batches + batch_idx
            n_iter = epoch + batch_idx/num_batches
            self.writer.add_scalar('Test/mAP', mAP, n_iter)
            self.writer.add_scalar('Test/rank1', rank1, n_iter)
            self.writer.add_scalar('Train/time', batch_time.avg, n_iter)
            self.writer.add_scalar('Train/data', data_time.avg, n_iter)
            for name, meter in losses.meters.items():
                self.writer.add_scalar('Train/' + name, meter.avg, n_iter)
            
            end = time.time()

    def test(self):
        print('##### Evaluating #####')
        q_num, g_num = len(self.dataset.q_pids), len(self.dataset.g_pids)
        score_mat = np.full((q_num, g_num), -np.inf)
        
        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(len(self.test_data))):
                query = self.test_data[i]
                qidx = query.qid.tolist()[0]
                for k in query.keys:
                    query[k] = query[k].to(self.device)
                scores = self.model(query).tolist()
                for score, imgidx in zip(scores, query.imgidxs.tolist()):
                    score_mat[qidx, int(imgidx)] = score
        
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

        
        









