import torch
from torch_geometric.utils import to_dense_batch

from utils.utils import masking

"""
pairwise loss class
"""


class BasePairwiseLoss():
    def __init__(self):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def __call__(self, scores, labels, batch_vec):
        """
        * the three input tensors have shape (N, ), N being the number of nodes in the batch
        * what makes possible to split values by query (i.e. graph) is the batch_vec vector, indicating which node
        belongs to which graph
        we want to compute all the pairwise contributions in the batch, dealing with:
        1. not mixing between graphs
        2. variable number of valid pairs between graphs (using masking)
        """
        ids_pos = labels == 1
        ids_neg = labels == 0
        batch_vec_pos = batch_vec[ids_pos]
        batch_vec_neg = batch_vec[ids_neg]
        pos_scores = scores[ids_pos]
        neg_scores = scores[ids_neg]
        # print('1', ids_pos.size(), ids_neg.size(), batch_vec_pos.size(), batch_vec_neg.size(), pos_scores.size(), neg_scores.size())
        # print('2', pos_scores.size(), set(batch_vec_pos.tolist()))
        # print('3', neg_scores.size(), set(batch_vec_neg.tolist()))
        # densify the tensors (see: https://rusty1s.github.io/pytorch_geometric/build/html/modules/utils.html?highlight=to_dense#torch_geometric.utils.to_dense_batch)
        dense_pos_scores, pos_mask = to_dense_batch(pos_scores, batch_vec_pos, fill_value=0)
        # dense_pos_scores has shape (nb_graphs, padding => max number nodes for graphs in batch)
        pos_len = torch.sum(pos_mask, dim=-1)  # shape (nb_graphs, ), actual number of nodes per graph
        dense_neg_scores, neg_mask = to_dense_batch(neg_scores, batch_vec_neg, fill_value=0)
        neg_len = torch.sum(neg_mask, dim=-1)
        max_pos_len = pos_len.max()  # == the padding value for the positive scores
        max_neg_len = neg_len.max()

        # print('4', pos_scores.size(), neg_scores.size())
        # print('5', dense_pos_scores.size(), pos_mask.size(), pos_len.size(), max_pos_len.item())
        # print('6', dense_neg_scores.size(), neg_mask.size(), neg_len.size(), max_neg_len.item())
        # print('7', pos_mask)
        # print('8', neg_mask)
        # print('---')

        # pos_mask_old, neg_mask_old = pos_mask, neg_mask
        pos_mask = masking(pos_len, max_pos_len.item())
        neg_mask = masking(neg_len, max_neg_len.item())

        # print('9', pos_mask.size(), neg_mask.size())
        # print('10', pos_mask, pos_mask==pos_mask_old)
        # print('11', neg_mask, neg_mask==neg_mask_old)
        # print('\n')

        # assert dense_pos_scores.size(0) == 16, (batch_vec_pos.tolist(), )
        # print(dense_pos_scores.size(), dense_neg_scores.size())
        # print(dense_pos_scores.view(-1, 1, dense_pos_scores.size(1)).size(), dense_neg_scores.view(-1, dense_neg_scores.size(1), 1).size())
        diff_ = dense_pos_scores.view(-1, 1, dense_pos_scores.size(1)) - dense_neg_scores.view(-1,
                                                                                               dense_neg_scores.size(1),
                                                                                               1)
        # print(diff_.size())
        # now we use the mask and some reshaping to only extract the valid pair contributions:
        pos_mask_ = pos_mask.repeat(1, neg_mask.size(1))
        neg_mask_ = neg_mask.view(-1, neg_mask.size(1), 1).repeat(1, 1, pos_mask.size(1)).view(-1, neg_mask.size(
            1) * pos_mask.size(1))
        flattened_mask = (pos_mask_ * neg_mask_).view(-1).long()
        valid_diff_ = diff_.view(-1)[flattened_mask > 0]
        loss = self.compute_loss(valid_diff_)
        return loss

    def compute_loss(self, valid_diff_):
        raise NotImplementedError


class BPRLoss(BasePairwiseLoss):
    """
    BPR loss: compute the P(i >> j) = sigmoid(si - sj) and then do cross-entropy
    """

    def __init__(self):
        super().__init__()
        self.loss = torch.nn.BCEWithLogitsLoss(reduction="mean")

    def compute_loss(self, valid_diff_):
        labels = torch.ones(valid_diff_.size(0)).to(self.device).float()  # we only have labels == 1 because we compute
        # the s(+) - s(-)
        return self.loss(valid_diff_, labels)


class BCELoss():
    def __init__(self):
        self.loss = torch.nn.BCEWithLogitsLoss(reduction="mean")
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    def __call__(self, scores, labels, batch_vec):
        scores, labels = scores.view(-1,), labels.view(-1,)
        return self.loss(scores, labels)



class Triplet():
    def __init__(self):
        print('Triplet loss')
        margin = 1
        self.ranking_loss = torch.nn.MarginRankingLoss(margin=margin)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def __call__(self, scores, labels, batch_vec):
        """
        * the three input tensors have shape (N, ), N being the number of nodes in the batch
        * what makes possible to split values by query (i.e. graph) is the batch_vec vector, indicating which node
        belongs to which graph
        we want to compute all the pairwise contributions in the batch, dealing with:
        1. not mixing between graphs
        2. variable number of valid pairs between graphs (using masking)
        """
        ids_pos = labels == 1
        ids_neg = labels == 0
        batch_vec_pos = batch_vec[ids_pos]
        batch_vec_neg = batch_vec[ids_neg]
        pos_scores = scores[ids_pos]
        neg_scores = scores[ids_neg]
        # print('1', ids_pos.size(), ids_neg.size(), batch_vec_pos.size(), batch_vec_neg.size(), pos_scores.size(), neg_scores.size())
        # print('2', pos_scores.size(), set(batch_vec_pos.tolist()))
        # print('3', neg_scores.size(), set(batch_vec_neg.tolist()))
        # densify the tensors (see: https://rusty1s.github.io/pytorch_geometric/build/html/modules/utils.html?highlight=to_dense#torch_geometric.utils.to_dense_batch)
        dense_pos_scores, pos_mask = to_dense_batch(pos_scores, batch_vec_pos, fill_value=0)
        # dense_pos_scores has shape (nb_graphs, padding => max number nodes for graphs in batch)
        pos_len = torch.sum(pos_mask, dim=-1)  # shape (nb_graphs, ), actual number of nodes per graph
        dense_neg_scores, neg_mask = to_dense_batch(neg_scores, batch_vec_neg, fill_value=0)
        neg_len = torch.sum(neg_mask, dim=-1)
        max_pos_len = pos_len.max()  # == the padding value for the positive scores
        max_neg_len = neg_len.max()

        # print('4', pos_scores.size(), neg_scores.size())
        # print('5', dense_pos_scores.size(), pos_mask.size(), pos_len.size(), max_pos_len.item())
        # print('6', dense_neg_scores.size(), neg_mask.size(), neg_len.size(), max_neg_len.item())
        # print('7', pos_mask)
        # print('8', neg_mask)
        # print('---')

        # pos_mask_old, neg_mask_old = pos_mask, neg_mask
        pos_mask = masking(pos_len, max_pos_len.item())
        neg_mask = masking(neg_len, max_neg_len.item())

        # print('9', pos_mask.size(), neg_mask.size())
        # print('10', pos_mask, pos_mask==pos_mask_old)
        # print('11', neg_mask, neg_mask==neg_mask_old)
        # print('\n')

        # assert dense_pos_scores.size(0) == 16, (batch_vec_pos.tolist(), )
        # print(dense_pos_scores.size(), dense_neg_scores.size())
        # print(dense_pos_scores.view(-1, 1, dense_pos_scores.size(1)).size(), dense_neg_scores.view(-1, dense_neg_scores.size(1), 1).size())
        dense_pos_scores_ = dense_pos_scores.repeat(1, neg_mask.size(1))
        dense_neg_scores_ = dense_neg_scores.view(-1, dense_neg_scores.size(1), 1).repeat(1, 1, dense_pos_scores.size(1)).view(-1, neg_mask.size(1) * pos_mask.size(1))

        pos_mask_ = pos_mask.repeat(1, neg_mask.size(1))
        neg_mask_ = neg_mask.view(-1, neg_mask.size(1), 1).repeat(1, 1, pos_mask.size(1)).view(-1, neg_mask.size(1) * pos_mask.size(1))
        flattened_mask = (pos_mask_ * neg_mask_).view(-1).long()
        valid_pos_scores_ = dense_pos_scores_.view(-1)[flattened_mask > 0]
        valid_neg_scores_ = dense_neg_scores_.view(-1)[flattened_mask > 0]
        y = torch.ones_like(valid_pos_scores_).to(self.device).float()
        loss = self.ranking_loss(valid_pos_scores_, valid_neg_scores_, y)
        return loss


class TripletHard():
    def __init__(self):
        print('Triplet Hard loss')
        margin = 1
        self.ranking_loss = torch.nn.MarginRankingLoss(margin=margin)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def __call__(self, scores, labels, batch_vec):
        """
        * the three input tensors have shape (N, ), N being the number of nodes in the batch
        * what makes possible to split values by query (i.e. graph) is the batch_vec vector, indicating which node
        belongs to which graph
        we want to compute all the pairwise contributions in the batch, dealing with:
        1. not mixing between graphs
        2. variable number of valid pairs between graphs (using masking)
        """
        ids_pos = labels == 1
        ids_neg = labels == 0
        batch_vec_pos = batch_vec[ids_pos]
        batch_vec_neg = batch_vec[ids_neg]
        pos_scores = scores[ids_pos]
        neg_scores = scores[ids_neg]
        # print('1', ids_pos.size(), ids_neg.size(), batch_vec_pos.size(), batch_vec_neg.size(), pos_scores.size(), neg_scores.size())
        # print('2', pos_scores.size(), set(batch_vec_pos.tolist()))
        # print('3', neg_scores.size(), set(batch_vec_neg.tolist()))
        # densify the tensors (see: https://rusty1s.github.io/pytorch_geometric/build/html/modules/utils.html?highlight=to_dense#torch_geometric.utils.to_dense_batch)
        dense_pos_scores, pos_mask = to_dense_batch(pos_scores, batch_vec_pos, fill_value=0)
        # dense_pos_scores has shape (nb_graphs, padding => max number nodes for graphs in batch)
        pos_len = torch.sum(pos_mask, dim=-1)  # shape (nb_graphs, ), actual number of nodes per graph
        dense_neg_scores, neg_mask = to_dense_batch(neg_scores, batch_vec_neg, fill_value=0)
        neg_len = torch.sum(neg_mask, dim=-1)
        max_pos_len = pos_len.max()  # == the padding value for the positive scores
        max_neg_len = neg_len.max()

        # print('4', pos_scores.size(), neg_scores.size())
        # print('5', dense_pos_scores.size(), pos_mask.size(), pos_len.size(), max_pos_len.item())
        # print('6', dense_neg_scores.size(), neg_mask.size(), neg_len.size(), max_neg_len.item())
        # print('7', pos_mask)
        # print('8', neg_mask)
        # print('---')

        # pos_mask_old, neg_mask_old = pos_mask, neg_mask
        pos_mask = masking(pos_len, max_pos_len.item())
        neg_mask = masking(neg_len, max_neg_len.item())

        # print('9', pos_mask.size(), neg_mask.size())
        # print('10', pos_mask, pos_mask==pos_mask_old)
        # print('11', neg_mask, neg_mask==neg_mask_old)
        # print('\n')

        # assert dense_pos_scores.size(0) == 16, (batch_vec_pos.tolist(), )
        # print(dense_pos_scores.size(), dense_neg_scores.size())
        # print(dense_pos_scores.view(-1, 1, dense_pos_scores.size(1)).size(), dense_neg_scores.view(-1, dense_neg_scores.size(1), 1).size())
        dense_pos_scores_ = dense_pos_scores.repeat(1, neg_mask.size(1))
        dense_neg_scores_ = dense_neg_scores.view(-1, dense_neg_scores.size(1), 1).repeat(1, 1, dense_pos_scores.size(1)).view(-1, neg_mask.size(1) * pos_mask.size(1))

        pos_mask_ = pos_mask.repeat(1, neg_mask.size(1))
        neg_mask_ = neg_mask.view(-1, neg_mask.size(1), 1).repeat(1, 1, pos_mask.size(1)).view(-1, neg_mask.size(1) * pos_mask.size(1))

        scores_ap, scores_an = [], []
        for i in range(n):
            scores_ap.append()
        flattened_mask = (pos_mask_ * neg_mask_).view(-1).long()
        valid_pos_scores_ = dense_pos_scores_.view(-1)[flattened_mask > 0]
        valid_neg_scores_ = dense_neg_scores_.view(-1)[flattened_mask > 0]
        y = torch.ones_like(valid_pos_scores_).to(self.device).float()
        loss = self.ranking_loss(valid_pos_scores_, valid_neg_scores_, y)
        return loss