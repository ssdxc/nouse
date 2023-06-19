import torch

from .pairwise import BPRLoss
from .bce import BCELoss, BBCELoss

class BCEBPRLoss():
    def __init__(self):
        self.loss_bce = BCELoss()
        self.loss_bpr = BPRLoss()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    def __call__(self, scores, labels, batch_vec):
        return self.loss_bce(scores, labels, batch_vec) + self.loss_bpr(scores, labels, batch_vec)


class BBCEBPRLoss():
    def __init__(self):
        self.loss_bce = BBCELoss()
        self.loss_bpr = BPRLoss()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    def __call__(self, scores, labels, batch_vec):
        return self.loss_bce(scores, labels, batch_vec) + self.loss_bpr(scores, labels, batch_vec)


class WBBCEBPRLoss():
    def __init__(self):
        self.loss_bce = BBCELoss()
        self.loss_bpr = BPRLoss()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    def __call__(self, scores, labels, batch_vec):
        # return 0.01*self.loss_bce(scores, labels, batch_vec) + 0.99*self.loss_bpr(scores, labels, batch_vec)
        return self.loss_bpr(scores, labels, batch_vec)