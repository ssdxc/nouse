import torch


class BCELoss():
    def __init__(self):
        self.loss = torch.nn.BCEWithLogitsLoss(reduction="mean")
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    def __call__(self, scores, labels, batch_vec):
        scores, labels = scores.view(-1,), labels.view(-1,)
        return self.loss(scores, labels)


class BBCELoss():
    def __init__(self):
        self.loss_pos = torch.nn.BCEWithLogitsLoss(reduction="mean")
        self.loss_neg = torch.nn.BCEWithLogitsLoss(reduction="mean")
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    def __call__(self, scores, labels, batch_vec):
        ids_pos = labels == 1
        ids_neg = labels == 0
        pos_scores = scores[ids_pos].view(-1,)
        neg_scores = scores[ids_neg].view(-1,)
        pos_labels = labels[ids_pos].view(-1,)
        neg_labels = labels[ids_neg].view(-1,)
        return self.loss_pos(pos_scores, pos_labels) + self.loss_neg(neg_scores, neg_labels)