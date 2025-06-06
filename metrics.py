from sklearn.metrics import roc_auc_score
import torch
import heapq


class Metric:
    def __init__(self, scores, labels):
        self.scores = scores
        self.labels = labels

    def auc(self):
        return roc_auc_score(self.labels, self.scores)


def ndcg_n(pred, labels, top_n=20):
    auc_pre = []
    for scores in pred:
        auc_pre.append(scores[-1])
    scores = torch.tensor(auc_pre)
    labels = torch.tensor(labels)
    rank = (-scores).argsort(dim=0)
    rank = rank[:top_n]
    hits = labels.gather(0, rank)
    position = torch.arange(2, 2 + top_n)
    weights = 1 / torch.log2(position + 1)
    dcg = (hits * weights).sum()
    idcg = weights.sum()
    return dcg / idcg


# 预测正确，概率在0.5以上
def hr_n(pred, labels, top_n=20):
    auc_pre = []
    right = 0
    for scores in pred:
        auc_pre.append(scores[-1])
    max_n = heapq.nlargest(top_n, auc_pre)
    for t in max_n:
        index = auc_pre.index(t)
        if labels[index] == 1 and t > 0.5:
            right += 1
    return right / labels.count(1)


def mrr_n(pred, labels, top_n=20):
    auc_pre = []
    sum, i = 0, 1
    for scores in pred:
        auc_pre.append(scores[-1])
    max_n = heapq.nlargest(top_n, auc_pre)
    for t in max_n:
        index = auc_pre.index(t)
        if labels[index] == 1 and t > 0.5 or labels[index] == 0 and t < 0.5:
            sum += 1/i
            i += 1
    return sum


def precision_n(pred, labels, top_n=20):
    auc_pre = []
    right = 0
    for scores in pred:
        auc_pre.append(scores[-1])
    max_n = heapq.nlargest(top_n, auc_pre)
    for t in max_n:
        index = auc_pre.index(t)
        if labels[index] == 1:
            right += 1
    return right / top_n


# 召回正确，在召回列表中就算，不用看概率是否在0.5以上
def recall_n(pred, labels, top_n=20):
    auc_pre = []
    right = 0
    for scores in pred:
        auc_pre.append(scores[-1])
    max_n = heapq.nlargest(top_n, auc_pre)
    for t in max_n:
        index = auc_pre.index(t)
        if labels[index] == 1:
            right += 1
    return right / labels.count(1)


def acc_n(pred, labels, top_n=20):
    auc_pre = []
    sum = 0
    for scores in pred:
        auc_pre.append(scores[-1])
    max_n = heapq.nlargest(top_n, auc_pre)
    for t in max_n:
        index = auc_pre.index(t)
        if labels[index] == 1 and t > 0.5 or labels[index] == 0 and t < 0.5:
            sum += 1
    return sum / top_n
