import torch


def con_model_cross_entropy(pred, target, weight, labels, regular):
    # other
    pred = pred[:, 1]
    # ...
    if pred.min() == 0 or pred.max() == 1:
        pred = torch.clamp(pred, 0.0001, 0.9999)
    if regular.min() == 0 or regular.max() == 1:
        regular = torch.clamp(regular, 0.0001, 0.9999)
    k = 100
    loss = (-weight * (target * torch.log(pred) + (1 - target) * torch.log(1 - pred))).sum() / weight.sum() + \
           (- k * (labels * torch.log(regular) + (1 - labels) * torch.log(1 - regular))).sum() / len(labels)
    return loss


def base_model_cross_entropy(pred, target, weight):
    # other
    pred = pred[:, 1]
    # ...
    if pred.min() == 0 or pred.max() == 1:
        pred = torch.clamp(pred, 0.0001, 0.9999)
    loss = (-weight * (target * torch.log(pred) + (1 - target) * torch.log(1 - pred))).sum() / weight.sum()
    return loss


def denoise_model_cross_entropy(regular, labels):
    loss = - (labels * torch.log(regular) + (1 - labels) * torch.log(1 - regular)).sum() / len(labels)
    return loss


def l2_model_cross_entropy(pred, target, weight):
    pred_0 = pred[:, 1]
    if pred_0.min() == 0 or pred_0.max() == 1:
        pred_0 = torch.clamp(pred_0, 0.0001, 0.9999)
    loss = (-weight * (target * torch.log(pred_0) + (1 - target) * torch.log(1 - pred_0))).sum() / weight.sum() - \
           0.04 * (weight ** 2).mean()
    return loss
