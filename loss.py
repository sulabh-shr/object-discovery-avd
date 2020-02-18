import torch
import torch.nn as nn
import torch.nn.functional as f

def triplet_loss(ref, pos, neg, min_dist_neg=1):
    """

    Parameters
    ----------
    ref: N,D
    pos: N,D
    neg: N,D
    min_dist_neg

    Returns
    -------

    """
    func_dist = nn.PairwiseDistance(p=2)
    ref = f.normalize(ref, p=2)  # default dim=1, so works for both 2d or 1d tensor
    pos = f.normalize(pos, p=2)
    neg = f.normalize(neg, p=2)

    pos_distance = func_dist(ref, pos)
    neg_distance = func_dist(ref, neg)

    distance = pos_distance - neg_distance + min_dist_neg
    loss = torch.mean(torch.max(distance, torch.zeros_like(distance)))

    return loss
