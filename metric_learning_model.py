import torch
import torch.nn as nn
from torchvision import models


class MetricLearningNet(nn.Module):
    def __init__(self, pretrained=True):
        super(MetricLearningNet, self).__init__()
        self.pretrained = pretrained
        self.embedder = models.resnet50(pretrained=pretrained)

    def forward(self, ref, pos, neg):
        ref_emb = self.embedder(ref)
        pos_emb = self.embedder(pos)
        neg_emb = self.embedder(neg)

        return ref_emb, pos_emb, neg_emb
