import torch
import torch.nn as nn
from torchvision import models


class MetricLearningNet(nn.Module):
    def __init__(self, model='resnet18', pretrained=True):

        super(MetricLearningNet, self).__init__()
        self.model = model
        self.pretrained = pretrained

        if model == 'resnet18':
            self.embedder = models.resnet18(pretrained=self.pretrained)
        elif model == 'resnet50':
            self.embedder = models.resnet50(pretrained=self.pretrained)
        else:
            raise NotImplementedError(f'Model {self.model} not found!!!')

    def forward(self, ref, pos, neg, concat=True):

        if concat:
            batch_size = len(ref)
            triplets = torch.cat((ref, pos, neg), dim=0)
            emb = self.embedder(triplets)

            return emb[:batch_size], emb[batch_size:2*batch_size], emb[2*batch_size:]
        else:
            ref_emb = self.embedder(ref)
            pos_emb = self.embedder(pos)
            neg_emb = self.embedder(neg)

            return ref_emb, pos_emb, neg_emb
