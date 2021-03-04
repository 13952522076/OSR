import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ArcFaceLoss(nn.Module):
    def __init__(self,scaling=32, m=0.5):
        super(ArcFaceLoss, self).__init__()
        self.scaling = scaling
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        # ensure cos(theta+m) decreases in the range of (0,pi)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, net_out, targets):
        cosine = net_out["cosine_fea2cen"]
        cosine = cosine.clamp(-1, 1)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, targets.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.scaling
        loss = F.cross_entropy(output, targets)

        return {
            "total": loss
        }



def demo():
    n = 3
    c = 5
    sim_fea2cen = torch.rand([n, c])

    label = torch.empty(3, dtype=torch.long).random_(c)
    print(label)
    loss = ArcFaceLoss(1.)
    netout = {
        "dotproduct_fea2cen": sim_fea2cen,
        "cosine_fea2cen": sim_fea2cen
    }
    dist_loss = loss(netout, label)
    print(dist_loss)


demo()
