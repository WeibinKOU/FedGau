import torch
import torch.nn as nn
import torch.nn.functional as F

class MOONLoss(nn.Module):
    def __init__(self, mu=0.001, tau=3, ign_idx=255):
        super(MOONLoss, self).__init__()
        self.mu = mu
        self.tau = tau
        self.ce = nn.CrossEntropyLoss(ignore_index=ign_idx)
        self.sim = nn.CosineSimilarity(dim=-1)
        self.con = nn.CrossEntropyLoss()

    def forward(self, logits, targets, z, z_prev, z_g):
        device = targets.device
        #loss1 = self.ce(logits, targets)

        pred_masks = [F.softmax(logit, dim=1) for logit in logits]
        dist = [self.ce(pred_mask.to(device), targets) for pred_mask in pred_masks]
        loss1 = sum(dist)

        positive = self.sim(z.reshape(z.size(0), -1), z_g.reshape(z_g.size(0), -1)).reshape(-1, 1)
        negative = self.sim(z.reshape(z.size(0), -1), z_prev.reshape(z_prev.size(0), -1)).reshape(-1, 1)
        moon_logits = torch.cat([positive, negative], dim=1)
        moon_logits /= self.tau
        moon_labels = torch.zeros(z.size(0)).to(device).long()

        loss2 = self.con(moon_logits, moon_labels)

        total_loss = loss1 + self.mu * loss2

        return total_loss
