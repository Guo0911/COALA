import torch

import torch.nn.functional as F

def get_loss(
    name:str,
):
    if name == "MPD-loss":
        return MPD_loss
    if name == "DPD-loss":
        return DPD_loss
    else:
        raise NotImplementedError

def MPD_loss(pos_socres, neg_socres):
    loss = 0

    neg_lse = torch.logsumexp(neg_socres, dim=0) # Log(sum(exp(neg_socres)))
    for pos_socre in pos_socres:
        poneg_socres_lse = torch.logsumexp(torch.stack([pos_socre, neg_lse]), dim=0)
        loss += -(pos_socre - poneg_socres_lse) # -log(exp(pos_socre) / (exp(pos_socre) + sum(exp(neg_socres))))

    loss = loss / len(pos_socres)

    return loss, 0.0, 0.0

def DPD_loss(pos_socres, neg_socres):
    pos_loss = F.binary_cross_entropy_with_logits(pos_socres, torch.ones_like(pos_socres))
    neg_loss = F.binary_cross_entropy_with_logits(neg_socres, torch.zeros_like(neg_socres))

    loss = pos_loss + neg_loss

    return loss, pos_loss, neg_loss
