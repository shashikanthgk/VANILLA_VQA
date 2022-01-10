import torch.nn.functional as F
import torch.nn as nn
import torch
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
    def __name__(self):
      return "CUSTOM1"

    def forward(self, inputs, targets, smooth=1):
        
        # inputs = inputs.view(-1)
        # targets = targets.view(-1)
        CEloss = F.cross_entropy(inputs,targets)
        # KLloss = F.kl_div(torch.transpose(inputs, 0, 1),targets,log_target=True)
        return CEloss

