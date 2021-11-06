# import torch.nn.functional as F
import torch.nn as nn
# import torch
class CustomLoss1(nn.Module):
    def __init__(self):
        super(CustomLoss1, self).__init__()
    def __name__(self):
      return "CUSTOM1"

    def forward(self, inputs, targets, smooth=1):

        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        CEloss = nn.CrossEntropyLoss()
        ce_output = CEloss(inputs, targets)
        ce_output.backwared()

        KLloss = nn.CrossEntropyLoss()
        kl_output = KLloss(inputs, targets)
        kl_output.backwared()

        return ce_output+kl_output

