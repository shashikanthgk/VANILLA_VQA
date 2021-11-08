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


        print(targets,inputs)
        print(targets.shape,inputs.shape)
        y_onehot = torch.FloatTensor(inputs.shape[0], inputs.shape[1])
        # In your for loop
        y_onehot.zero_()
        y_onehot.scatter_(1, targets, 1)
        print(y_onehot.shape)
        CEloss = F.cross_entropy(inputs,targets)
        # KLloss = F.kl_div(torch.transpose(inputs, 0, 1),targets,log_target=True)
        return CEloss

