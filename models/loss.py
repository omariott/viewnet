import torch
import torch.nn as nn
import torchvision.models
import numpy as np


class PerceptualLoss(nn.Module):
    def __init__(self, losstype='mse', reduction="mean"):
        super(PerceptualLoss, self).__init__()

        def double_loss(output, target):
            return .5 * (nn.functional.mse_loss(output, target) + nn.functional.l1_loss(output, target))
        def cosine_loss(output, target):
            bsize = output.size(0)
            return 1 - nn.functional.cosine_similarity(output.view(bsize, -1), target.view(bsize, -1))

        self.net = torchvision.models.vgg16(pretrained=True).features.eval()
#        print(self.net)
#        exit()
        """
        vgg11 : [2,5,10,15,20]
        vgg11_bn : [3,7,14,21,28]
        vgg16 : [3,8,15,22,29]
        vgg16_bn : [5,12,22,32,42]
        vgg19 : [3,8,17,26,35]
        """
        self.layers_id = [0, 3, 8, 15]#, 22, 29]
        self.weights = [10, 1, 1, 1,]# 1, 1]
        if losstype=='l1':
            self.loss = nn.L1Loss(reduction=reduction)
        elif losstype=='mse':
            self.loss = nn.MSELoss(reduction=reduction)
        elif losstype=='both':
            self.loss = double_loss
        elif losstype=='cosine':
            self.loss = cosine_loss
        else:
            print("error, I don't know the loss ", losstype)
            exit()
        self.intrinsic_weight = np.array(self.weights).sum()

        self.avg = torch.tensor([[0.485, 0.456, 0.406]]).unsqueeze(-1).unsqueeze(-1)
        self.var =  torch.tensor([[0.229, 0.224, 0.225]]).unsqueeze(-1).unsqueeze(-1)

    def cuda(self):
        self.avg = self.avg.cuda()
        self.var = self.var.cuda()
        self.net = self.net.cuda()
        return self

    def forward(self, input_data, target):
        x = (input_data - self.avg) / self.var
        tar = (target - self.avg) / self.var
        losses = 0
        weight_id = 0
        for id, module in enumerate(self.net):
            if id in self.layers_id:
                lossval = self.loss(x.clone(), tar.clone())
                losses += self.weights[weight_id] * lossval
                weight_id += 1
            if id == self.layers_id[-1]:
                break
            x = module(x)
            tar = module(tar)
        return losses/self.intrinsic_weight
