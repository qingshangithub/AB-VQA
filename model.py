# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
# from lib.non_local_concatenation import NONLocalBlock2D
from lib.non_local_gaussian import NONLocalBlock2D
#from lib.non_local_embedded_gaussian import NONLocalBlock2D
# from lib.non_local_dot_product import NONLocalBlock2D

def TP(q, tau=12, beta=0.5):
    """subjectively-inspired temporal pooling"""
    q = torch.unsqueeze(torch.t(q), 0)
    qm = -float('inf')*torch.ones((1, 1, tau-1)).to(q.device)
    qp = 10000.0 * torch.ones((1, 1, tau - 1)).to(q.device)  #
    l = -F.max_pool1d(torch.cat((qm, -q), 2), tau, stride=1)
    m = F.avg_pool1d(torch.cat((q * torch.exp(-q), qp * torch.exp(-qp)), 2), tau, stride=1)
    n = F.avg_pool1d(torch.cat((torch.exp(-q), torch.exp(-qp)), 2), tau, stride=1)
    m = m / n
    return beta * m + (1 - beta) * l

class NIMA(nn.Module):
    """Neural IMage Assessment model by Google"""
    def __init__(self, base_model,input_channels):
        super(NIMA, self).__init__()
        #self.features = base_model.features
        self.vgg_non_model = nn.Sequential()
        i = 0
        endlayer = 30
        use_maxpool = True
        for layer in list(base_model.features):
            if isinstance(layer, nn.Conv2d):
                name = "conv_" + str(i)
                name_non = "non_local"+ str(i)
                if i>15 and i<22:
                    self.vgg_non_model.add_module(name_non,NONLocalBlock2D(in_channels=layer.in_channels))
                self.vgg_non_model.add_module(name, layer)

            if isinstance(layer, nn.ReLU):
                name = "relu_" + str(i)
                layer = nn.ReLU(inplace=True)
                self.vgg_non_model.add_module(name, layer)

            if isinstance(layer, nn.MaxPool2d):
                name = "pool_" + str(i)
                self.vgg_non_model.add_module(name, layer)
            i += 1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.avgpool = nn.AvgPool2d((14, 14))i
        self.rnn = nn.GRU(512, 32, batch_first=True)
        self.q = nn.Linear(32, 1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=1))

    def forward(self, x):
        out = self.vgg_non_model(x)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = out.unsqueeze(0)
        output,_ = self.rnn(out, self._get_initial_state(out.size(0), out.device))
        output = self.q(output)
        qi = output[0, :]
        qi = TP(qi)
        score = torch.mean(qi)  # video overall quality
            
        return score

    def _get_initial_state(self, batch_size, device):
        h0 = torch.zeros(1, batch_size, 32, device=device)
        return h0

def regloss_v1(x, y):
    loss = torch.nn.MSELoss(reduction='mean')
    return loss(x.float(), y.float())

