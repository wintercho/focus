import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import numpy as np

class SpreadLoss(_Loss):

    def __init__(self, m_min=0.2, m_max=0.9, num_class=10):
        super(SpreadLoss, self).__init__()
        self.m_min = m_min
        self.m_max = m_max
        self.num_class = num_class

    def forward(self, x, target, r):
        b, E = x.shape
        assert E == self.num_class
        margin = self.m_min + (self.m_max - self.m_min)*r

        at = torch.cuda.FloatTensor(b).fill_(0)
        for i, lb in enumerate(target):
            at[i] = x[i][lb]
        at = at.view(b, 1).repeat(1, E)

        zeros = x.new_zeros(x.shape)
        loss = torch.max(margin - (at - x), zeros)
        loss = loss**2
        loss = loss.sum() / b - margin**2

        return loss



def one_hot(label):    #pytorch had
    onehot = torch.zeros(label.shape[0],2).cuda()
    onehot = onehot.scatter_(1, label, 1)
    return onehot



class Reconstructloss(nn.Module):   #dense
    def __init__(self):
        super(Reconstructloss,self).__init__()
        self.fc = nn.Sequential(
                nn.Linear(32,7*7*32),
                nn.BatchNorm1d(1568),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                )
        self.deconv1_1 = nn.Sequential(
                nn.ConvTranspose2d(32, 32, 2, stride=2, bias=False),
                nn.ReLU(inplace=True),
                )
        self.deconv1_2 = nn.Sequential(
                nn.ConvTranspose2d(32, 32, 4, stride=4, bias=False),
                nn.ReLU(inplace=True),
                )
        self.deconv1_3 = nn.Sequential(
                nn.ConvTranspose2d(32, 32, 8, stride=8, bias=False),
                nn.ReLU(inplace=True),
                )
        self.deconv1_4 = nn.Sequential(
                nn.ConvTranspose2d(32, 32, 16, stride=16, bias=False),
                nn.ReLU(inplace=True),
                )
        self.deconv1_5 = nn.Sequential(
                nn.ConvTranspose2d(32, 32, 32, stride=32, bias=False),
                nn.ReLU(inplace=True),
                )
        self.deconv2_1 = nn.Sequential(
                nn.ConvTranspose2d(32, 32, 2, stride=2, bias=False),
                nn.ReLU(inplace=True),
                )
        self.deconv2_2 = nn.Sequential(
                nn.ConvTranspose2d(32, 32, 4, stride=4, bias=False),
                nn.ReLU(inplace=True),
                )
        self.deconv2_3 = nn.Sequential(
                nn.ConvTranspose2d(32, 32, 8, stride=8, bias=False),
                nn.ReLU(inplace=True),
                )
        self.deconv2_4 = nn.Sequential(
                nn.ConvTranspose2d(32, 32, 16, stride=16, bias=False),
                nn.ReLU(inplace=True),
                )
        self.deconv3_1 = nn.Sequential(
                nn.ConvTranspose2d(32, 32, 2, stride=2, bias=False),
                nn.ReLU(inplace=True),
                )
        self.deconv3_2 = nn.Sequential(
                nn.ConvTranspose2d(32, 32, 4, stride=4, bias=False),
                nn.ReLU(inplace=True),
                )
        self.deconv3_3 = nn.Sequential(
                nn.ConvTranspose2d(32, 32, 8, stride=8, bias=False),
                nn.ReLU(inplace=True),
                )
        self.deconv4_1 = nn.Sequential(
                nn.ConvTranspose2d(32, 32, 2, stride=2, bias=False),
                nn.ReLU(inplace=True),
                )
        self.deconv4_2 = nn.Sequential(
                nn.ConvTranspose2d(32, 32, 4, stride=4, bias=False),
                nn.ReLU(inplace=True),
                )
        self.deconv5_1 = nn.Sequential(
                nn.ConvTranspose2d(32, 32, 2, stride=2, bias=False),
                nn.ReLU(inplace=True),
                )
        self.conv1 = nn.Sequential(
                nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                )
        self.conv2 = nn.Sequential(
                nn.Conv2d(16, 3, kernel_size=1, stride=1),
                nn.BatchNorm2d(3),
                nn.Sigmoid(),
                )
        self.squareloss = nn.MSELoss()
    def forward(self, out, x, y):
        y_onehot = one_hot(torch.unsqueeze(y,1))
        y_onehot = torch.unsqueeze(y_onehot, 2)
        out = torch.squeeze(out,1)
        out = out.mul(y_onehot).view(out.shape[0], -1)
        out = self.fc(out)
        out = out.view(-1, 32, 7, 7)
        out1_1 = self.deconv1_1(out)
        out1_2 = self.deconv1_2(out)
        out1_3 = self.deconv1_3(out)
        out1_4 = self.deconv1_4(out)
        out1_5 = self.deconv1_5(out)
        out2 = out1_1 #14
        out2_1 = self.deconv2_1(out2)
        out2_2 = self.deconv2_2(out2)
        out2_3 = self.deconv2_3(out2)
        out2_4 = self.deconv2_4(out2)
        out3 = out1_2 + out2_1 # 28
        out3_1 = self.deconv3_1(out3)
        out3_2 = self.deconv3_2(out3)
        out3_3 = self.deconv3_3(out3)
        out4 = out1_3 + out2_2 + out3_1 #56
        out4_1 = self.deconv4_1(out4)
        out4_2 = self.deconv4_2(out4)
        out5 = out1_4 + out2_3 + out3_2 + out4_1 #112
        out5_1 = self.deconv5_1(out5)
        outp = out1_5 + out2_4 + out3_3 + out4_2 + out5_1 #224
        outp = self.conv1(outp)
        outp = self.conv2(outp)
        
        loss = self.squareloss(outp, x)
        return loss



class Reconstructloss_fc(nn.Module):  #fc
    def __init__(self):
        super(Reconstructloss_fc,self).__init__()
        self.fc1 = nn.Sequential(
                nn.Linear(32,512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                )
        self.fc2 = nn.Sequential(
                nn.Linear(512,1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                )
        self.fc3 = nn.Sequential(
                nn.Linear(1024,224*224*3),
                nn.BatchNorm1d(224*224*3),
                nn.Sigmoid(),
                )
        self.squareloss = nn.MSELoss()

    def forward(self, out, x, y):
        y_onehot = one_hot(torch.unsqueeze(y,1))
        y_onehot = torch.unsqueeze(y_onehot, 2)
        out = torch.squeeze(out,1)
        out = out.mul(y_onehot).view(out.shape[0], -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = out.view(-1, 3, 224, 224)
        loss = self.squareloss(out, x)

        return loss


class TripletLoss(_Loss):

    def __init__(self, m_min=0.2, m_max=0.9):
        super(TripletLoss, self).__init__()
        self.m_min = m_min
        self.m_max = m_max
        self.loss_func = nn.TripletMarginLoss(margin=0.2, p=2)

    def forward(self, anchor, pos, neg, r):
        # margin = self.m_min + (self.m_max - self.m_min)*r
        # loss_func = nn.TripletMarginLoss(margin=margin, p=2)
        return torch.mean(self.loss_func(anchor,pos,neg))


def triple_loss(anchor,pos,neg,m):
    loss_func = nn.TripletMarginLoss(margin=m, p=2)
    return torch.mean(loss_func(anchor,pos,neg))


def softmax_loss_1d(x,y,weight=None):
    loss_mat = V(torch.zeros(x.size(0))).cuda()
    x = torch.exp(x)
    sums = torch.sum(x,1).view(-1,1)
    probility = x/sums 
    for i in range(x.size(0)):
        a = torch.mm(y[i,:].view(-1,1), torch.log(probility[i,:].view(-1,1).t()))
        b = torch.tanh(a)
        c = b*V(weight)
        #print(c)
        loss = torch.sum(c)
        
        loss_mat[i] = loss
    return torch.mean(loss_mat)
        


class CenterLoss(torch.nn.Module):
    def __init__(self, num_classes, feat_dim, loss_weight=1.0):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.loss_weight = loss_weight
        self.centers = torch.nn.Parameter(torch.randn(num_classes, feat_dim))
        self.use_cuda = False


    def forward(self, y, feat):
        if self.use_cuda:
            hist = V(torch.histc(y.cpu().data.float(), bins=self.num_classes, min=0, max=self.num_classes) + 1).cuda()
        else:
            hist = V(torch.histc(y.data.float(), bins=self.num_classes, min=0, max=self.num_classes) + 1)


        centers_count = hist.index_select(0, y.long())  # 计算每个类别对应的数目


        batch_size = feat.size()[0]
        feat = feat.view(batch_size, 1, 1, -1).squeeze()
        if feat.size()[1] != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's dim: {1}".format(self.feat_dim,
                                                                                                    feat.size()[1]))
        centers_pred = self.centers.index_select(0, y.long())
        diff = feat-centers_pred
        loss = self.loss_weight * 1/2.0 * (diff.pow(2).sum(1) / centers_count).sum()
        return loss
    
    def cuda(self, device_id=None):
        self.use_cuda = True
        return self._apply(lambda t: t.cuda(device_id))





class PairwiseDistance(_Loss):
    def __init__(self, p):
        super(PairwiseDistance, self).__init__()
        self.norm = p

    def forward(self, x1, x2):
        assert x1.size() == x2.size()
        eps = 1e-4 / x1.size(1)
        diff = torch.abs(x1 - x2)
        out = torch.pow(diff, self.norm).sum(dim=1)
        return torch.pow(out + eps, 1. / self.norm)


class TripletMarginLoss(_Loss):
    """Triplet loss function.
    """
    def __init__(self, margin):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin
        self.pdist = PairwiseDistance(2)  # norm 2

    def forward(self, anchor, positive, negative):
        d_p = self.pdist.forward(anchor, positive)
        d_n = self.pdist.forward(anchor, negative)

        dist_hinge = torch.clamp(self.margin + d_p - d_n, min=0.0)
        loss = torch.mean(dist_hinge)
        return loss

if __name__ == '__main__':
    model = Reconstructloss()
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('model parameters:', params)
    


