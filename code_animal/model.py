# -*- coding: utf-8 -*-

from __future__ import division
import torch.nn as nn
from collections import OrderedDict
import torch
import torch.nn.functional as F
from torch.nn import init


class ResidualBlock(nn.Module):
    def __init__(self,inchannel,midchannel,outchannel,stride):
        super(ResidualBlock,self).__init__()
        self.outchannel = outchannel
        self.left = nn.Sequential(
                nn.Conv2d(inchannel,midchannel,kernel_size=1,stride=1),
                nn.BatchNorm2d(midchannel),
                nn.ReLU(inplace=True),
                nn.Conv2d(midchannel,midchannel,kernel_size=3,stride=stride,padding=1),
                nn.BatchNorm2d(midchannel),
                nn.ReLU(inplace=True),
                nn.Conv2d(midchannel,outchannel,kernel_size=1,stride=1),
                nn.BatchNorm2d(outchannel)
                )
        self.right = nn.Sequential(
                nn.Conv2d(inchannel,outchannel,kernel_size=1,stride=stride),
                nn.BatchNorm2d(outchannel)
                )
        self.globalpool = nn.AdaptiveAvgPool2d(output_size=1)
        
        self.fc = nn.Sequential(
                nn.Linear(outchannel,outchannel//16),
                nn.BatchNorm1d(outchannel//16),
                nn.ReLU(inplace=True),
                nn.Linear(outchannel//16,outchannel),
                nn.Sigmoid()
                )

        self.activate = nn.ReLU(inplace=True)
        
    def forward(self,x):
        left = self.left(x)
        weight = self.fc(self.globalpool(left).view(-1,self.outchannel)).view(-1,self.outchannel,1,1)
        right = self.right(x)
        out = self.activate(left*weight+right)
        return out

class MyResNet(nn.Module):
    def __init__(self):
        super(MyResNet,self).__init__() #256
        self.prelayer = nn.Sequential(
                nn.Conv2d(3,64,kernel_size=11,stride=4,padding=5), #64
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3,stride=2,padding=1) #32
                )
        self.layer1 = ResidualBlock(64,32,128,1)  #32
        self.layer2 = ResidualBlock(128,64,128,1)    

        self.layer3 = ResidualBlock(128,64,256,2) #16 
        self.layer4 = ResidualBlock(256,128,256,1)
        
        self.layer5 = ResidualBlock(256,128,512,2) #8
        self.layer6 = ResidualBlock(512,256,512,1)
        
        self.layer7= ResidualBlock(512,256,1024,2) #4
        self.layer8= ResidualBlock(1024,512,1024,1)
               
        self.globalpool = nn.MaxPool2d(kernel_size=2,stride=2) #2
        
        self.fc = nn.Sequential(
                nn.Linear(4096,512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512,2)
                )
        
    def forward(self,x):
        x = self.prelayer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        feature = self.globalpool(x).view(-1,4096)
        result = self.fc(feature)
        return result



class SEBlock(nn.Module):
    def __init__(self,inchannel,outchannel):
        super(SEBlock,self).__init__()
        self.outchannel = outchannel
        self.left = nn.Conv2d(inchannel,outchannel,kernel_size=3,padding=1)
        self.globalpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.right = nn.Sequential(
                nn.Linear(outchannel,outchannel//64),
                nn.BatchNorm1d(outchannel//64),
                nn.ReLU(inplace=True),
                nn.Linear(outchannel//64,outchannel),
                nn.Sigmoid()
                )
        self.bn = nn.BatchNorm2d(outchannel)
        self.activate = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)   
    def forward(self,x):
        left = self.left(x)
        right = self.globalpool(left).view(-1,self.outchannel,1,1)
        return self.pool(self.activate(self.bn(left*right)))
        
        
        
class MyAlexNet(nn.Module):
    def __init__(self):
        super(MyAlexNet,self).__init__()
        self.prelayer = nn.Sequential(
                nn.Conv2d(3,64,kernel_size=11,stride=4,padding=5), #64
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3,stride=2,padding=1) #32
                )
        self.layer1 = nn.Sequential(
                nn.Conv2d(64,192,kernel_size=5,padding=2), #32
                nn.BatchNorm2d(192),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3,stride=2,padding=1) #16
                )
        self.layer2 = SEBlock(192,384) #8
        self.layer3 = SEBlock(384,512) #4
        self.fc = nn.Sequential(
                nn.Linear(8192,512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512,6)
                )
    def forward(self,x):
        x = self.prelayer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feature = x.view(-1,8192)
        out = self.fc(feature)
        return feature, out
    
class Alexnet(nn.Module):
    def __init__(self):
        super(Alexnet,self).__init__()  #230
        self.features = nn.Sequential(
                nn.Conv2d(3,64,11,stride=4,padding=2),  #56
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3,stride=2,padding=1), #28
                nn.Conv2d(64,192,kernel_size=5,stride=1,padding=2),#28
                nn.BatchNorm2d(192),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3,stride=2,padding=1),#14
                nn.Conv2d(192,384,kernel_size=3,padding=1),
                nn.BatchNorm2d(384),
                nn.ReLU(inplace=True),
                nn.Conv2d(384,256,kernel_size=3,padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256,256,kernel_size=3,padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3,stride=2) #6
                )
        self.fc = nn.Sequential(
                nn.Linear(9216,4096),
                nn.BatchNorm1d(4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096,4096),
                nn.BatchNorm1d(4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096,6)
                )
    def forward(self,x):
        features = self.features(x).view(-1,9216)
        out = self.fc(features)
        return features, out
        
        
                
class VGG(nn.Module):

    def __init__(self, features, num_classes=2, init_weights=True):

        super(VGG, self).__init__()

        self.features = features

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(512, num_classes)

        # self.classifier1 = nn.Sequential(
            # nn.Linear(512 * 7 * 7, 4096),
            # nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(4096, 4096),
            # nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(4096, num_classes),
        # )

        if init_weights:

            self._initialize_weights()



    def forward(self, x):

        x = self.features(x)
        fea = x

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        # x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return fea, x
        # return x



    def _initialize_weights(self):

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:

                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):

                nn.init.constant_(m.weight, 1)

                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):

                nn.init.normal_(m.weight, 0, 0.01)

                nn.init.constant_(m.bias, 0)





def make_layers(cfg, batch_norm=False):

    layers = []

    in_channels = 3

    for v in cfg:

        if v == 'M':

            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        else:

            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)

            if batch_norm:

                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]

            else:

                layers += [conv2d, nn.ReLU(inplace=True)]

            in_channels = v

    return nn.Sequential(*layers)





cfgs = {

    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],

    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],

    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],

    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],

}


def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):

    if pretrained:

        kwargs['init_weights'] = False

    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)

    if pretrained:

        state_dict = load_state_dict_from_url(model_urls[arch],

                                              progress=progress)

        model.load_state_dict(state_dict)

    return model



def vgg16_bn(pretrained=False, progress=True, **kwargs):

    r"""VGG 16-layer model (configuration "D") with batch normalization

    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_



    Args:

        pretrained (bool): If True, returns a model pre-trained on ImageNet

        progress (bool): If True, displays a progress bar of the download to stderr

    """

    return _vgg('vgg16_bn', 'D', True, pretrained, progress, **kwargs)



class VGG16_BN(nn.Module):
    def __init__(self, num_classes=2,init_weights=False):
        super(VGG16_BN, self).__init__()
        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_3 = nn.BatchNorm2d(256)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_3 = nn.BatchNorm2d(512)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2)  # 1/16

        # conv5
        # self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        # self.bn5_1 = nn.BatchNorm2d(512)
        # self.relu5_1 = nn.ReLU(inplace=True)
        # self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        # self.bn5_2 = nn.BatchNorm2d(512)
        # self.relu5_2 = nn.ReLU(inplace=True)
        # self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        # self.bn5_3 = nn.BatchNorm2d(512)
        # self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2)  # 1/32


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

        if init_weights:
            self._initialize_weights()



    def forward(self, x):
        h = x
        h = self.relu1_1(self.bn1_1(self.conv1_1(h)))
        h = self.relu1_2(self.bn1_2(self.conv1_2(h)))
        h = self.pool1(h)

        h = self.relu2_1(self.bn2_1(self.conv2_1(h)))
        h = self.relu2_2(self.bn2_2(self.conv2_2(h)))
        h = self.pool2(h)

        h = self.relu3_1(self.bn3_1(self.conv3_1(h)))
        h = self.relu3_2(self.bn3_2(self.conv3_2(h)))
        h = self.relu3_3(self.bn3_3(self.conv3_3(h)))
        h = self.pool3(h)

        h = self.relu4_1(self.bn4_1(self.conv4_1(h)))
        h = self.relu4_2(self.bn4_2(self.conv4_2(h)))
        h = self.relu4_3(self.bn4_3(self.conv4_3(h)))
        h = self.pool4(h)
        fea4 = h
        fea5 = self.pool5(h)

        # h = self.relu5_1(self.bn5_1(self.conv5_1(h)))
        # h = self.relu5_2(self.bn5_2(self.conv5_2(h)))
        # h = self.relu5_3(self.bn5_3(self.conv5_3(h)))
        # h = self.pool5(h)
        # fea5 = h
        
        h = self.avgpool(h)
        h = torch.flatten(h, 1)
        h = self.classifier(h)

        return fea4, fea5, h



    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)



class VGG16_BN_CAM(nn.Module):
    def __init__(self, num_classes=2,init_weights=True):
        super(VGG16_BN_CAM, self).__init__()
        # conv1
        self.conv1_p = nn.Conv2d(1, 64, 3, padding=1)
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_3 = nn.BatchNorm2d(256)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_3 = nn.BatchNorm2d(512)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(512)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_3 = nn.BatchNorm2d(512)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2)  # 1/32


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

        if init_weights:
            self._initialize_weights()



    def forward(self, x, cam=None):
        if cam is not None:
            h = self.relu1_1(self.bn1_1(self.conv1_1(x) + self.conv1_p(cam)))
        else:
            h = self.relu1_1(self.bn1_1(self.conv1_1(x)))
        h = self.relu1_2(self.bn1_2(self.conv1_2(h)))
        h = self.pool1(h)

        h = self.relu2_1(self.bn2_1(self.conv2_1(h)))
        h = self.relu2_2(self.bn2_2(self.conv2_2(h)))
        h = self.pool2(h)

        h = self.relu3_1(self.bn3_1(self.conv3_1(h)))
        h = self.relu3_2(self.bn3_2(self.conv3_2(h)))
        h = self.relu3_3(self.bn3_3(self.conv3_3(h)))
        h = self.pool3(h)
        pool3 = h  # 1/8

        h = self.relu4_1(self.bn4_1(self.conv4_1(h)))
        h = self.relu4_2(self.bn4_2(self.conv4_2(h)))
        h = self.relu4_3(self.bn4_3(self.conv4_3(h)))
        h = self.pool4(h)
        pool4 = h  # 1/16

        h = self.relu5_1(self.bn5_1(self.conv5_1(h)))
        h = self.relu5_2(self.bn5_2(self.conv5_2(h)))
        h = self.relu5_3(self.bn5_3(self.conv5_3(h)))
        h = self.pool5(h)
        feature = h
        
        h = self.avgpool(h)
        h = torch.flatten(h, 1)
        h = self.classifier(h)

        return feature, h



    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


######## CAM ############
import torch.nn.functional as F
def CAM(feature, net, idxs=[0,0,0]):
    wsize = 224
    size_upsample = (wsize, wsize)
    bz, nc, h, w = feature.shape
    
    params = list(net.parameters())
    weight_softmax = params[-2]
    
    output_cam = torch.zeros(bz,wsize,wsize).cuda()
    for i in range(bz):
        idx = idxs[i]
        cam = weight_softmax[idx:idx+1,:].mm(feature[i,:,:,:].reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - torch.min(cam)
        cam_img = cam / torch.max(cam)
        cam_img = F.interpolate(cam_img.reshape(1,1,h,w), size=size_upsample, mode='bilinear')
        output_cam[i,:,:] = cam_img.squeeze()
    return output_cam.unsqueeze(1)

def crop(img, prob_map, margin):
    wsize = 224
    size_upsample = (wsize, wsize)

    (N, C, W, H) = prob_map.shape
    minA = 0
    maxA = W
    minB = 0
    maxB = H
    
    output_img = torch.zeros(N,3,wsize,wsize).cuda()
    for i in range(N):
        binary_mask = (prob_map[i:i+1,:,:,:] >= 0.8)

        arr = torch.nonzero(binary_mask)
        minA = arr[:, 2].min().item()
        maxA = arr[:, 2].max().item()
        minB = arr[:, 3].min().item()
        maxB = arr[:, 3].max().item()

        bbox = [int(max(minA - margin, 0)), int(min(maxA + margin + 1, W)), \
        int(max(minB - margin, 0)), int(min(maxB + margin + 1, H))]
        cropped_image = img[i:i+1, :, bbox[0]: bbox[1], bbox[2]: bbox[3]].clone()
        cropped_image = F.interpolate(cropped_image, size=size_upsample, mode='bilinear')
        output_img[i:i+1,:,:,:] = cropped_image
    return output_img

def cam_idx(logit):
    h_x = F.softmax(logit, dim=1).data.cpu()
    probs, idx = h_x.sort(1, True)
    idx = idx.numpy()
    return idx
    


def mask(img, prob_map, margin):
    size_upsample = (224, 224)

    (N, C, W, H) = prob_map.shape
    minA = 0
    maxA = W
    minB = 0
    maxB = H
    
    output_img = torch.zeros(N,1,224,224).cuda()
    for i in range(N):
        binary_mask = (prob_map[i:i+1,:,:,:] >= 0.8)

        arr = torch.nonzero(binary_mask)
        minA = arr[:, 2].min().item()
        maxA = arr[:, 2].max().item()
        minB = arr[:, 3].min().item()
        maxB = arr[:, 3].max().item()

        bbox = [int(max(minA - margin, 0)), int(min(maxA + margin + 1, W)), \
        int(max(minB - margin, 0)), int(min(maxB + margin + 1, H))]
        output_img[i:i+1, :, bbox[0]: bbox[1], bbox[2]: bbox[3]] = 1
    output_img = output_img * img
    
    return output_img

# model
class FC_CAM(nn.Module):
    def __init__(self, num_classes=2):
        super(FC_CAM, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # self.fc = nn.Sequential(
        # nn.Linear(1024,512),
        # nn.BatchNorm1d(512),
        # nn.ReLU(inplace=True),
        # nn.Dropout(0.5),
        # nn.Linear(512,2)
        # )
        self.fc = nn.Linear(1024,2)

    def forward(self, x1, x2):
        x1 = self.avgpool(x1)
        x2 = self.avgpool(x2)

        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        x = torch.cat((x1, x2), 1)

        x = self.fc(x)

        return x


class FC_CAM1(nn.Module):
    def __init__(self, num_classes=2):
        super(FC_CAM1, self).__init__()

        self.se = SELayer(1024)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024,2)

    def forward(self, x1, x2):
        x = torch.cat((x1,x2), 1)
        x = self.se(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class FC_CAM2(nn.Module):
    def __init__(self, num_classes=2):
        super(FC_CAM2, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.wei = nn.Sequential(
        nn.Linear(1024,512),
        nn.Sigmoid(),
        )
        
        self.fc = nn.Linear(512,2)

    def forward(self, x1, x2):

        x = torch.cat((x1,x2), 1)
        # weight
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        xw1 = self.wei(x)
        xw2 = 1 - xw1
        
        b, c = xw1.size()
        xw1 = xw1.view(b,c,1,1)
        xw2 = xw2.view(b,c,1,1)
        x1 = x1 * xw1.expand_as(x1)
        x2 = x2 * xw2.expand_as(x2)
        
        y = x1 + x2
        y = self.avgpool(y)
        y = torch.flatten(y, 1)
        y = self.fc(y)

        return y


class FC_CAM3(nn.Module):
    def __init__(self, num_classes=2):
        super(FC_CAM3, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.se1 = SELayer(512)
        self.se2 = SELayer(512)
        
        self.conv1 = nn.Sequential(
            nn.Conv3d(2, 4, kernel_size=(1,7,7), stride=1),
            nn.ReLU(inplace=True),
            )
        self.conv2 = nn.Sequential(
            nn.Conv3d(4, 2, kernel_size=(1,1,1), stride=1),
            nn.Sigmoid(),
            )
        
        self.fc = nn.Linear(512,2)

    def forward(self, x1, x2):
        x11 = self.se1(x1)
        x21 = self.se2(x2)
        
        x12 = x1.unsqueeze(1)
        x22 = x2.unsqueeze(1)
        x = torch.cat((x12,x22), 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x1w = x[:,0,:,:,:]
        x1 = x1 * x1w.expand_as(x1)
        x1 = x1 + x11
        
        x2w = x[:,1,:,:,:]
        x2 = x2 * x2w.expand_as(x2)
        x2 = x2 + x21

        x = x1 + x2
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


######## CAM ###############


########### BCNN ###################
class SELayer(nn.Module):   #ChannelWiseAttention
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class BCNN_SE(nn.Module):
    def __init__(self):
        super(BCNN_SE, self).__init__()

        self.se = SELayer(channel=512, reduction=4)
        self.classifiers = nn.Sequential(
            nn.Linear(512 ** 2, 2),
        )

    def forward(self, x):
        x1 = x
        x = self.se(x)
        bz, c, h, w = x.shape
        x = x.view(bz, c, h*w)
        x1 = x1.view(bz, c, h*w)
        x = (torch.bmm(x, torch.transpose(x1, 1, 2)) / (h*w)).view(bz, -1)
        x = F.normalize(torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-10))

        x = self.classifiers(x)

        return x


class BCNN1(nn.Module):
    def __init__(self):
        super(BCNN1, self).__init__()

        self.classifiers = nn.Sequential(
            nn.Linear(512 ** 2, 2),
        )

    def forward(self, x):
        bz, c, h, w = x.shape
        x = x.view(bz, c, h*w)
        x = (torch.bmm(x, torch.transpose(x, 1, 2)) / (h*w)).view(bz, -1)
        x = F.normalize(torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-10))

        x = self.classifiers(x)

        return x
########### BCNN #################


########### triplet model ##############
class TripletModel(nn.Module):

    def __init__(self, num_classes):

        super(TripletModel, self).__init__()

        self.model = vgg16_bn(num_classes=num_classes, init_weights=False)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.model.classifier = nn.Linear(512, num_classes)


    def l2_norm(self,input):

        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output



    def forward(self, x):

        x = self.model.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        self.features = self.l2_norm(x)

        # Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
        alpha=10
        self.features = self.features*alpha

        return self.features



    def forward_classifier(self, features):
        res = self.model.classifier(features)
        return res






