# -*- coding: utf-8 -*-

#### train cnn baseline with capsnet & cam, cnn baseline "share" weight of the two branch; cam for crop img
import torch
from torchvision import transforms as T
from torch.utils.data import DataLoader
from data_process import get_processed_datas, Dataset, upsample
from torch.autograd import Variable as V
from torch import optim
from cnn_finetune import make_model
import torch.nn as nn
from model import *
import math
import os
import numpy as np
from capsules import *
from loss import SpreadLoss, Reconstructloss, Reconstructloss_fc, TripletLoss
from torchvision import models
from visual import *

### change the path for lists
list_root = '../lists/animals/training.txt'
list_root_test = '../lists/animals/testing.txt'


# set gpu name
from setproctitle import setproctitle
setproctitle('train')



def train():
    for epoch in range(1, epoches+1):
        running_loss = 0
        running_loss1 = 0
        running_loss2 = 0
        train_acc = 0
        run_nums = 0
        model1.train()
        model2.train()
        for ii, (img, label) in enumerate(dataloader):
            img, label = V(img).cuda(), V(label).cuda()
            optimizer1.zero_grad()
            features1, prediction1 = model1(img)
            idx = cam_idx(prediction1)
            cam = CAM(features1, model1, idxs=idx[:,0])
            crop_img = crop(img, cam, margin=20)
            features2 = model1.features(crop_img)
            prediction, pose = model2(features1, features2)
            # print('predict:%.4f,label:%.4f', prediction, label)
            r = (1.*ii + (epoch-1)*train_len) / (epoches*train_len)
            loss1 = net_loss(prediction,label,r)
            # entropy loss
            loss2 = net_loss1(prediction1,label)
            if epoch < 5:
                lamb = 0.4
            elif epoch < 9:
                lamb = 0.9
            elif epoch < 16:
                lamb = 0.99
            else:
                lamb = 0.95
            loss = lamb*loss1 + (1-lamb)*loss2
            loss.backward()
            optimizer1.step()
            
            running_loss += loss.data.cpu() #whole loss
            running_loss1 += loss1.data.cpu() #spread loss
            running_loss2 += loss2.data.cpu() #cnn loss
            _,result1 = torch.max(prediction,1)
            correct1 = torch.sum(result1 == label.data).to(torch.float32)
            train_acc += correct1
            run_nums += label.shape[0]
        acc = train_acc/run_nums
        # scheduler.step(acc)
        # if scheduler.get_lr()[0] > 1e-5:
            # scheduler.step()
        scheduler.step()

        valacc, valacc1 = validation(model1, model2)
        print("After %d epochs, loss=%.4f, loss1=%.4f, loss2=%.4f, tra_acc=%.4f, val_acc=%.4f, val_acc1=%.4f"%(epoch,running_loss/train_len,running_loss1/train_len,running_loss2/train_len, acc, valacc, valacc1))
        torch.save(model1.state_dict(), path+'epoch1-%d.pth'%(epoch))
        torch.save(model2.state_dict(), path+'epoch2-%d.pth'%(epoch))
        
        output = open(log_file, 'a+')
        output.write("Epoch[%02d/%d], loss=%.4f, loss1=%.4f, loss2=%.4f, tra_acc=%.4f, val_acc=%.4f, val_acc1=%.4f" \
                     %(epoch, epoches, running_loss/train_len,running_loss1/train_len,running_loss2/train_len, acc, valacc, valacc1))
        output.write('\n')
        output.close()

def validation(model1, model2):
    model1.eval()
    model2.eval()
    runningacc = 0
    runningacc1 = 0
    for ii, (test_img, test_labels) in enumerate(test_dataloader):
        test_img = V(test_img).cuda()
        test_labels = V(test_labels).cuda()
        featurest1, out1 = model1(test_img)
        idx = cam_idx(out1)
        cam = CAM(featurest1, model1, idxs=idx[:,0])
        test_crop_img = crop(test_img, cam, margin=20)
        featurest2 = model1.features(test_crop_img)
        out, pout = model2(featurest1, featurest2)

        _,result = torch.max(out,1)
        correct = torch.sum(result == test_labels.data).to(torch.float32)
        runningacc += correct
        
        _,result1 = torch.max(out1,1)
        correct1 = torch.sum(result1 == test_labels.data).to(torch.float32)
        runningacc1 += correct1
        
    val_acc = runningacc/test_img_num
    val_acc1 = runningacc1/test_img_num
    return val_acc, val_acc1




if __name__ == '__main__':
    
    # hyper parameters #
    # saving trained model in the path
    path = './pths1/'
    log_file = os.path.join(path, 'training_info.txt')
    output = open(log_file, 'w')
    output.close()

    epoches = 100
    batch_sizes = 16
    train_img_num = 5542
    test_img_num = 1848
    input_sizes = (224,224)
    num_class = 2
    em_iters = 2
    A = 512
    B = 16
    C = 8
    D = 4


    #### train ####
    # model1 = make_model('resnet34',num_classes=num_class,pretrained=True,input_size=input_sizes,dropout_p=0.5).cuda()
    
    # model1 = vgg16_bn(num_classes=num_class, init_weights=True).cuda()
    # model1.load_state_dict(torch.load(path+'epoch1-53.pth'))
    
    pretrained_model = models.vgg16_bn(num_classes=1000)
    pretrained_model.load_state_dict(torch.load('../code_face/pretrained/vgg16_bn.pth'))
    pretrained_dict = pretrained_model.state_dict()
    model1 = vgg16_bn(num_classes=num_class, init_weights=False)
    model_dict = model1.state_dict()
    pretrained_dict = {k: v for k,v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model1.load_state_dict(model_dict)
    model1 = model1.cuda()
    
    model2 = MyCapsNet_cam16(A=A, B=B, C=C, D=D, E=num_class, iters=em_iters).cuda()
    # model2.load_state_dict(torch.load(path+'epoch2-53.pth'))


    # numbers of parameters
    model_parameters1 = filter(lambda p: p.requires_grad, model1.parameters())
    params1 = sum([np.prod(p.size()) for p in model_parameters1])
    print('model1 parameters:', params1)
    model_parameters2 = filter(lambda p: p.requires_grad, model2.parameters())
    params2 = sum([np.prod(p.size()) for p in model_parameters2])
    print('model2 parameters:', params2)
    output = open(log_file, 'a+')
    output.write("model1 parameters:%d, model2 parameters:%d"%(params1,params2))
    output.write('\n')
    output.close()
    
    imgs,labels,name_list = get_processed_datas(list_root)
    imgs,labels = upsample(imgs,labels)
    dataset = Dataset(imgs,labels,is_train=True)
    dataloader = DataLoader(dataset,batch_size=batch_sizes,shuffle=True,num_workers=0,drop_last=True)
    train_len = len(dataloader)
    print('train_len:', train_len)
    # loss
    # weights = torch.FloatTensor([0.75,0.25]).cuda()
    net_loss1 = torch.nn.CrossEntropyLoss(weight=None, size_average=True)
    net_loss = SpreadLoss(num_class=num_class, m_min=0.2, m_max=0.9)

    # optimizer1 = optim.Adam(params=model.parameters(),lr=0.0001,weight_decay=1e-5)
    # optimizer1 = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    lr = 0.01
    # convp_params = list(map(id, model1.conv1_p.parameters()))
    # base_params = filter(lambda p: id(p) not in convp_params, model1.parameters())
    optimizer1 = optim.SGD([
                {'params': model1.parameters(), 'lr': lr},
                {'params': model2.parameters(), 'lr': lr}
            ], lr=lr, momentum=0.9, weight_decay=0.0005)
    # optimizer1 = optim.Adam([
                # {'params': model1.parameters(), 'lr': lr},
                # {'params': model2.parameters(), 'lr': lr}
            # ], lr=lr, weight_decay=0)
            
    # Use exponential decay leanring rate
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer1, gamma=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer1, 'max', factor=0.5)
    # scheduler = optim.lr_scheduler.StepLR(optimizer1, step_size=53, gamma=0.1)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer1, milestones=[7,14], gamma=0.1)
    
    #### validation ####
    test_imgs, test_labels, test_name_list = get_processed_datas(list_root_test)
    test_dataset = Dataset(test_imgs,test_labels,is_train=False)
    test_dataloader = DataLoader(test_dataset,batch_size=1,shuffle=False,num_workers=0,drop_last=False)
    
    train()
    











