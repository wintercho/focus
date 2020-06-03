# -*- coding: utf-8 -*-

# test for double branch model, eg: vgg+cam+caps

from __future__ import division
import os
from PIL import Image
from torch.utils import data 
from torchvision import transforms as T
from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable as V
from cnn_finetune import make_model
import csv
import pylab as pl
import numpy as np
# from graphviz import Digraph
from model import *
from capsules import *
from visual import *
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

### change the path for lists
list_root_train = '../lists/face/training.txt'
list_root_test = '../lists/face/testing.txt'

# set gpu name
from setproctitle import setproctitle
setproctitle('test')


def save_csv(name, lab, r0, r1, r2, savepath):
    with open(savepath, 'w', newline='') as csvfile:
        writer  = csv.writer(csvfile)
        list_ = [name,lab,r0,r1,r2]
        for row in zip(*list_):
            writer.writerow(row)


def logit2pred(out):
    _,result = torch.max(out,1)
    result = result.cpu().data.numpy()
    return result


def test():
    preds0 = [] #both
    preds1 = [] #1 branch
    preds2 = [] #2 branch
    for ii, test_img in enumerate(test_dataloader):
        test_img = V(test_img).cuda()
        featurest1, out1 = model(test_img)
        idx = cam_idx(out1)
        cam = CAM(featurest1, model, idxs=idx[:,0])
        test_crop_img = crop(test_img, cam, margin=20)
        # visul(test_crop_img, ii, 'crop_img/')
        # visul(test_img, ii, 'imgs/')
        featurest2, out2 = model(test_crop_img)
        out,_ = model2(featurest1, featurest2)
        result = logit2pred(out)
        preds0.append(int(result))
        
        result1 = logit2pred(out1)
        preds1.append(int(result1))
        result2 = logit2pred(out2)
        preds2.append(int(result2))
    return preds0, preds1, preds2

def test_accuracy(pred, labels):
    pred = np.array(pred)
    num = len(pred)
    accuracy = ((pred-labels).tolist().count(0))/num
    return accuracy
    
def test_prfa(pred, labels):
    precision = precision_score(labels, pred)
    recall = recall_score(labels, pred)
    f1 = f1_score(labels, pred)
    acc = accuracy_score(labels, pred)
    return precision, recall, f1, acc


if __name__ == '__main__':

    # hyper parameters #
    # the same path with trained model
    path = './pths1/'
    log_file = os.path.join(path, 'test_result.txt')
    output = open(log_file, 'w')
    output.close()
    
    model_path = path+'epoch1.pth'
    model_path2 = path+'epoch2.pth'
    input_sizes = (224,224)
    test_img_num = 3309
    num_class = 2
    em_iters = 2
    A = 512
    B = 32
    C = 32
    D = 32

    from data_process import get_processed_datas, Test_Dataset
    test_imgs, test_labels, test_name_list = get_processed_datas(list_root_test)
    
    transform = T.Compose([
        T.Resize(input_sizes),
        T.ToTensor(),
        T.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]),
        ])


    model = vgg16_bn(num_classes=num_class, init_weights=False).cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model2 = MyCapsNet_cam(A=A, B=B, C=C, D=D, E=num_class, iters=em_iters).cuda()
    model2.load_state_dict(torch.load(model_path2))
    model2.eval()

    test_dataset = Test_Dataset(test_imgs,transforms=transform)
    test_dataloader = DataLoader(test_dataset,batch_size=1,shuffle=False,num_workers=0,drop_last=False)
    preds0, preds1, preds2 = test()
    
    savepath = path + 'results.csv'
    r0 = preds0
    save_csv(test_name_list,test_labels,r0,preds1,preds2,savepath)

    print('accuracy0:', test_accuracy(preds0, test_labels))
    print('accuracy1:', test_accuracy(preds1, test_labels))
    print('accuracy2:', test_accuracy(preds2, test_labels))
    
    precision, recall, f1, acc = test_prfa(preds0, test_labels)
    print('precision:', precision)
    print('recall:', recall)
    print('f1:', f1)
    print('accuracy:', acc)



    output = open(log_file, 'a+')
    output.write("model_path:%s"%(model_path))
    output.write('\n')
    output.write("precision:%.4f"%(precision))
    output.write('\n')
    output.write("recall:%.4f"%(recall))
    output.write('\n')
    output.write("f1_score:%.4f"%(f1))
    output.write('\n')
    output.write("accuracy:%.4f"%(acc))
    output.write('\n')
    output.close()

    # pl.close("all")
    # for i in range(200,220,1):
        # pl.figure(i)
        # img = Image.open(test_imgs[i])
        # pl.suptitle(label_num[preds[i]])
        # pl.imshow(img)

        












 
 
    
    
    
    
    
    
    
    
    
    
        