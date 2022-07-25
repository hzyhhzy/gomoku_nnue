
from dataset import trainset
from model import ModelDic


import argparse
import glob
import sys
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.functional as F
import torch.nn as nn
import torch
import os
import time

try:
    os.mkdir("export")
except:
    pass
else:
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int,
                        default=0, help='which gpu')
    parser.add_argument('--cpu', action='store_true', default=False, help='whether use cpu')
    parser.add_argument('--model', type=str ,default='test', help='model path')
    parser.add_argument('--export', type=str ,default='', help='export path')
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    if(args.cpu):
        device=torch.device('cpu')
    modelname=args.model
    exportname=args.export
    if(exportname==''):
        exportname=modelname







    file_path = f'saved_models/{modelname}.pth'
    model_type=None
    if os.path.exists(file_path):
        data = torch.load(file_path)
        model_type=data['model_name']
        model_param=data['model_size']
        model = ModelDic[model_type](*model_param).to(device)

        model.load_state_dict(data['state_dict'])
        totalstep = data['step']
        print(f"loaded model: type={data['model_name']}, size={model.model_size}, totalstep={totalstep}")
    else:
        print("Invalid Model Path")
        exit(0)

    model.eval()


    time0=time.time()
    print("Start")

    #prepare data
    def fullData(length):  # 三进制全部排列
        x = np.zeros((1, length), dtype=np.int64)
        for i in range(length):
            x1 = x.copy()
            x1[:, i] = 1
            x2 = x.copy()
            x2[:, i] = 2
            x = np.concatenate((x, x1, x2), axis=0)
        return x

    buf=np.ones((4*(3**11),4),dtype=np.float64)*114514

    pow3 = np.array([1, 3, 9, 27, 81, 243, 729, 2187, 6561,3**9,3**10,3**11], dtype=np.int64)
    pow3=pow3[:,np.newaxis]
    #无边界和单边边界 长度9

    data=fullData(11)
    label=model.exportMapTable(data,device).numpy()

    #无边界与正边界
    for r in range(6):
        ids=np.matmul(data[:,r:],pow3[:11-r])+pow3[12-r:].sum()
        for i in range(ids.shape[0]):
            buf[ids[i]]=label[:,i,5+r]

    #负边界
    for l in range(1,6):
        ids=np.matmul(data[:,:11-l],pow3[l:-1])+2*3**11+pow3[:l-1].sum()
        for i in range(ids.shape[0]):
            buf[ids[i]]=label[:,i,5-l]

    #双边边界
    for left in range(1,6):
        for right in range(1,6):
            L=11-left-right
            data = fullData(L)
            label=model.exportMapTable(data,device).numpy()
            idbias=3*3**11+pow3[0:left-1].sum((0,1))+pow3[12-right:-1].sum((0,1))
            ids=np.matmul(data,pow3[left:11-right])+idbias
            for i in range(ids.shape[0]):
                buf[ids[i]]=label[:,i,5-left]

    #avoid bias
    useful=(buf[:,0]!=114514).astype(np.float64)
    usefulcount=useful.sum()
    avg=(useful[:,np.newaxis]*buf).sum(0)/usefulcount
    print("Useful=",usefulcount)
    print("Bias=",avg)
    policybias=avg[0]
    valuebias=avg[1:4].mean()
    buf[:,0]-=policybias
    buf[:,1:4]-=valuebias




    #np.set_printoptions(suppress=True,precision=3)

    exportfile=open('export/'+exportname+'.txt','w')
    for i in range(buf.shape[0]):
        if useful[i]>0.5:
            print(i,end=' ',file=exportfile)
            for j in range(4):
                print(('%.3f' % buf[i,j]),end=' ',file=exportfile)
            print('',file=exportfile)
        #else:
            #print(i)

    exportfile.close()
    print("success")






