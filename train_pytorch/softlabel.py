
from dataset import trainset
from model import ModelDic


import argparse
import glob
import sys
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import os
import time





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str,
                        default='0', help='which gpu')
    parser.add_argument('--device', type=str,
                        default='cuda', help='cpu or cuda')
    parser.add_argument('--bs', type=int,
                        default=128, help='batch size')
    parser.add_argument('--data', type=str,
                        default='../alldata_p1_v1.npz', help='trainset path')
    parser.add_argument('--model', type=str , help='model save pth')
    opt = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    device=opt.device
    batch_size=opt.bs
    trainset_path=opt.data
    model_name=opt.model

    model_path = f'saved_models/{model_name}.pth'
    save_path=trainset_path[:-4]+'_'+model_name+'.npz'





    print("Loading data")
    myDataSet = trainset(trainset_path,sym=False)
    myDataSet.vt=np.zeros((myDataSet.__len__()))#delete pt to save memory
    myDataSet.pt=np.zeros((myDataSet.__len__()))#delete pt to save memory
    print("Finished loading data")

    data = torch.load(model_path)
    model = ModelDic[data['model_name']](*data['model_size']).to(device)
    model.load_state_dict(data['state_dict'])
    print(f"loaded model: type={data['model_name']}, size={data['model_size']}, totalstep={data['step']}")


    dataloader = DataLoader(myDataSet, shuffle=False, batch_size=batch_size)
    model.eval()

    new_vt=[]
    new_pt=[]

    expected_step=myDataSet.__len__()/batch_size

    with torch.no_grad():
        for step, (board,  valueTarget, policyTarget) in enumerate(dataloader):
            # data
            board = board.to(device)
            value, policy = model(board)
            value=value.cpu()
            policy=policy.cpu()
            value=torch.softmax(value,dim=1)
            policy=torch.softmax(policy.flatten(start_dim=1),dim=1).view(policy.shape)
            new_vt.append(value.numpy())
            new_pt.append(policy.numpy())
            if(step%1000==0):
                print('{:.2%}'.format(step/expected_step))

    new_vt=np.concatenate(new_vt)
    new_pt=np.concatenate(new_pt)
    print(new_pt.shape,new_vt.shape)



    np.savez_compressed(save_path, bf=myDataSet.bf, gf=myDataSet.gf, vt=new_vt, pt=new_pt)