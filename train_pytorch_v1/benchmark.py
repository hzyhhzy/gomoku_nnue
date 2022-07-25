
from torch.utils.data import Dataset, DataLoader
from dataset import trainset
from model import ModelDic
import argparse
import glob
import sys
import matplotlib.pyplot as plt
import torch.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import os
import time
import numpy as np


def depth_weight(d):
    return 1
def cross_entropy_loss(output, target):
    t = torch.log_softmax(output,dim=1)
    loss = torch.mean(torch.sum(-t*target, dim=1), dim=0)
    return loss


def train(batch_size, lr, device, myDataSet, model, maxstep):
    totalstep = 0
    #print(model.device)
    optimizer = optim.Adam(model.parameters(), lr)
    dataloader = DataLoader(myDataSet, shuffle=True, batch_size=batch_size)
    model.train()


    time0=0
    for step, (board,  valueTarget, policyTarget) in enumerate(dataloader):
        # data
        board = board.to(device)
        valueTarget = valueTarget.to(device)
        policyTarget = policyTarget.to(device)
        # optimize
        optimizer.zero_grad()
        value, policy = model(board)
        vloss = cross_entropy_loss(value, valueTarget)
        ploss = cross_entropy_loss(policy.flatten(start_dim=1), policyTarget.flatten(start_dim=1))
        loss = 1.2*vloss+1.0*ploss

        loss.backward()
        optimizer.step()

        # logs
        totalstep += 1
        if(totalstep  == 3):
            time0=time.time()
        if(totalstep  == 3+maxstep):
            time0=time.time()-time0
            break
    #print(f"{maxstep} batches, {time0} seconds, {maxstep/time0} batches/s, {batch_size*maxstep/time0} samples/s")
    speed=batch_size * maxstep / time0

    return speed


def forw(batch_size, lr, device, myDataSet, model, maxstep):
    totalstep = 0
    #print(model.device)
    dataloader = DataLoader(myDataSet, shuffle=True, batch_size=batch_size)
    model.eval()


    time0=0

    for step, (board,  valueTarget, policyTarget) in enumerate(dataloader):
        # data
        board = board.to(device)
        # optimize
        value, policy = model(board)
        # logs
        totalstep += 1
        if(totalstep  == 3):
            time0=time.time()
        if(totalstep  == 3+maxstep):
            time0=time.time()-time0
            break
    #print(f"{maxstep} batches, {time0} seconds, {maxstep/time0} batches/s, {batch_size*maxstep/time0} samples/s")
    speed=batch_size * maxstep / time0

    return speed



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int,
                        default=0, help='which gpu')
    parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')
    parser.add_argument('--data', type=str,
                        default='../small_p1_v1.npz', help='trainset path')
    parser.add_argument('--eval', action="store_true", help='test or eval')
    parser.add_argument('--type', type=str,default='res', help='model type defined in model.py')
    parser.add_argument('--step', type=int,
                        default=5, help='max step to train')
    parser.add_argument('-s', type=int,
                        default=8, help='batchsize in range(s,e,j)')
    parser.add_argument('-e', type=int,
                        default=2060, help='batchsize in range(s,e,j)')
    parser.add_argument('-j', type=int,
                        default=8, help='batchsize in range(s,e,j)')

    parser.add_argument('--size', nargs='+',type=int,
                        default=(6,96), help='model size')

    opt = parser.parse_args()

    device = torch.device(f"cuda:{opt.gpu}")
    print(opt.size)
    batchsize_x = []
    speed_y = []
    best_speed=0
    best_bs=0

    myDataSet = trainset(opt.data)
    model = ModelDic[opt.type](*opt.size).to(device)
    param_num=sum([p.numel() for p in model.parameters() if p.requires_grad])
    print("Param num", param_num)
    for bs in range(opt.s,opt.e+1,opt.j):
        try:
            if(opt.eval):
                speed=forw(batch_size=bs,
                  lr=opt.lr,
                  device=device,
                  myDataSet=myDataSet,
                  model=model,
                  maxstep=opt.step
                  )

            else:
                speed=train(batch_size=bs,
                  lr=opt.lr,
                  device=device,
                  myDataSet=myDataSet,
                  model=model,
                  maxstep=opt.step
                  )
        except:
            print(f"Failed to test batchsize {bs}")
            print(f"Best batchsize is {best_bs}, speed is {best_speed}")
            break
        else:
            batchsize_x.append(bs)
            speed_y.append(speed)
            if(speed>best_speed):
                best_speed=speed
                best_bs=bs

            print(f"bs={bs},  {speed} samples/s, bestbs={best_bs}, bestspeed={best_speed}")


    plt.plot(batchsize_x,speed_y)
    np.savez(f"benchmark_{opt.type}_{opt.size}.npz",batchsize_x=np.array(batchsize_x),speed_y=np.array(speed_y))
    plt.show()