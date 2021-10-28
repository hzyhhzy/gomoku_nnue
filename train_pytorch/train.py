
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

forced_save_points=[]


try:
    os.mkdir("saved_models")
except:
    pass
else:
    pass

try:
    os.mkdir("logs")
except:
    pass
else:
    pass


def cross_entropy_loss(output, target):
    t = torch.log_softmax(output,dim=1)
    loss = torch.mean(torch.sum(-t*target, dim=1), dim=0)+torch.mean(torch.sum(torch.log(target+1e-8)*target, dim=1), dim=0)
    return loss




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int,
                        default=0, help='which gpu')
    parser.add_argument('--cpu', action='store_true', default=False, help='whether use cpu')
    parser.add_argument('--bs', type=int,
                        default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')
    parser.add_argument('--wd', type=float, default=0, help='weight decay')
    parser.add_argument('--data', type=str,
                        default='../alldata_p1_v1_6x96.npz', help='trainset path')
    parser.add_argument('--type', type=str, default='sum1',help='model type defined in model.py')
    parser.add_argument('--save', type=str ,default='test', help='model save pth')
    parser.add_argument('--new', action='store_true', default=False, help='whether to retrain')
    parser.add_argument('--epoch', type=int,
                        default=100000, help='epoch num')
    parser.add_argument('--maxstep', type=int,
                        default=5000000000, help='max step to train')
    parser.add_argument('--savestep', type=int,
                        default=2000, help='step to save')
    parser.add_argument('--infostep', type=int,
                        default=500, help='step to logger')
    parser.add_argument('--size', nargs='+',type=int,
                        default=(128), help='model size')
    parser.add_argument('--val', action='store_true', default=False, help='whether use validation')
    parser.add_argument('--vdata', type=str,
                        default='../val_100k.npz', help='valset path')
    parser.add_argument('--vstep', type=int,
                        default=5000, help="how many step each validation")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    if(args.cpu):
        device=torch.device('cpu')
    batch_size=args.bs
    lr=args.lr
    trainset_path=args.data
    vset_path=args.vdata
    model_type=args.type
    save_name=args.save
    maxepoch=args.epoch
    maxstep=args.maxstep
    savestep=args.savestep
    infostep=args.infostep
    modelsize=args.size






    print("Loading data")
    tDataSet = trainset(trainset_path,sym=True)
    dataloader = DataLoader(tDataSet, shuffle=True, batch_size=batch_size)
    if(args.val):
        vDataSet = trainset(vset_path)
        vdataloader = DataLoader(vDataSet, shuffle=True, batch_size=batch_size)
    print("Finished loading data")
    totalstep = 0
    file_path = f'saved_models/{save_name}.pth'

    if os.path.exists(file_path) and (not args.new) and (args.save != 'test'):
        data = torch.load(file_path)
        model_type=data['model_name']
        model_param=data['model_size']
        model = ModelDic[model_type](*model_param).to(device)

        model.load_state_dict(data['state_dict'])
        totalstep = data['step']
        print(f"loaded model: type={data['model_name']}, size={model.model_size}, totalstep={totalstep}")
    else:
        model = ModelDic[model_type](*modelsize).to(device)

    optimizer = optim.Adam(model.parameters(), lr,weight_decay=args.wd)
    model.train()


    time0=time.time()
    loss_record=[0,0,0,1e-7]
    print("Start training")
    for epochs in range(maxepoch):
        for step, (board, valueTarget, policyTarget) in enumerate(dataloader):
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

            loss_record[0]+=(vloss.detach()+ploss.detach())
            loss_record[1]+=vloss.detach()
            loss_record[2]+=ploss.detach()
            loss_record[3]+=1

            loss.backward()
            optimizer.step()

            # logs
            totalstep += 1
            if(totalstep % infostep == 0):

                print("name: {}, time: {:.2f} s, step: {}, totalloss: {:.4f}, vloss: {:.4f}, ploss: {:.4f}"
                      .format(save_name,time.time()-time0,totalstep,loss_record[0]/loss_record[3],loss_record[1]/loss_record[3],loss_record[2]/loss_record[3]))

                logfile = open(f'logs/train_{save_name}.txt','a')
                print("{} {:.3f} {:.3f} {:.3f} "
                      .format(totalstep,loss_record[0]/loss_record[3],loss_record[1]/loss_record[3],loss_record[2]/loss_record[3]),file=logfile)
                logfile.close()

                loss_record = [0, 0, 0, 1e-7]
                time0=time.time()

            if(totalstep % args.vstep == 0 and args.val):

                loss_record_val = [0, 0, 0, 1e-7]
                with torch.no_grad():
                    for s, (board, valueTarget, policyTarget) in enumerate(vdataloader):
                        # data
                        board = board.to(device)
                        valueTarget = valueTarget.to(device)
                        policyTarget = policyTarget.to(device)

                        value, policy = model(board)

                        vloss = cross_entropy_loss(value, valueTarget)
                        ploss = cross_entropy_loss(policy.flatten(start_dim=1), policyTarget.flatten(start_dim=1))
                        loss = 1.2*vloss+1.0*ploss

                        loss_record_val[0]+=(vloss.detach()+ploss.detach())
                        loss_record_val[1]+=vloss.detach()
                        loss_record_val[2]+=ploss.detach()
                        loss_record_val[3]+=1


                print("val: name: {}, step: {}, totalloss: {:.4f}, vloss: {:.4f}, ploss: {:.4f}"
                      .format(save_name,totalstep,loss_record_val[0]/loss_record_val[3],loss_record_val[1]/loss_record_val[3],loss_record_val[2]/loss_record_val[3]))
                logfile = open(f'logs/val_{save_name}.txt','a')

                print("{} {:.3f} {:.3f} {:.3f} "
                      .format(totalstep,loss_record_val[0]/loss_record_val[3],loss_record_val[1]/loss_record_val[3],loss_record_val[2]/loss_record_val[3]),file=logfile)
                logfile.close()


            if totalstep in forced_save_points:
                file_path_mid = f'saved_models/{save_name}_s{totalstep}.pth'
                print(f"Finished training {totalstep} steps")
                torch.save(
                    {'step': totalstep, 'state_dict': model.state_dict(), 'model_name': model.model_name,'model_size':model.model_size}, file_path_mid)
                print('Model saved in {}\n'.format(file_path_mid))

            if totalstep%savestep==0:
                print(f"Finished training {totalstep} steps")
                torch.save(
                    {'step': totalstep, 'state_dict': model.state_dict(), 'model_name': model.model_name,'model_size':model.model_size}, file_path)
                print('Model saved in {}\n'.format(file_path))

            if step >= maxstep:
                break

    print(f"Finished training {totalstep} steps")
    torch.save(
        {'step': totalstep, 'state_dict': model.state_dict(), 'model_name': model.model_name,
         'model_size': model.model_size}, file_path)
    print('Model saved in {}\n'.format(file_path))