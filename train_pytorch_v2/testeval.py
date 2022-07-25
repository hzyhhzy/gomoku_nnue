
from dataset import trainset
from model import ModelDic


import argparse
import glob
import sys
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import os
import time





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str ,default='test', help='model path')
    args = parser.parse_args()

    modelname=args.model



    file_path = f'../saved_models/{modelname}/model.pth'
    model_type=None
    if os.path.exists(file_path):
        data = torch.load(file_path)
        model_type=data['model_type']
        model_param=data['model_param']
        model = ModelDic[model_type](*model_param)

        model.load_state_dict(data['state_dict'])
        totalstep = data['totalstep']
        print(f"loaded model: type={model_type}, size={model_param}, totalstep={totalstep}")
    else:
        print("Invalid Model Path")
        exit(0)

    model.eval()

    boardstr='' \
             '. . . . . . . . . . . . . . . '\
             '. . . . . . . . . . . . . . . '\
             '. . . . . . . . . . . . . . . '\
             '. . . . . . . . . . . . . . . '\
             '. . . . . . . . . . . . . . . '\
             '. . . . . . . . . . . . . . . '\
             '. . . . . . . . x . . . . . . '\
             '. . . . . . . x . o . . . . . '\
             '. . . . . . o . . . . . . . . '\
             '. . . . . . . . . . . . . . . '\
             '. . . . . . . . . . . . . . . '\
             '. . . . . . . . . . . . . . . '\
             '. . . . . . . . . . . . . . . '\
             '. . . . . . . . . . . . . . . '\
             '. . . . . . . . . . . . . . . '


    board=torch.zeros((1,2,15,15))
    for y in range(15):
        for x in range(15):
            c=boardstr[2*(y*15+x)]
            if(c=='x'):
                board[0,0,y,x]=1
            if(c=='o'):
                board[0,1,y,x]=1
    v,p=model(board)
    print(v,p)
    model.testeval(board,7*15+7)