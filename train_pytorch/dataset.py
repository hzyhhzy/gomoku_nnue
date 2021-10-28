#import torch
from torch.utils.data import Dataset, DataLoader

import os
import numpy as np

def apply_sym(d,sym,dimY,dimX):
    x_sym=sym%2
    sym=sym//2

    y_sym=sym%2
    sym=sym//2

    transp=sym%2

    if(x_sym>0.5):
        d=np.flip(d,axis=dimX)
    if(y_sym>0.5):
        d=np.flip(d,axis=dimY)
    if(transp>0.5):
        d=np.swapaxes(d,dimX,dimY)

    return d.copy()

class trainset(Dataset):
    def __init__(self, npz_path,sym=False):
        data = np.load(npz_path)
        self.bf=data["bf"]
        self.gf=data["gf"]
        self.vt=data["vt"]
        self.pt=data["pt"]
        self.sym=sym
        #print(f"Total {self.vt.shape[0]} rows")
    def __getitem__(self, index):

        sym=0
        if(self.sym):
            sym = index % 8
            index = index // 8

        bf1=self.bf[index].astype(np.float32)
        pt1=self.pt[index].astype(np.float32)
        assert (bf1.ndim==3)

        #apply symmetry
        if(self.sym):
            bf1=apply_sym(bf1,sym,1,2)
            pt1=apply_sym(pt1,sym,0,1)

        #concat bf and gf
        gf1=self.gf[index].astype(np.float32)
        gf1 = gf1.reshape((gf1.shape[0], 1, 1)).repeat(bf1.shape[1], axis=1).repeat(bf1.shape[2], axis=2)
        bf1 = np.concatenate((bf1, gf1), axis=0)

        vt1=self.vt[index].astype(np.float32)
        # if(len(gf1.shape)==1):
        #     #print("type1")
        #     gf1 = gf1.reshape((gf1.shape[0], 1, 1)).repeat(bf1.shape[1], axis=1).repeat(bf1.shape[2], axis=2)
        #     bf1 = np.concatenate((bf1, gf1), axis=0)
        # elif(len(gf1.shape)==2):
        #     #print("type2")
        #     gf1 = gf1.reshape((gf1.shape[0], gf1.shape[1], 1, 1)).repeat(bf1.shape[2], axis=2).repeat(bf1.shape[3], axis=3)
        #     bf1 = np.concatenate((bf1, gf1), axis=1)
        # else:
        #     print("Unknown index type")

        return bf1,vt1,pt1
    def __len__(self):
        if self.sym:
            return 8*self.vt.shape[0]
        else:
            return self.vt.shape[0]



