#import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

from config import *

def apply_sym(d,sym,dimY,dimX):
    x_sym=sym%2
    sym=sym//2

    y_sym=sym%2
    sym=sym//2

    transp=sym%2

    if(x_sym):
        d=np.flip(d,axis=dimX)
    if(y_sym):
        d=np.flip(d,axis=dimY)
    if(transp):
        d=np.swapaxes(d,dimX,dimY)

    return d.copy()

def apply_sym_policyTarget(pt,sym):
    if(sym==0):
        return pt
    assert(pt.ndim==1)
    assert(pt.shape[0]==boardH*boardW+1)
    ptboard=pt[0:boardH*boardW].reshape(boardH,boardW)
    ptboard=apply_sym(ptboard,sym,0,1)
    ptboard=ptboard.reshape(-1)
    pt=np.append(ptboard,pt[-1])
    return pt.copy()

class trainset(Dataset):
    #randomsym 每个样本随机对称，总数不变
    #symcopy 每个样本的8个随机都作为单独样本，总数乘8
    #两者不可同时为True
    def __init__(self, npz_path,randomsym=False,symcopy=False):

        data = np.load(npz_path)
        self.bf=data["bf"]
        self.gf=data["gf"]
        self.vt=data["vt"]
        self.pt=data["pt"]
        assert not (randomsym and symcopy),"randomsym和symcopy两者不可同时为True"
        self.symcopy=symcopy

        if randomsym:
            self.syms=np.random.randint(0,8,self.vt.shape[0])
        else:
            self.syms=np.zeros(self.vt.shape[0],dtype=np.int)

        #print(f"Total {self.vt.shape[0]} rows")
    def __getitem__(self, index):

        if(self.symcopy):
            sym = index % 8
            index = index // 8
        else:
            sym=self.syms[index]

        bf1=self.bf[index].astype(np.float32)
        pt1=self.pt[index].astype(np.float32)

        #apply symmetry
        bf1=apply_sym(bf1,sym,1,2)
        pt1=apply_sym_policyTarget(pt1,sym=sym)

        #concat bf and gf
        gf1=self.gf[index].astype(np.float32)
        gf1 = gf1.reshape((gf1.shape[0], 1, 1)).repeat(bf1.shape[1], axis=1).repeat(bf1.shape[2], axis=2)
        bf1 = np.concatenate((bf1, gf1), axis=0)

        vt1=self.vt[index].astype(np.float32)

        return bf1,vt1,pt1
    def __len__(self):
        if self.symcopy:
            return 8*self.vt.shape[0]
        else:
            return self.vt.shape[0]



