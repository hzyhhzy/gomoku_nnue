import torch
import torch.nn as nn
import math
import torch.functional as F
import numpy as np


def fullData(length):  # 三进制全部排列
    x = np.zeros((1, length), dtype=np.int32)
    for i in range(length):
        x1 = x.copy()
        x1[:, i] = 1
        x2 = x.copy()
        x2[:, i] = 2
        x = np.concatenate((x, x1, x2), axis=0)
    b = (x == 1)
    w = (x == 2)
    out = np.stack((b, w), axis=0).astype(np.float32)
    return out[np.newaxis]
print(fullData(3))
print(fullData(9).shape)
#h1=torch.tensor(np.arange(9).reshape(3,3),requires_grad=True,dtype=torch.float32)
#nn.LSTM
#h=torch.stack((h1[1],h1[2],h1[1],0),dim=0)
#w=torch.tensor([[[0,1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1,0],[0,0,0,1,0,0,0,0,0],[0,0,0,0,0,1,0,0,0]]],requires_grad=False,dtype=torch.float32).reshape(4,1,3,3)
#h=torch.conv2d(h1,w,None,padding=1)
#print(h1.std())
#uf=nn.Unfold(3,padding=1)
#fo=nn.Fold(output_size=(3,3),kernel_size=3,padding=1)
#h=uf(h1)
#h1=h1[1:4,1:4,1:4]
#print(h1,h)
#h=fo(h)
#print(h)
#
# print(h1)
#
# pad=nn.ZeroPad2d(1)
#
# h=pad(h1)
# print(h)
#
# h=torch.stack((h[0:-2,1:-1],h[2:,1:-1],h[1:-1,0:-2],h[1:-1,2:]),dim=1)
# print(h,h.shape)
#
# loss=torch.sum(h,dim=(0,1,2))
# loss.backward()
# print(h1.grad)
#
# h2=torch.tensor(np.arange(9).reshape(3,3),requires_grad=True,dtype=torch.float32)
#
# print(h1,h2,h1+h2)