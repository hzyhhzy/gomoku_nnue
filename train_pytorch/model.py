from typing import Iterable
import torch
import torch.nn as nn
import torchvision.transforms as tt
import math
import torch.functional as F
from torch import randn
import numpy as np

boardH = 15
boardW = 15
input_c=3



def swish(x):
    return torch.sigmoid(x)*x
def tupleOp(f,x):
    return (f(x[0]),f(x[1]),f(x[2]),f(x[3]))
def conv1dDirection(x,w,b,zerow,type):

    out_c=w.shape[1]
    in_c=w.shape[2]
    if(type==0):
        w=torch.stack((zerow,zerow,zerow,
                       w[0],w[1],w[2],
                       zerow,zerow,zerow),dim=2)
    elif(type==1):
        w=torch.stack((zerow,w[0],zerow,
                       zerow,w[1],zerow,
                       zerow,w[2],zerow),dim=2)
    elif(type==2):
        w=torch.stack((w[0],zerow,zerow,
                       zerow,w[1],zerow,
                       zerow,zerow,w[2]),dim=2)
    elif(type==3):
        w=torch.stack((zerow,zerow,w[2],
                       zerow,w[1],zerow,
                       w[0],zerow,zerow),dim=2)
    w=w.view(out_c,in_c,3,3)

    x = torch.conv2d(x,w,b,padding=1)
    return x


class Conv1dLayer(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.zerow = nn.Parameter(torch.zeros((out_c, in_c)),False) #constant zero
        self.w = nn.Parameter(torch.empty((3,out_c, in_c)), True)
        nn.init.kaiming_uniform_(self.w)
        self.b = nn.Parameter(torch.zeros((out_c,)), True)


    def forward(self, x):
        y=(conv1dDirection(x[0],self.w,self.b,self.zerow,0),
           conv1dDirection(x[1],self.w,self.b,self.zerow,1),
           conv1dDirection(x[2],self.w,self.b,self.zerow,2),
           conv1dDirection(x[3],self.w,self.b,self.zerow,3),
           )
        return y

class Conv0dResnetBlock(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.conv1=nn.Conv2d(c,c,1,1,0)
        self.conv2=nn.Conv2d(c,c,1,1,0)


    def forward(self, x):
        y=tupleOp(self.conv1,x)
        y=tupleOp(swish,y)
        y=tupleOp(self.conv2,y)
        y=tupleOp(swish,y)
        y=(y[0]+x[0],y[1]+x[1],y[2]+x[2],y[3]+x[3])
        return y

class Conv1dResnetBlock(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.conv1=Conv1dLayer(c,c)
        self.conv2=nn.Conv2d(c,c,1,1,0)


    def forward(self, x):
        y=self.conv1(x)
        y=tupleOp(swish,y)
        y=tupleOp(self.conv2,y)
        y=tupleOp(swish,y)
        y=(y[0]+x[0],y[1]+x[1],y[2]+x[2],y[3]+x[3])
        return y

class channelwiseLeakyRelu(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.c=c
        self.slope = nn.Parameter(torch.ones((c))*0.5,True)
        self.bias = nn.Parameter(torch.zeros((c)),True)


    def forward(self, x):
        dim=len(x.shape)
        if(dim==2):
            wshape = (1,-1)
        elif(dim==3):
            wshape = (1,-1,1)
        elif(dim==4):
            wshape = (1,-1,1,1)
        elif(dim==5):
            wshape = (1,-1,1,1,1)
        return torch.relu(x)*(1-self.slope.view(wshape))+x*self.slope.view(wshape)+self.bias.view(wshape)

class Mapping1(nn.Module):

    def __init__(self,midc,outc):
        super().__init__()
        self.midc=midc
        self.outc=outc
        self.firstConv=Conv1dLayer(2,midc)#len=3
        self.conv1=Conv1dResnetBlock(midc)#len=5
        self.conv2=Conv1dResnetBlock(midc)#len=7
        self.conv3=Conv1dResnetBlock(midc)#len=9
        self.conv4=Conv1dResnetBlock(midc)#len=11
        self.conv5=Conv0dResnetBlock(midc)#len=11
        self.finalconv=nn.Conv2d(midc,outc,1,1,0)

    def forward(self, x):
        x=x[:,0:2]
        y=self.firstConv((x,x,x,x))
        y=tupleOp(swish,y)
        y=self.conv1(y)
        y=self.conv2(y)
        y=self.conv3(y)
        y=self.conv4(y)
        y=self.conv5(y)
        y=tupleOp(self.finalconv,y)
        y=torch.stack(y,dim=1)#shape=(n,4,c,h,w)

        return y

class Model_sum1(nn.Module):

    def __init__(self,midc=128):
        super().__init__()
        self.model_name = "sum1"
        self.model_size = (midc,)

        self.mapping=Mapping1(midc,4)

    def forward(self, x):
        map=self.mapping(x)
        map=map.sum(1)
        p=map[:,0]
        win=map[:,1].mean((1,2))
        loss=map[:,2].mean((1,2))
        draw=map[:,3].mean((1,2))
        v=torch.stack((win,loss,draw),dim=1)
        return v,p

    def exportMapTable(self,x,device): #x.shape=(n=1,c=2,h,w<=9)
        b = (x == 1)
        w = (x == 2)
        x = np.stack((b, w), axis=0).astype(np.float32)

        x=torch.tensor(x[np.newaxis],dtype=torch.float32,device=device)
        with torch.no_grad():
            map = self.mapping(x) #shape=(n=1,4,c=4,h,w<=9)
            return map[0,0].to('cpu') #shape=(c=4,h,w<=9)

class Model_sum1relu(nn.Module):

    def __init__(self,midc=64):
        super().__init__()
        self.model_name = "sum1relu"
        self.model_size = (midc,)

        self.mapping=Mapping1(midc,4)

        self.leakyrelu_slope = nn.Parameter(torch.ones((1,4,1,1))*0.5,True)
        self.leakyrelu_bias = nn.Parameter(torch.zeros((1,4,1,1)),True)

    def forward(self, x):
        map=self.mapping(x)
        map=map.sum(1)
        map=torch.relu(map)*(1-self.leakyrelu_slope)+map*self.leakyrelu_slope+self.leakyrelu_bias
        p=map[:,0]
        win=map[:,1].mean((1,2))
        loss=map[:,2].mean((1,2))
        draw=map[:,3].mean((1,2))
        v=torch.stack((win,loss,draw),dim=1)
        return v,p

    def exportMapTable(self,x,device): #x.shape=(n=1,c=2,h,w<=9)
        b = (x == 1)
        w = (x == 2)
        x = np.stack((b, w), axis=0).astype(np.float32)

        x=torch.tensor(x[np.newaxis],dtype=torch.float32,device=device)
        with torch.no_grad():
            map = self.mapping(x) #shape=(n=1,4,c=4,h,w<=9)
            return map[0,0].to('cpu') #shape=(c=4,h,w<=9)

class Model_sumrelu1(nn.Module):

    def __init__(self,midc=64,pc=2,v1c=2,v2c=2,v3c=2):#1d卷积通道，policy通道，胜负和通道
        super().__init__()
        self.model_name = "sumrelu1"
        self.model_size = (midc,pc,v1c,v2c,v3c)
        self.c=(pc,v1c,v2c,v3c)
        sumc=pc+v1c+v2c+v3c

        self.mapping=Mapping1(midc,sumc)

        self.leakyrelu_slope = nn.Parameter(torch.ones((1,sumc,1,1))*0.5,True)
        self.vbias = nn.Parameter(torch.zeros((3)),True)

    def forward(self, x):
        map=self.mapping(x)
        map=map.sum(1)
        map=torch.relu(map)*(1-self.leakyrelu_slope)+map*self.leakyrelu_slope
        p=map[:,0:self.c[0]].mean((1))
        win=map[:,self.c[0]:self.c[1]+self.c[0]].mean((1,2,3))+self.vbias[0]
        loss=map[:,self.c[0]+self.c[1]:self.c[2]+self.c[1]+self.c[0]].mean((1,2,3))+self.vbias[1]
        draw=map[:,self.c[2]+self.c[1]+self.c[0]:].mean((1,2,3))+self.vbias[2]
        v=torch.stack((win,loss,draw),dim=1)
        return v,p

    def exportMapTable(self,x,device): #x.shape=(n=1,c=2,h,w<=9)
        b = (x == 1)
        w = (x == 2)
        x = np.stack((b, w), axis=0).astype(np.float32)

        x=torch.tensor(x[np.newaxis],dtype=torch.float32,device=device)
        with torch.no_grad():
            map = self.mapping(x) #shape=(n=1,4,c=4,h,w<=9)
            return map[0,0].to('cpu') #shape=(c=4,h,w<=9)

class Model_sumrelu1ep(nn.Module):#空点估值，非空置零

    def __init__(self,midc=64,pc=2,v1c=2,v2c=2,v3c=2):#1d卷积通道，policy通道，胜负和通道
        super().__init__()
        self.model_name = "sumrelu1ep"
        self.model_size = (midc,pc,v1c,v2c,v3c)
        self.c=(pc,v1c,v2c,v3c)
        sumc=pc+v1c+v2c+v3c

        self.mapping=Mapping1(midc,sumc)

        self.leakyrelu_slope = nn.Parameter(torch.ones((1,sumc,1,1))*0.5,True)
        self.vbias = nn.Parameter(torch.zeros((3)),True)

    def forward(self, x):
        emptypoint_mask=1-x[:,0]-x[:,1]
        map=self.mapping(x)*emptypoint_mask.view(-1,1,1,boardH,boardW)
        map=map.sum(1)
        map=torch.relu(map)*(1-self.leakyrelu_slope)+map*self.leakyrelu_slope
        p=map[:,0:self.c[0]].mean((1))-100*(1-emptypoint_mask)
        win=map[:,self.c[0]:self.c[1]+self.c[0]].mean((1,2,3))+self.vbias[0]
        loss=map[:,self.c[0]+self.c[1]:self.c[2]+self.c[1]+self.c[0]].mean((1,2,3))+self.vbias[1]
        draw=map[:,self.c[2]+self.c[1]+self.c[0]:].mean((1,2,3))+self.vbias[2]
        v=torch.stack((win,loss,draw),dim=1)
        return v,p

    def exportMapTable(self,x,device): #x.shape=(n=1,c=2,h,w<=9)
        b = (x == 1)
        w = (x == 2)
        x = np.stack((b, w), axis=0).astype(np.float32)

        x=torch.tensor(x[np.newaxis],dtype=torch.float32,device=device)
        with torch.no_grad():
            map = self.mapping(x) #shape=(n=1,4,c=4,h,w<=9)
            return map[0,0].to('cpu') #shape=(c=4,h,w<=9)

class Model_sumrelu2(nn.Module):

    def __init__(self,midc=64,pc=2,v1c=2,v2c=2,v3c=2):#1d卷积通道，policy通道，胜负和通道
        super().__init__()
        self.model_name = "sumrelu2"
        self.model_size = (midc,pc,v1c,v2c,v3c)
        self.c=(pc,v1c,v2c,v3c)
        sumc=pc+v1c+v2c+v3c

        self.mapping=Mapping1(midc,sumc)

        self.leakyrelu_slope = nn.Parameter(torch.ones((1,sumc,1,1))*0.5,True)
        self.leakyrelu_bias = nn.Parameter(torch.zeros((1,sumc,1,1)),True)
        self.leakyrelu_slope_final = nn.Parameter(torch.ones((4))*0.5,True)
        self.vbias = nn.Parameter(torch.zeros((3)),True)

    def forward(self, x):
        map=self.mapping(x)
        map=map.sum(1)
        map=torch.relu(map)*(1-self.leakyrelu_slope)+map*self.leakyrelu_slope+self.leakyrelu_bias
        p=map[:,0:self.c[0]].mean((1))
        p=torch.relu(p)*(1-self.leakyrelu_slope_final[0])+p*self.leakyrelu_slope_final[0]
        
        win=map[:,self.c[0]:self.c[1]+self.c[0]].mean((1))
        win=torch.relu(win)*(1-self.leakyrelu_slope_final[1])+win*self.leakyrelu_slope_final[1]
        win=win.mean((1,2))+self.vbias[0]
        loss=map[:,self.c[0]+self.c[1]:self.c[2]+self.c[1]+self.c[0]].mean((1))
        loss=torch.relu(loss)*(1-self.leakyrelu_slope_final[2])+loss*self.leakyrelu_slope_final[2]
        loss=loss.mean((1,2))+self.vbias[1]
        draw = map[:, self.c[2] + self.c[1] + self.c[0]:].mean((1))
        draw = torch.relu(draw) * (1 - self.leakyrelu_slope_final[3]) + draw * self.leakyrelu_slope_final[3]
        draw = draw.mean((1, 2)) + self.vbias[2]
        v=torch.stack((win,loss,draw),dim=1)
        return v,p

    def exportMapTable(self,x,device): #x.shape=(n=1,c=2,h,w<=9)
        b = (x == 1)
        w = (x == 2)
        x = np.stack((b, w), axis=0).astype(np.float32)

        x=torch.tensor(x[np.newaxis],dtype=torch.float32,device=device)
        with torch.no_grad():
            map = self.mapping(x) #shape=(n=1,4,c=4,h,w<=9)
            return map[0,0].to('cpu') #shape=(c=4,h,w<=9)

class Model_sumrelu3(nn.Module):

    def __init__(self,midc=64,pc=2,v1c=2,v2c=2,v3c=2):#1d卷积通道，policy通道，胜负和通道
        super().__init__()
        self.model_name = "sumrelu3"
        self.model_size = (midc,pc,v1c,v2c,v3c)
        self.c=(pc,v1c,v2c,v3c)
        sumc=pc+v1c+v2c+v3c

        self.mapping=Mapping1(midc,sumc)

        self.leakyrelu_slope = nn.Parameter(torch.ones((1,1,sumc,1,1))*0.5,True)
        self.vbias = nn.Parameter(torch.zeros((3)),True)

    def forward(self, x):
        map=self.mapping(x)
        map=torch.stack((map[:,0]+map[:,1],map[:,0]+map[:,2],map[:,0]+map[:,3],map[:,1]+map[:,2],map[:,1]+map[:,3],map[:,2]+map[:,3]),dim=1)
        map=torch.relu(map)*(1-self.leakyrelu_slope)+map*self.leakyrelu_slope
        map=map.sum(1)
        p=map[:,0:self.c[0]].mean((1))
        win=map[:,self.c[0]:self.c[1]+self.c[0]].mean((1,2,3))+self.vbias[0]
        loss=map[:,self.c[0]+self.c[1]:self.c[2]+self.c[1]+self.c[0]].mean((1,2,3))+self.vbias[1]
        draw=map[:,self.c[2]+self.c[1]+self.c[0]:].mean((1,2,3))+self.vbias[2]
        v=torch.stack((win,loss,draw),dim=1)
        return v,p

    def exportMapTable(self,x,device): #x.shape=(n=1,c=2,h,w<=9)
        b = (x == 1)
        w = (x == 2)
        x = np.stack((b, w), axis=0).astype(np.float32)

        x=torch.tensor(x[np.newaxis],dtype=torch.float32,device=device)
        with torch.no_grad():
            map = self.mapping(x) #shape=(n=1,4,c=4,h,w<=9)
            return map[0,0].to('cpu') #shape=(c=4,h,w<=9)

class Model_sum2(nn.Module):

    def __init__(self,midc=128):
        super().__init__()
        self.model_name = "sum2"
        self.model_size = (midc,)

        self.mapping=Mapping1(midc,5)

    def forward(self, x):
        map=self.mapping(x)
        map=map.sum(1)
        p=map[:,0]
        win=map[:,1].mean((1,2))
        loss=map[:,2].mean((1,2))
        draw=map[:,3].mean((1,2))
        vfactor=torch.sigmoid(map[:,4].mean((1,2)))
        v=torch.stack((win,loss,draw),dim=1)*vfactor.unsqueeze(1)
        return v,p

class Model_mlp1(nn.Module):

    def __init__(self,midc=128,outc=16,mlpc=8):
        super().__init__()
        self.model_name = "mlp1"
        self.model_size = (midc,outc,mlpc)
        self.outc=outc
        self.mlpc=mlpc


        self.mapping=Mapping1(midc,outc)
        self.mlp_layer1=nn.Conv2d(4*outc,mlpc,1,1,0)
        self.mlp_layer2=nn.Conv2d(mlpc,4,1,1,0)

    def forward(self, x):
        map=self.mapping(x)
        map=map.view(-1,4*self.outc,boardH,boardW)

        y=self.mlp_layer1(map)
        y=swish(y)
        y=self.mlp_layer2(y)

        p=y[:,0]
        win=y[:,1].mean((1,2))
        loss=y[:,2].mean((1,2))
        draw=y[:,3].mean((1,2))
        v=torch.stack((win,loss,draw),dim=1)
        return v,p

class Model_mlp2(nn.Module):

    def __init__(self,midc=128,outc=16,mlpc=8):
        super().__init__()
        self.model_name = "mlp2"
        self.model_size = (midc,outc,mlpc)
        self.outc=outc
        self.mlpc=mlpc


        self.mapping=Mapping1(midc,outc)
        self.mlp_layer1=nn.Conv2d(4*outc,mlpc,1,1,0)
        self.mlp_layer2=nn.Conv2d(mlpc,mlpc,1,1,0)
        self.mlp_layer3=nn.Conv2d(mlpc,4,1,1,0)

    def forward(self, x):
        map=self.mapping(x)
        map=map.view(-1,4*self.outc,boardH,boardW)

        y=self.mlp_layer1(map)
        y=swish(y)
        y=self.mlp_layer2(y)
        y=swish(y)
        y=self.mlp_layer3(y)

        p=y[:,0]
        win=y[:,1].mean((1,2))
        loss=y[:,2].mean((1,2))
        draw=y[:,3].mean((1,2))
        v=torch.stack((win,loss,draw),dim=1)
        return v,p


class Model_summlp1(nn.Module):

    def __init__(self,midc=128,outc=16,mlpc=32):
        super().__init__()
        self.model_name = "summlp1"
        self.model_size = (midc,outc,mlpc)
        self.outc=outc
        self.mlpc=mlpc


        self.mapping=Mapping1(midc,outc)
        self.mlp_layer1=nn.Conv2d(4*outc,mlpc,1,1,0)
        self.phead1=nn.Conv2d(mlpc,1,1,1,0)
        self.mlp_layer2=nn.Linear(mlpc,mlpc)
        self.mlp_layer3=nn.Linear(mlpc,3)

    def forward(self, x):
        map=self.mapping(x)
        map=map.view(-1,4*self.outc,boardH,boardW)

        y=self.mlp_layer1(map)
        y=swish(y)
        p=self.phead1(y)
        v=y.mean((2,3))
        v=self.mlp_layer2(v)
        v=swish(v)
        v=self.mlp_layer3(v)

        return v,p


class Model_mix1(nn.Module):

    def __init__(self,midc=128,outc=16,mlpc=8):
        super().__init__()
        self.model_name = "mix1"
        self.model_size = (midc,outc,mlpc)
        self.outc=outc
        self.mlpc=mlpc


        self.mapping=Mapping1(midc,outc)
        self.mlp_layer1=nn.Conv2d(4*outc,mlpc,1,1,0)
        self.mlp_layer2=nn.Conv2d(mlpc,4,3,padding=1)

    def forward(self, x):
        map=self.mapping(x)
        map=map.view(-1,4*self.outc,boardH,boardW)

        y=self.mlp_layer1(map)
        y=swish(y)
        y=self.mlp_layer2(y)

        p=y[:,0]
        win=y[:,1].mean((1,2))
        loss=y[:,2].mean((1,2))
        draw=y[:,3].mean((1,2))
        v=torch.stack((win,loss,draw),dim=1)
        return v,p

class Model_mix2(nn.Module):

    def __init__(self,midc=128,outc=16,mlpc=8):
        super().__init__()
        self.model_name = "mix2"
        self.model_size = (midc,outc,mlpc)
        self.outc=outc
        self.mlpc=mlpc


        self.mapping=Mapping1(midc,outc)
        self.mlp_layer1=nn.Conv2d(4*outc,mlpc,1,1,0)
        self.phead=nn.Conv2d(mlpc,1,5,padding=2)
        self.mlp_layer2=nn.Linear(mlpc,32)
        self.mlp_layer3=nn.Linear(32,3)

    def forward(self, x):
        map=self.mapping(x)
        map=map.view(-1,4*self.outc,boardH,boardW)

        y=self.mlp_layer1(map)
        y=swish(y)
        p=self.phead(y)
        v=y.mean((2,3))
        v=self.mlp_layer2(v)
        v=swish(v)
        v=self.mlp_layer3(v)

        return v,p

class Model_mix3(nn.Module):

    def __init__(self,midc=64,outc=16,mlpc=32):
        super().__init__()
        self.model_name = "mix3"
        self.model_size = (midc,outc,mlpc)
        self.outc=outc
        self.mlpc=mlpc


        self.mapping=Mapping1(midc,outc)
        self.phead=nn.Conv2d(4*outc,1,5,padding=2)
        self.mlp_layer2=nn.Linear(4*outc,mlpc)
        self.mlp_layer3=nn.Linear(mlpc,3)

    def forward(self, x):
        map=self.mapping(x)
        map=map.view(-1,4*self.outc,boardH,boardW)

        p=self.phead(map)
        v=map.mean((2,3))
        v=self.mlp_layer2(v)
        v=swish(v)
        v=self.mlp_layer3(v)

        return v,p


class Model_mix4noconv(nn.Module):

    def __init__(self, midc=64, pc=6, vc=10):  # 1d卷积通道，policy通道，胜负和通道
        super().__init__()
        self.model_name = "mix4noconv"
        self.model_size = (midc, pc, vc)
        self.pc=pc
        self.vc=vc

        self.mapping = Mapping1(midc, pc+vc)

        self.map_leakyrelu=channelwiseLeakyRelu(pc+vc)
        #self.policy_conv=nn.Conv2d(pc,1,kernel_size=1,padding=0)
        self.value_leakyrelu=channelwiseLeakyRelu(vc)
        self.value_linear=nn.Linear(vc,3)

    def forward(self, x):
        map = self.mapping(x)
        map = map.mean(1)
        map=self.map_leakyrelu(map)
        policy=map[:,:self.pc]
        p=policy.mean(1)
        value=map[:,self.pc:].mean((2,3))
        value=self.value_leakyrelu(value)
        v=self.value_linear(value)
        return v, p

    def exportMapTable(self, x, device):  # x.shape=(n=1,c=2,h,w<=9)
        b = (x == 1)
        w = (x == 2)
        x = np.stack((b, w), axis=0).astype(np.float32)

        x = torch.tensor(x[np.newaxis], dtype=torch.float32, device=device)
        with torch.no_grad():
            map = self.mapping(x)  # shape=(n=1,4,c=4,h,w<=9)
            return map[0, 0].to('cpu')  # shape=(c=4,h,w<=9)


class Model_mix4(nn.Module):

    def __init__(self, midc=64, pc=6, vc=10):  # 1d卷积通道，policy通道，胜负和通道
        super().__init__()
        self.model_name = "mix4"
        self.model_size = (midc, pc, vc)
        self.pc=pc
        self.vc=vc

        self.mapping = Mapping1(midc, pc+vc)

        self.map_leakyrelu=channelwiseLeakyRelu(pc+vc)
        self.policy_conv=nn.Conv2d(pc,1,kernel_size=3,padding=1)
        self.policy_leakyrelu=channelwiseLeakyRelu(1)
        self.value_leakyrelu=channelwiseLeakyRelu(vc)
        self.value_linear1=nn.Linear(vc,vc)
        self.value_linear2=nn.Linear(vc,3)

    def forward(self, x):
        map = self.mapping(x)
        map = map.sum(1)
        map=self.map_leakyrelu(map)
        policy=map[:,:self.pc]
        p=self.policy_conv(policy)
        p=self.policy_leakyrelu(p)
        value=map[:,self.pc:].mean((2,3))
        value=self.value_leakyrelu(value)
        value=self.value_linear1(value)
        value=torch.relu(value)
        v=self.value_linear2(value)
        return v, p

    def exportMapTable(self, x, device):  # x.shape=(n=1,c=2,h,w<=9)
        b = (x == 1)
        w = (x == 2)
        x = np.stack((b, w), axis=0).astype(np.float32)

        x = torch.tensor(x[np.newaxis], dtype=torch.float32, device=device)
        with torch.no_grad():
            map = self.mapping(x)  # shape=(n=1,4,c=pc+vc,h,w<=9)
            return map[0, 0].to('cpu')  # shape=(c=pc+vc,h,w<=9)
class Model_mix4convep(nn.Module):

    def __init__(self, midc=64, pc=6, vc=10):  # 1d卷积通道，policy通道，胜负和通道
        super().__init__()
        self.model_name = "mix4convep"
        self.model_size = (midc, pc, vc)
        self.pc=pc
        self.vc=vc

        self.mapping = Mapping1(midc, pc+vc)
        self.nonempty_map=nn.Parameter(torch.zeros((1,pc+vc,1,1)),True)
        self.map_leakyrelu=channelwiseLeakyRelu(pc+vc)
        self.policy_conv=nn.Conv2d(pc,1,kernel_size=5,padding=2)
        self.policy_leakyrelu=channelwiseLeakyRelu(1)
        self.value_leakyrelu=channelwiseLeakyRelu(vc)
        self.value_linear1=nn.Linear(vc,vc)
        self.value_linear2=nn.Linear(vc,3)

    def forward(self, x):
        hasStone=x[:,0]+x[:,1]
        map = self.mapping(x)
        map = map.mean(1)*(1-hasStone.view(-1,1,boardH,boardW))
        map=self.map_leakyrelu(map)+self.nonempty_map*hasStone.view(-1,1,boardH,boardW)
        policy=map[:,:self.pc]
        p=self.policy_conv(policy)
        p=self.policy_leakyrelu(p)
        value=map[:,self.pc:].mean((2,3))
        value=self.value_leakyrelu(value)
        value=self.value_linear1(value)
        value=torch.relu(value)
        v=self.value_linear2(value)
        return v, p

    def exportMapTable(self, x, device):  # x.shape=(n=1,c=2,h,w<=9)
        b = (x == 1)
        w = (x == 2)
        x = np.stack((b, w), axis=0).astype(np.float32)

        x = torch.tensor(x[np.newaxis], dtype=torch.float32, device=device)
        with torch.no_grad():
            map = self.mapping(x)  # shape=(n=1,4,c=4,h,w<=9)
            return map[0, 0].to('cpu')  # shape=(c=4,h,w<=9)
class Model_mix5(nn.Module):

    def __init__(self, midc=64, pc=6, vc=10):  # 1d卷积通道，policy通道，胜负和通道
        super().__init__()
        self.model_name = "mix5"
        self.model_size = (midc, pc, vc)
        self.pc=pc
        self.vc=vc

        self.mapping = Mapping1(midc, pc+vc)

        self.map_leakyrelu=channelwiseLeakyRelu(vc)
        self.policy_conv=nn.Conv2d(pc*4,1,kernel_size=3,padding=1)
        self.policy_leakyrelu=channelwiseLeakyRelu(1)
        self.value_leakyrelu=channelwiseLeakyRelu(vc)
        self.value_linear1=nn.Linear(vc,vc)
        self.value_linear2=nn.Linear(vc,3)

    def forward(self, x):
        map = self.mapping(x)
        policy=map[:,:,self.pc].view(-1,self.pc*4,boardH,boardW)
        p=self.policy_conv(policy)
        p=self.policy_leakyrelu(p)
        value = map[:,:,self.pc:].sum(1)
        value=self.map_leakyrelu(value)
        value=value.mean((2,3))
        value=self.value_leakyrelu(value)
        value=self.value_linear1(value)
        value=torch.relu(value)
        v=self.value_linear2(value)
        return v, p

    def exportMapTable(self, x, device):  # x.shape=(n=1,c=2,h,w<=9)
        b = (x == 1)
        w = (x == 2)
        x = np.stack((b, w), axis=0).astype(np.float32)

        x = torch.tensor(x[np.newaxis], dtype=torch.float32, device=device)
        with torch.no_grad():
            map = self.mapping(x)  # shape=(n=1,4,c=pc+vc,h,w<=9)
            return map[0, 0].to('cpu')  # shape=(c=pc+vc,h,w<=9)
ModelDic = {
    "sum1": Model_sum1,#基础版
    "sum1relu": Model_sum1relu,#实验
    "sum2": Model_sum2,#实验
    "sumrelu1": Model_sumrelu1, #性价比较高  recommend size=(32,5,1,1,1)
    "sumrelu1ep": Model_sumrelu1ep,
    "sumrelu2": Model_sumrelu2,#实验
    "sumrelu3": Model_sumrelu3,#实验
    "mlp1": Model_mlp1,#实验
    "summlp1": Model_summlp1,#实验
    "mix1": Model_mix1,#实验
    "mix2": Model_mix2,#实验
    "mix3": Model_mix3, #实验
    "mix4noconv": Model_mix4noconv,#实验
    "mix4": Model_mix4, #sumrelu1的改进版，性价比较高
    "mix4convep": Model_mix4convep #实验
}
