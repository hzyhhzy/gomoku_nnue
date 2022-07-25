
import torch
import torch.nn as nn
import numpy as np
from config import *
input_c=3



def swish(x):
    return torch.sigmoid(x)*x
def tupleOp(f,x):
    return (f(x[0]),f(x[1]),f(x[2]),f(x[3]))
def conv1dDirectionOld(x,w,b,zerow,type):

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
def conv1dDirection(x,w,b,zerow,type,L=3,groups=1):

    out_c=w.shape[1]
    in_c=w.shape[2]

    w=torch.concat((zerow.view(1,out_c,in_c),w),dim=0)
    mapping=[0 for i in range(L*L)]
    mid=(L-1)//2
    for i in range(L):
        if(type==0):
            loc=mid*L+i
        elif(type==1):
            loc=mid+i*L
        elif(type==2):
            loc=i*L+i
        elif(type==3):
            loc=-i*L+i+L*(L-1)
        mapping[loc]=i+1

    w=w[mapping]
    w=w.permute(1,2,0)

    w=w.reshape(out_c,in_c,L,L)

    x = torch.conv2d(x,w,b,padding=mid,groups=groups)
    return x


class Conv1dLayerTupleOp(nn.Module):
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
           conv1dDirection(x[3],self.w,self.b,self.zerow,3)
           )
        #print(conv1dDirection(x[3],self.w,self.b,self.zerow,3)-conv1dDirectionOld(x[3],self.w,self.b,self.zerow,3))
        return y

class Conv1dGroupLayerTupleOp(nn.Module):
    def __init__(self, in_c, out_c,L,groups):
        super().__init__()
        self.groups=groups
        self.L=L
        self.zerow = nn.Parameter(torch.zeros((out_c, in_c//groups)),False) #constant zero
        self.w = nn.Parameter(torch.empty((L,out_c, in_c//groups)), True)
        nn.init.kaiming_uniform_(self.w)
        self.b = nn.Parameter(torch.zeros((out_c,)), True)


    def forward(self, x):
        y=(conv1dDirection(x[0],self.w,self.b,self.zerow,0,groups=self.groups,L=self.L),
           conv1dDirection(x[1],self.w,self.b,self.zerow,1,groups=self.groups,L=self.L),
           conv1dDirection(x[2],self.w,self.b,self.zerow,2,groups=self.groups,L=self.L),
           conv1dDirection(x[3],self.w,self.b,self.zerow,3,groups=self.groups,L=self.L)
           )
        return y
#一个方向的conv1d
#只在四条线使用不同映射表的情况下使用。四条线使用相同映射表应该使用Conv1dLayerTupleOp
class Conv1dLayerOneDirection(nn.Module):
    def __init__(self, in_c, out_c,direction):
        super().__init__()
        self.direction=direction
        self.zerow = nn.Parameter(torch.zeros((out_c, in_c)),False) #constant zero
        self.w = nn.Parameter(torch.empty((3,out_c, in_c)), True)
        nn.init.kaiming_uniform_(self.w)
        self.b = nn.Parameter(torch.zeros((out_c,)), True)


    def forward(self, x):
        y=conv1dDirection(x,self.w,self.b,self.zerow,self.direction)
        return y

class Conv0dLayer(nn.Module):
    def __init__(self,in_c,out_c,groups=1):
        super().__init__()
        self.conv=nn.Conv2d(in_c,out_c,1,1,0,groups=groups)
        self.bn=nn.BatchNorm2d(out_c)


    def forward(self, x):
        y=self.conv(x)
        y=self.bn(y)
        y=torch.relu(y)
        return y

class Conv1dLayer(nn.Module):
    def __init__(self,in_c,out_c,groups=1):
        super().__init__()
        self.conv=nn.Conv2d(in_c,out_c,kernel_size=(3,1),padding=(1,0),groups=groups)
        self.bn=nn.BatchNorm2d(out_c)

    def forward(self, x):
        y=self.conv(x)
        y=self.bn(y)
        y=torch.relu(y)
        return y

class Conv0dResnetBlock(nn.Module):
    def __init__(self,c,groups=1):
        super().__init__()
        self.conv1=Conv0dLayer(c,c,groups=groups)
        self.conv2=Conv0dLayer(c,c,groups=groups)


    def forward(self, x):
        y=self.conv1(x)
        y=self.conv2(y)+x
        return y

class Conv1dResnetBlock(nn.Module):
    def __init__(self, c, groups=1):
        super().__init__()
        self.conv1 = Conv1dLayer(c, c, groups=groups)
        self.conv2 = Conv1dLayer(c, c, groups=groups)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y) + x
        return y


class Conv0dResnetBlockTupleOp(nn.Module):
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

class Conv1dResnetBlockOneDirection(nn.Module):
    def __init__(self,c,direction):
        super().__init__()
        self.direction=direction

        self.conv1=Conv1dLayerOneDirection(c,c,direction=direction)
        self.conv2=nn.Conv2d(c,c,1,1,0)


    def forward(self, x):
        y=self.conv1(x)
        y=torch.relu(y)
        y=self.conv2(y)
        y=torch.relu(y)
        y=y+x
        return y


class Conv1dResnetBlockTupleOp(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.conv1=Conv1dLayerTupleOp(c,c)
        self.conv2=nn.Conv2d(c,c,1,1,0)


    def forward(self, x):
        y=self.conv1(x)
        y=tupleOp(swish,y)
        y=tupleOp(self.conv2,y)
        y=tupleOp(swish,y)
        y=(y[0]+x[0],y[1]+x[1],y[2]+x[2],y[3]+x[3])
        return y

class channelwiseLeakyRelu(nn.Module):
    def __init__(self,c,bias=True,bound6=False):
        super().__init__()
        self.c=c
        self.slope = nn.Parameter(torch.ones((c))*0.5,True)
        self.bias = nn.Parameter(torch.zeros((c)),True)
        self.useBias=bias
        self.useBound6=bound6


    def forward(self, x, dim=1):
        xdim=len(x.shape)
        wshape=[1 for i in range(xdim)]
        wshape[dim]=-1

        slope = self.slope.view(wshape)
        if(self.useBound6):
            slope=torch.tanh(self.slope.view(wshape)/6)*6

        res=-torch.relu(-x)*(slope-1)+x
        if(self.useBias):
            res+=self.bias.view(wshape)
        return res

class PRelu1(nn.Module):
    def __init__(self,c,bias=True,bound=0):
        super().__init__()
        self.c=c
        self.slope = nn.Parameter(torch.ones((c))*0.5,True)
        self.bias = nn.Parameter(torch.zeros((c)),True)
        self.useBias=bias
        self.bound=bound


    def forward(self, x, dim=1):
        xdim=len(x.shape)
        wshape=[1 for i in range(xdim)]
        wshape[dim]=-1

        slope = self.slope.view(wshape)
        if(self.bound>0):
            slope=torch.tanh(slope/self.bound)*self.bound

        y=x
        if(self.useBias):
            y=y+self.bias.view(wshape)

        y=torch.maximum(y,slope*y)
        return y
    
class PRelu3a(nn.Module):#三段折线
    def __init__(self,c,bound=0):
        super().__init__()
        self.c=c
        self.slope1 = nn.Parameter(torch.ones((c)),True)
        self.bias1 = nn.Parameter(torch.ones((c))*-1,True)
        self.slope2 = nn.Parameter(torch.ones((c))*0,True)
        self.bias2 = nn.Parameter(torch.ones((c))*0,True)
        self.slope3 = nn.Parameter(torch.ones((c)),True)
        self.bias3 = nn.Parameter(torch.ones((c)),True)
        self.bound=bound


    def forward(self, x, dim=1):
        xdim=len(x.shape)
        wshape=[1 for i in range(xdim)]
        wshape[dim]=-1

        slope1 = self.slope1.view(wshape)
        slope2 = self.slope2.view(wshape)
        slope3 = self.slope3.view(wshape)
        bias1 = self.bias1.view(wshape)
        bias2 = self.bias2.view(wshape)
        bias3 = self.bias3.view(wshape)
        if(self.bound>0):
            slope1=torch.tanh(slope1/self.bound)*self.bound
            slope2=torch.tanh(slope2/self.bound)*self.bound
            slope3=torch.tanh(slope3/self.bound)*self.bound

        y=x

        y=torch.maximum(slope1*y+bias1,slope2*y+bias2)
        y=torch.minimum(y,slope3*y+bias3)
        return y


class PRelu3b(nn.Module):  # 三段折线
    def __init__(self, c, bound=0):
        super().__init__()
        self.c = c
        self.slope1 = nn.Parameter(torch.ones((c)), True)
        self.bias1 = nn.Parameter(torch.ones((c)), True)
        self.slope2 = nn.Parameter(torch.ones((c)) * 0, True)
        self.bias2 = nn.Parameter(torch.ones((c)) * 0, True)
        self.slope3 = nn.Parameter(torch.ones((c)), True)
        self.bias3 = nn.Parameter(torch.ones((c)) * -1, True)
        self.bound = bound

    def forward(self, x, dim=1):
        xdim = len(x.shape)
        wshape = [1 for i in range(xdim)]
        wshape[dim] = -1

        slope1 = self.slope1.view(wshape)
        slope2 = self.slope2.view(wshape)
        slope3 = self.slope3.view(wshape)
        bias1 = self.bias1.view(wshape)
        bias2 = self.bias2.view(wshape)
        bias3 = self.bias3.view(wshape)
        if (self.bound > 0):
            slope1 = torch.tanh(slope1 / self.bound) * self.bound
            slope2 = torch.tanh(slope2 / self.bound) * self.bound
            slope3 = torch.tanh(slope3 / self.bound) * self.bound

        y = x

        y = torch.minimum(slope1 * y + bias1, slope2 * y + bias2)
        y = torch.maximum(y, slope3 * y + bias3)
        return y

class PRelu3c(nn.Module):  # 三段折线,凸
    def __init__(self, c, bound=0):
        super().__init__()
        self.c = c
        self.slope1 = nn.Parameter(torch.ones((c))*0, True)
        self.bias1 = nn.Parameter(torch.ones((c))*-0.5, True)
        self.slope2 = nn.Parameter(torch.ones((c)) * 0.5, True)
        self.bias2 = nn.Parameter(torch.ones((c)) * 0, True)
        self.slope3 = nn.Parameter(torch.ones((c)), True)
        self.bias3 = nn.Parameter(torch.ones((c)) * -0.5, True)
        self.bound = bound

    def forward(self, x, dim=1):
        xdim = len(x.shape)
        wshape = [1 for i in range(xdim)]
        wshape[dim] = -1

        slope1 = self.slope1.view(wshape)
        slope2 = self.slope2.view(wshape)
        slope3 = self.slope3.view(wshape)
        bias1 = self.bias1.view(wshape)
        bias2 = self.bias2.view(wshape)
        bias3 = self.bias3.view(wshape)
        if (self.bound > 0):
            slope1 = torch.tanh(slope1 / self.bound) * self.bound
            slope2 = torch.tanh(slope2 / self.bound) * self.bound
            slope3 = torch.tanh(slope3 / self.bound) * self.bound

        y = x

        y = torch.maximum(slope1 * y + bias1, slope2 * y + bias2)
        y = torch.maximum(y, slope3 * y + bias3)
        return y


class PRelu4a(nn.Module):#四段折线 minmax
    def __init__(self,c,bound=0):
        super().__init__()
        self.c=c
        self.slope1 = nn.Parameter(torch.ones((c))*1,True)
        self.bias1 = nn.Parameter(torch.ones((c))*1,True)
        self.slope2 = nn.Parameter(torch.ones((c))*0,True)
        self.bias2 = nn.Parameter(torch.ones((c))*0,True)
        self.slope3 = nn.Parameter(torch.ones((c))*1,True)
        self.bias3 = nn.Parameter(torch.ones((c))*0,True)
        self.slope4 = nn.Parameter(torch.ones((c))*0,True)
        self.bias4 = nn.Parameter(torch.ones((c))*1,True)
        self.bound=bound


    def forward(self, x, dim=1):
        xdim=len(x.shape)
        wshape=[1 for i in range(xdim)]
        wshape[dim]=-1

        slope1 = self.slope1.view(wshape)
        slope2 = self.slope2.view(wshape)
        slope3 = self.slope3.view(wshape)
        slope4 = self.slope4.view(wshape)
        bias1 = self.bias1.view(wshape)
        bias2 = self.bias2.view(wshape)
        bias3 = self.bias3.view(wshape)
        bias4 = self.bias4.view(wshape)
        if(self.bound>0):
            slope1=torch.tanh(slope1/self.bound)*self.bound
            slope2=torch.tanh(slope2/self.bound)*self.bound
            slope3=torch.tanh(slope3/self.bound)*self.bound
            slope4=torch.tanh(slope4/self.bound)*self.bound


        y=torch.maximum(torch.minimum(slope1*x+bias1,slope2*x+bias2),torch.minimum(slope3*x+bias3,slope4*x+bias4))
        return y

class PRelu4b(nn.Module):#四段折线 maxmin
    def __init__(self,c,bound=0):
        super().__init__()
        self.c=c
        self.slope1 = nn.Parameter(torch.ones((c))*0,True)
        self.bias1 = nn.Parameter(torch.ones((c))*-1,True)
        self.slope2 = nn.Parameter(torch.ones((c))*1,True)
        self.bias2 = nn.Parameter(torch.ones((c))*0,True)
        self.slope3 = nn.Parameter(torch.ones((c))*0,True)
        self.bias3 = nn.Parameter(torch.ones((c))*0,True)
        self.slope4 = nn.Parameter(torch.ones((c))*1,True)
        self.bias4 = nn.Parameter(torch.ones((c))*-1,True)
        self.bound=bound


    def forward(self, x, dim=1):
        xdim=len(x.shape)
        wshape=[1 for i in range(xdim)]
        wshape[dim]=-1

        slope1 = self.slope1.view(wshape)
        slope2 = self.slope2.view(wshape)
        slope3 = self.slope3.view(wshape)
        slope4 = self.slope4.view(wshape)
        bias1 = self.bias1.view(wshape)
        bias2 = self.bias2.view(wshape)
        bias3 = self.bias3.view(wshape)
        bias4 = self.bias4.view(wshape)
        if(self.bound>0):
            slope1=torch.tanh(slope1/self.bound)*self.bound
            slope2=torch.tanh(slope2/self.bound)*self.bound
            slope3=torch.tanh(slope3/self.bound)*self.bound
            slope4=torch.tanh(slope4/self.bound)*self.bound


        y=torch.minimum(torch.maximum(slope1*x+bias1,slope2*x+bias2),torch.maximum(slope3*x+bias3,slope4*x+bias4))
        return y

class Mapping1(nn.Module):

    def __init__(self,midc,outc):
        super().__init__()
        self.midc=midc
        self.outc=outc
        self.firstConv=Conv1dLayerTupleOp(2,midc)#len=3
        self.conv1=Conv1dResnetBlockTupleOp(midc)#len=5
        self.conv2=Conv1dResnetBlockTupleOp(midc)#len=7
        self.conv3=Conv1dResnetBlockTupleOp(midc)#len=9
        self.conv4=Conv1dResnetBlockTupleOp(midc)#len=11
        self.conv5=Conv0dResnetBlockTupleOp(midc)#len=11
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

class Mapping1big(nn.Module):

    def __init__(self,b,midc,outc):
        super().__init__()
        self.midc=midc
        self.outc=outc
        self.firstConv=Conv1dLayerTupleOp(2,midc)#len=3
        self.conv1=Conv1dResnetBlockTupleOp(midc)#len=5
        self.conv2=Conv1dResnetBlockTupleOp(midc)#len=7
        self.conv3=Conv1dResnetBlockTupleOp(midc)#len=9
        self.conv4=Conv1dResnetBlockTupleOp(midc)#len=11

        self.trunk = nn.ModuleList()
        for i in range(b):
            self.trunk.append(Conv0dResnetBlockTupleOp(midc))
        self.finalconv=nn.Conv2d(midc,outc,1,1,0)



    def forward(self, x):
        x=x[:,0:2]
        y=self.firstConv((x,x,x,x))
        y=tupleOp(swish,y)
        y=self.conv1(y)
        y=self.conv2(y)
        y=self.conv3(y)
        y=self.conv4(y)
        for block in self.trunk:
            y = block(y)
        y=tupleOp(self.finalconv,y)
        y=torch.stack(y,dim=1)#shape=(n,4,c,h,w)

        return y


class Mapping1OneDirection(nn.Module):

    def __init__(self,b,midc,outc,direction):
        super().__init__()
        self.direction=direction
        self.midc=midc
        self.outc=outc
        self.firstConv=Conv1dLayerOneDirection(2,midc,direction=direction)#len=3
        self.conv1=Conv1dResnetBlockOneDirection(midc,direction=direction)#len=5
        self.conv2=Conv1dResnetBlockOneDirection(midc,direction=direction)#len=7
        self.conv3=Conv1dResnetBlockOneDirection(midc,direction=direction)#len=9
        self.conv4=Conv1dResnetBlockOneDirection(midc,direction=direction)#len=11

        self.trunk = nn.ModuleList()
        for i in range(b):
            self.trunk.append(Conv0dResnetBlock(midc))
        self.finalconv=nn.Conv2d(midc,outc,1,1,0)



    def forward(self, x):
        y=x[:,0:2]
        y=self.firstConv(y)
        y=torch.relu(y)
        y=self.conv1(y)
        y=self.conv2(y)
        y=self.conv3(y)
        y=self.conv4(y)
        for block in self.trunk:
            y = block(y)
        y=self.finalconv(y)

        return y

#4条线不同的映射表
class Mapping1dif(nn.Module):

    def __init__(self,b,midc,outc):
        super().__init__()

        self.map0=Mapping1OneDirection(b,midc,outc,0)
        self.map1=Mapping1OneDirection(b,midc,outc,1)
        self.map2=Mapping1OneDirection(b,midc,outc,2)
        self.map3=Mapping1OneDirection(b,midc,outc,3)



    def forward(self, x):
        y=(self.map0(x),self.map1(x),self.map2(x),self.map3(x))
        y=torch.stack(y,dim=1)#shape=(n,4,c,h,w)

        return y

#经实验，先提取每条线的特征，效率低下。因此还是使用旧的方法
class LineExtractConv(nn.Module):#将棋盘转化为每条线的特征

    def __init__(self):
        super().__init__()
        #self.conv=nn.Conv2d(3,4*3*11,(11,11),padding=5)
        #self.conv.bias.data=torch.zeros(self.conv.bias.data.shape)
        self.convweight=torch.zeros((3*4*11,3,11,11),requires_grad=False) #132,3,11,11

        d=0 # +x direction
        for color in range(3):
            for x in range(11):
                self.convweight[d*3*11+color*11+x,color,5,x]=1

        d=1 # +y direction
        for color in range(3):
            for x in range(11):
                self.convweight[d*3*11+color*11+x,color,x,5]=1

        d=2 # +x+y direction
        for color in range(3):
            for x in range(11):
                self.convweight[d*3*11+color*11+x,color,x,x]=1

        d=3 # +x-y direction
        for color in range(3):
            for x in range(11):
                self.convweight[d*3*11+color*11+x,color,10-x,x]=1


    def forward(self, x):
        if(self.convweight.device != x.device):
            self.convweight=self.convweight.to(x.device)
        y=x[:,0:2]
        y=torch.concat((y,torch.ones((y.shape[0],1,y.shape[2],y.shape[3]),device=y.device)), dim=1) #N3HW, 3 means me,opp,onboard
        y=torch.conv2d(x,self.convweight,bias=None,padding=5) #shape=(N,4*3*11,H,W)

        return y


#mlp形式的映射表
class Mapping2(nn.Module):

    def __init__(self,b,f,out_c):
        super().__init__()
        self.out_c=out_c
        self.lineExtractConv=LineExtractConv()
        self.firstLinear=Conv0dLayer(33,f)
        self.trunk=nn.ModuleList()
        for i in range(b):
            self.trunk.append(Conv0dResnetBlock(f))
        self.lastLinear=nn.Conv2d(f,out_c,1,1,0,groups=1)

    def forward(self, x):
        N=x.shape[0]
        y=self.lineExtractConv(x) #shape=(N,4*3*11,H,W)
        y=y.reshape((N*4,3*11,boardH,boardW)) #shape=(N*4,3*11,H,W)

        y=self.firstLinear(y)
        for block in self.trunk:
            y=block(y)
        y=self.lastLinear(y)  #shape=(N*4,out_c,H,W)

        y=y.reshape((N,4,self.out_c,boardH,boardW))

        return y

#使用1d卷积，几乎等价于Mapping1
class Mapping2c(nn.Module):

    def __init__(self,b,f,out_c):
        super().__init__()
        self.f=f
        self.out_c=out_c
        self.lineExtractConv=LineExtractConv()
        self.firstLinear=Conv1dLayer(3,f)
        self.trunk=nn.ModuleList()
        for i in range(b):
            self.trunk.append(Conv1dResnetBlock(f))
        self.lastLinear=nn.Conv2d(f*11,out_c,1,1,0,groups=1)

    def forward(self, x):
        N=x.shape[0]
        y=self.lineExtractConv(x) #shape=(N,4*3*11,H,W)
        y=y.reshape((N*4,3,11,boardH*boardW)) #shape=(N*4,3,11,H*W), conv on "11" axis

        y=self.firstLinear(y)
        for block in self.trunk:
            y=block(y)
        y=y.reshape((N*4,self.f*11,boardH,boardW)) #shape=(N*4,f*11,H,W)
        y=self.lastLinear(y)  #shape=(N*4,out_c,H,W)

        y=y.reshape((N,4,self.out_c,boardH,boardW))

        return y

#四条线采用不同的映射表
class Mapping3(nn.Module):

    def __init__(self,b,f,out_c):
        super().__init__()
        self.out_c=out_c
        self.lineExtractConv=LineExtractConv()
        self.firstLinear=Conv0dLayer(4*33,4*f,groups=4)
        self.trunk=nn.ModuleList()
        for i in range(b):
            self.trunk.append(Conv0dResnetBlock(4*f,groups=4))
        self.lastLinear=nn.Conv2d(4*f,4*out_c,1,1,0,groups=4)

    def forward(self, x):
        N=x.shape[0]
        y=self.lineExtractConv(x) #shape=(N,4*3*11,H,W)
        y=y.reshape((N,4*3*11,boardH,boardW)) #shape=(N,4*3*11,H,W) (not changed)

        y=self.firstLinear(y)
        for block in self.trunk:
            y=block(y)
        y=self.lastLinear(y)  #shape=(N,4*out_c,H,W)

        y=y.reshape((N,4,self.out_c,boardH,boardW))

        return y



class Model_sum1(nn.Module):

    def __init__(self,midc=128):
        super().__init__()
        self.model_type = "sum1"
        self.model_param = (midc,)

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
        self.model_type = "sum1relu"
        self.model_param = (midc,)

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
        self.model_type = "sumrelu1"
        self.model_param = (midc,pc,v1c,v2c,v3c)
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
        self.model_type = "sumrelu1ep"
        self.model_param = (midc,pc,v1c,v2c,v3c)
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
        self.model_type = "sumrelu2"
        self.model_param = (midc,pc,v1c,v2c,v3c)
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
        self.model_type = "sumrelu3"
        self.model_param = (midc,pc,v1c,v2c,v3c)
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
        self.model_type = "sum2"
        self.model_param = (midc,)

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
        self.model_type = "mlp1"
        self.model_param = (midc,outc,mlpc)
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
        self.model_type = "mlp2"
        self.model_param = (midc,outc,mlpc)
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
        self.model_type = "summlp1"
        self.model_param = (midc,outc,mlpc)
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
        self.model_type = "mix1"
        self.model_param = (midc,outc,mlpc)
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
        self.model_type = "mix2"
        self.model_param = (midc,outc,mlpc)
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
        self.model_type = "mix3"
        self.model_param = (midc,outc,mlpc)
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
        self.model_type = "mix4noconv"
        self.model_param = (midc, pc, vc)
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

    def __init__(self, midc=64, pc=8, vc=16):  # 1d卷积通道，policy通道，胜负和通道
        super().__init__()
        self.model_type = "mix4"
        self.model_param = (midc, pc, vc)
        self.pc=pc
        self.vc=vc

        self.mapping = Mapping1(midc, pc+vc)

        self.map_leakyrelu=channelwiseLeakyRelu(pc+vc)
        self.policy_conv=nn.Conv2d(pc,1,kernel_size=3,padding=1,bias=False)
        self.policy_leakyrelu=channelwiseLeakyRelu(1,bias=False)
        self.value_leakyrelu=channelwiseLeakyRelu(vc,bias=False)
        self.value_linear1=nn.Linear(vc,vc)
        self.value_linear2=nn.Linear(vc,vc)
        self.value_linearfinal=nn.Linear(vc,3)

    def forward(self, x):
        map = self.mapping(x)
        map=torch.relu(map+30)-30
        map=30-torch.relu(30-map)
        # |map|<30

        map = map.mean(1) # |map|<30
        map=self.map_leakyrelu(map) # |map|<300 if slope<10
        p=map[:,:self.pc]
        p=self.policy_conv(p)
        p=self.policy_leakyrelu(p)
        v=map[:,self.pc:].mean((2,3))
        v0=self.value_leakyrelu(v)
        v=self.value_linear1(v0)
        v=torch.relu(v)
        v=self.value_linear2(v)
        v=torch.relu(v)+v0
        v=self.value_linearfinal(v)
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
        self.model_type = "mix4convep"
        self.model_param = (midc, pc, vc)
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
        self.model_type = "mix5"
        self.model_param = (midc, pc, vc)
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


class Model_mix6(nn.Module):

    def __init__(self, midc=128, pc=16, vc=32, mapmax=30):  # 1d卷积通道，policy通道，胜负和通道
        super().__init__()
        self.model_type = "mix6"
        self.model_param = (midc, pc, vc, mapmax)
        self.pc=pc
        self.vc=vc

        self.mapmax=mapmax

        self.mapping = Mapping1(midc, pc+vc)
        self.map_leakyrelu=channelwiseLeakyRelu(pc+vc,bound6=True)
        self.policy_conv=nn.Conv2d(pc,pc,kernel_size=3,padding=1,bias=True,groups=pc)
        self.policy_linear=nn.Conv2d(pc,1,kernel_size=1,padding=0,bias=False)
        self.policy_leakyrelu=channelwiseLeakyRelu(1,bias=False)
        self.value_leakyrelu=channelwiseLeakyRelu(vc,bias=False)
        self.value_linear1=nn.Linear(vc,vc)
        self.value_linear2=nn.Linear(vc,vc)
        self.value_linearfinal=nn.Linear(vc,3)

    def forward(self, x):
        map = self.mapping(x)


        if(self.mapmax!=0):
            map=self.mapmax*torch.tanh(map/self.mapmax) # |map|<30

        map = map.mean(1) # |map|<30
        map=self.map_leakyrelu(map) # |map|<300 if slope<10
        p=map[:,:self.pc]
        p=self.policy_conv(p)
        p=torch.relu(p)
        p=self.policy_linear(p)
        p=self.policy_leakyrelu(p)
        v=map[:,self.pc:].mean((2,3))
        v0=self.value_leakyrelu(v)
        v=self.value_linear1(v0)
        v=torch.relu(v)
        v=self.value_linear2(v)
        v=torch.relu(v)+v0
        v=self.value_linearfinal(v)
        return v, p

    def testeval(self, x,loc):
        y0=loc//boardW
        x0=loc%boardW
        map = self.mapping(x)

        if(self.mapmax!=0):
            map=self.mapmax*torch.tanh(map/self.mapmax) # |map|<30

        w_scale = 200
        scale_now = w_scale  # 200
        print("map ",map[0,:,:,y0,x0]*scale_now)

        map = map.mean(1) # |map|<30
        w_scale=1
        scale_now*=w_scale #200
        print("mapsum ",map[0,:,y0,x0]*scale_now*4)
        map=self.map_leakyrelu(map) # |map|<300 if slope<10
        print("maplr ",map[0,:,y0,x0]*scale_now)
        p=map[:,:self.pc]
        p=self.policy_conv(p)
        w_scale=0.5
        scale_now*=w_scale #100
        print("pconv ",p[0,:,y0,x0]*scale_now)
        p=torch.relu(p)
        p=self.policy_linear(p)
        print("psum ",p[0,:,y0,x0])
        p=self.policy_leakyrelu(p)
        print("p \n",(p*32).long())
        v=map[:,self.pc:].mean((2,3))
        print("vmean ",v)
        v0=self.value_leakyrelu(v)
        print("vlayer0 ",v0)
        v=self.value_linear1(v0)
        print("vlayer1 ",v)
        v=torch.relu(v)
        v=self.value_linear2(v)
        print("vlayer2 ",v)
        v=torch.relu(v)+v0
        v=self.value_linearfinal(v)
        print("v ",v)
        print("vsoftmax ",torch.softmax(v,dim=1))
        return v, p
    def exportMapTable(self, x, device):  # x.shape=(h,w<=9)
        b = (x == 1)
        w = (x == 2)
        x = np.stack((b, w), axis=0).astype(np.float32)

        x = torch.tensor(x[np.newaxis], dtype=torch.float32, device=device)# x.shape=(n=1,c=2,h,w<=9)

        with torch.no_grad():
            batchsize=4096
            batchnum=1+(x.shape[2]-1)//batchsize
            buf=[]
            for i in range(batchnum):
                start=i*batchsize
                end=min((i+1)*batchsize,x.shape[2])

                map=self.mapping(x[:, :, start:end, :])[0, 0]
                if(self.mapmax!=0):
                    map=self.mapmax*torch.tanh(map/self.mapmax) # |map|<30

                buf.append(map.to('cpu').numpy())  # map.shape=(n=1,4,c=pc+vc,h,w<=9)
            buf=np.concatenate(buf,axis=1)
            return buf # shape=(c=pc+vc,h,w<=9)

class Model_mix6m1big(nn.Module):

    def __init__(self, midb=5, midc=128, pc=16, vc=32, mapmax=30):  # 1d卷积通道，policy通道，胜负和通道
        super().__init__()
        self.model_type = "mix6m1big"
        self.model_param = (midc, pc, vc, mapmax)
        self.pc=pc
        self.vc=vc

        self.mapmax=mapmax

        self.mapping = Mapping1big(midb, midc, pc+vc)
        self.map_leakyrelu=channelwiseLeakyRelu(pc+vc,bound6=True)
        self.policy_conv=nn.Conv2d(pc,pc,kernel_size=3,padding=1,bias=True,groups=pc)
        self.policy_linear=nn.Conv2d(pc,1,kernel_size=1,padding=0,bias=False)
        self.policy_leakyrelu=channelwiseLeakyRelu(1,bias=False)
        self.value_leakyrelu=channelwiseLeakyRelu(vc,bias=False)
        self.value_linear1=nn.Linear(vc,vc)
        self.value_linear2=nn.Linear(vc,vc)
        self.value_linearfinal=nn.Linear(vc,3)

    def forward(self, x):
        map = self.mapping(x)


        if(self.mapmax!=0):
            map=self.mapmax*torch.tanh(map/self.mapmax) # |map|<30

        map = map.mean(1) # |map|<30
        map=self.map_leakyrelu(map) # |map|<300 if slope<10
        p=map[:,:self.pc]
        p=self.policy_conv(p)
        p=torch.relu(p)
        p=self.policy_linear(p)
        p=self.policy_leakyrelu(p)
        v=map[:,self.pc:].mean((2,3))
        v0=self.value_leakyrelu(v)
        v=self.value_linear1(v0)
        v=torch.relu(v)
        v=self.value_linear2(v)
        v=torch.relu(v)+v0
        v=self.value_linearfinal(v)
        return v, p

    def testeval(self, x,loc):
        y0=loc//boardW
        x0=loc%boardW
        map = self.mapping(x)

        if(self.mapmax!=0):
            map=self.mapmax*torch.tanh(map/self.mapmax) # |map|<30

        w_scale = 200
        scale_now = w_scale  # 200
        print("map ",map[0,:,:,y0,x0]*scale_now)

        map = map.mean(1) # |map|<30
        w_scale=1
        scale_now*=w_scale #200
        print("mapsum ",map[0,:,y0,x0]*scale_now*4)
        map=self.map_leakyrelu(map) # |map|<300 if slope<10
        print("maplr ",map[0,:,y0,x0]*scale_now)
        p=map[:,:self.pc]
        p=self.policy_conv(p)
        w_scale=0.5
        scale_now*=w_scale #100
        print("pconv ",p[0,:,y0,x0]*scale_now)
        p=torch.relu(p)
        p=self.policy_linear(p)
        print("psum ",p[0,:,y0,x0])
        p=self.policy_leakyrelu(p)
        print("p \n",(p*32).long())
        v=map[:,self.pc:].mean((2,3))
        print("vmean ",v)
        v0=self.value_leakyrelu(v)
        print("vlayer0 ",v0)
        v=self.value_linear1(v0)
        print("vlayer1 ",v)
        v=torch.relu(v)
        v=self.value_linear2(v)
        print("vlayer2 ",v)
        v=torch.relu(v)+v0
        v=self.value_linearfinal(v)
        print("v ",v)
        print("vsoftmax ",torch.softmax(v,dim=1))
        return v, p
    def exportMapTable(self, x, device):  # x.shape=(h,w<=9)
        b = (x == 1)
        w = (x == 2)
        x = np.stack((b, w), axis=0).astype(np.float32)

        x = torch.tensor(x[np.newaxis], dtype=torch.float32, device=device)# x.shape=(n=1,c=2,h,w<=9)

        with torch.no_grad():
            batchsize=4096
            batchnum=1+(x.shape[2]-1)//batchsize
            buf=[]
            for i in range(batchnum):
                start=i*batchsize
                end=min((i+1)*batchsize,x.shape[2])

                map=self.mapping(x[:, :, start:end, :])[0, 0]
                if(self.mapmax!=0):
                    map=self.mapmax*torch.tanh(map/self.mapmax) # |map|<30

                buf.append(map.to('cpu').numpy())  # map.shape=(n=1,4,c=pc+vc,h,w<=9)
            buf=np.concatenate(buf,axis=1)
            return buf # shape=(c=pc+vc,h,w<=9)
#不同方向使用不同映射表
class Model_mix6m1dif(nn.Module):

    def __init__(self, midb=5, midc=128, pc=16, vc=32, mapmax=30):  # 1d卷积通道，policy通道，胜负和通道
        super().__init__()
        self.model_type = "mix6m1big"
        self.model_param = (midc, pc, vc, mapmax)
        self.pc=pc
        self.vc=vc

        self.mapmax=mapmax

        self.mapping = Mapping1dif(midb, midc, pc+vc)
        self.map_leakyrelu=channelwiseLeakyRelu(pc+vc,bound6=True)
        self.policy_conv=nn.Conv2d(pc,pc,kernel_size=3,padding=1,bias=True,groups=pc)
        self.policy_linear=nn.Conv2d(pc,1,kernel_size=1,padding=0,bias=False)
        self.policy_leakyrelu=channelwiseLeakyRelu(1,bias=False)
        self.value_leakyrelu=channelwiseLeakyRelu(vc,bias=False)
        self.value_linear1=nn.Linear(vc,vc)
        self.value_linear2=nn.Linear(vc,vc)
        self.value_linearfinal=nn.Linear(vc,3)

    def forward(self, x):
        map = self.mapping(x)


        if(self.mapmax!=0):
            map=self.mapmax*torch.tanh(map/self.mapmax) # |map|<30

        map = map.mean(1) # |map|<30
        map=self.map_leakyrelu(map) # |map|<300 if slope<10
        p=map[:,:self.pc]
        p=self.policy_conv(p)
        p=torch.relu(p)
        p=self.policy_linear(p)
        p=self.policy_leakyrelu(p)
        v=map[:,self.pc:].mean((2,3))
        v0=self.value_leakyrelu(v)
        v=self.value_linear1(v0)
        v=torch.relu(v)
        v=self.value_linear2(v)
        v=torch.relu(v)+v0
        v=self.value_linearfinal(v)
        return v, p

    def testeval(self, x,loc):
        y0=loc//boardW
        x0=loc%boardW
        map = self.mapping(x)

        if(self.mapmax!=0):
            map=self.mapmax*torch.tanh(map/self.mapmax) # |map|<30

        w_scale = 200
        scale_now = w_scale  # 200
        print("map ",map[0,:,:,y0,x0]*scale_now)

        map = map.mean(1) # |map|<30
        w_scale=1
        scale_now*=w_scale #200
        print("mapsum ",map[0,:,y0,x0]*scale_now*4)
        map=self.map_leakyrelu(map) # |map|<300 if slope<10
        print("maplr ",map[0,:,y0,x0]*scale_now)
        p=map[:,:self.pc]
        p=self.policy_conv(p)
        w_scale=0.5
        scale_now*=w_scale #100
        print("pconv ",p[0,:,y0,x0]*scale_now)
        p=torch.relu(p)
        p=self.policy_linear(p)
        print("psum ",p[0,:,y0,x0])
        p=self.policy_leakyrelu(p)
        print("p \n",(p*32).long())
        v=map[:,self.pc:].mean((2,3))
        print("vmean ",v)
        v0=self.value_leakyrelu(v)
        print("vlayer0 ",v0)
        v=self.value_linear1(v0)
        print("vlayer1 ",v)
        v=torch.relu(v)
        v=self.value_linear2(v)
        print("vlayer2 ",v)
        v=torch.relu(v)+v0
        v=self.value_linearfinal(v)
        print("v ",v)
        print("vsoftmax ",torch.softmax(v,dim=1))
        return v, p
    def exportMapTable(self, x, device):  # x.shape=(h,w<=9)
        b = (x == 1)
        w = (x == 2)
        x = np.stack((b, w), axis=0).astype(np.float32)

        x = torch.tensor(x[np.newaxis], dtype=torch.float32, device=device)# x.shape=(n=1,c=2,h,w<=9)

        with torch.no_grad():
            batchsize=4096
            batchnum=1+(x.shape[2]-1)//batchsize
            buf=[]
            for i in range(batchnum):
                start=i*batchsize
                end=min((i+1)*batchsize,x.shape[2])

                map=self.mapping(x[:, :, start:end, :])[0, 0]
                if(self.mapmax!=0):
                    map=self.mapmax*torch.tanh(map/self.mapmax) # |map|<30

                buf.append(map.to('cpu').numpy())  # map.shape=(n=1,4,c=pc+vc,h,w<=9)
            buf=np.concatenate(buf,axis=1)
            return buf # shape=(c=pc+vc,h,w<=9)

class Model_mix6m2(nn.Module):
    #using Mapping2
    def __init__(self, b, f, pc=16, vc=32, mapmax=30):  # 1d卷积通道，policy通道，胜负和通道
        super().__init__()
        self.model_type = "mix6m2"
        self.model_param = (b, f, pc, vc, mapmax)
        self.pc=pc
        self.vc=vc

        self.mapmax=mapmax

        self.mapping = Mapping2(b, f, pc+vc)
        self.map_leakyrelu=channelwiseLeakyRelu(pc+vc,bound6=True)
        self.policy_conv=nn.Conv2d(pc,pc,kernel_size=3,padding=1,bias=True,groups=pc)
        self.policy_linear=nn.Conv2d(pc,1,kernel_size=1,padding=0,bias=False)
        self.policy_leakyrelu=channelwiseLeakyRelu(1,bias=False)
        self.value_leakyrelu=channelwiseLeakyRelu(vc,bias=False)
        self.value_linear1=nn.Linear(vc,vc)
        self.value_linear2=nn.Linear(vc,vc)
        self.value_linearfinal=nn.Linear(vc,3)

    def forward(self, x):
        map = self.mapping(x)


        if(self.mapmax!=0):
            map=self.mapmax*torch.tanh(map/self.mapmax) # |map|<30

        map = map.mean(1) # |map|<30
        map=self.map_leakyrelu(map) # |map|<300 if slope<10
        p=map[:,:self.pc]
        p=self.policy_conv(p)
        p=torch.relu(p)
        p=self.policy_linear(p)
        p=self.policy_leakyrelu(p)
        v=map[:,self.pc:].mean((2,3))
        v0=self.value_leakyrelu(v)
        v=self.value_linear1(v0)
        v=torch.relu(v)
        v=self.value_linear2(v)
        v=torch.relu(v)+v0
        v=self.value_linearfinal(v)
        return v, p

    def testeval(self, x,loc):
        y0=loc//boardW
        x0=loc%boardW
        map = self.mapping(x)

        if(self.mapmax!=0):
            map=self.mapmax*torch.tanh(map/self.mapmax) # |map|<30

        w_scale = 200
        scale_now = w_scale  # 200
        print("map ",map[0,:,:,y0,x0]*scale_now)

        map = map.mean(1) # |map|<30
        w_scale=1
        scale_now*=w_scale #200
        print("mapsum ",map[0,:,y0,x0]*scale_now*4)
        map=self.map_leakyrelu(map) # |map|<300 if slope<10
        print("maplr ",map[0,:,y0,x0]*scale_now)
        p=map[:,:self.pc]
        p=self.policy_conv(p)
        w_scale=0.5
        scale_now*=w_scale #100
        print("pconv ",p[0,:,y0,x0]*scale_now)
        p=torch.relu(p)
        p=self.policy_linear(p)
        print("psum ",p[0,:,y0,x0])
        p=self.policy_leakyrelu(p)
        print("p \n",(p*32).long())
        v=map[:,self.pc:].mean((2,3))
        print("vmean ",v)
        v0=self.value_leakyrelu(v)
        print("vlayer0 ",v0)
        v=self.value_linear1(v0)
        print("vlayer1 ",v)
        v=torch.relu(v)
        v=self.value_linear2(v)
        print("vlayer2 ",v)
        v=torch.relu(v)+v0
        v=self.value_linearfinal(v)
        print("v ",v)
        print("vsoftmax ",torch.softmax(v,dim=1))
        return v, p
    def exportMapTable(self, x, device):  # x.shape=(h,w<=9)
        b = (x == 1)
        w = (x == 2)
        x = np.stack((b, w), axis=0).astype(np.float32)

        x = torch.tensor(x[np.newaxis], dtype=torch.float32, device=device)# x.shape=(n=1,c=2,h,w<=9)

        with torch.no_grad():
            batchsize=4096
            batchnum=1+(x.shape[2]-1)//batchsize
            buf=[]
            for i in range(batchnum):
                start=i*batchsize
                end=min((i+1)*batchsize,x.shape[2])

                map=self.mapping(x[:, :, start:end, :])[0, 0]
                if(self.mapmax!=0):
                    map=self.mapmax*torch.tanh(map/self.mapmax) # |map|<30

                buf.append(map.to('cpu').numpy())  # map.shape=(n=1,4,c=pc+vc,h,w<=9)
            buf=np.concatenate(buf,axis=1)
            return buf # shape=(c=pc+vc,h,w<=9)

class Model_mix6m2c(nn.Module):
    #using Mapping2c
    def __init__(self, b, f, pc=16, vc=32, mapmax=30):  # 1d卷积通道，policy通道，胜负和通道
        super().__init__()
        self.model_type = "mix6m2c"
        self.model_param = (b, f, pc, vc, mapmax)
        self.pc=pc
        self.vc=vc

        self.mapmax=mapmax

        self.mapping = Mapping2c(b, f, pc+vc)
        self.map_leakyrelu=channelwiseLeakyRelu(pc+vc,bound6=True)
        self.policy_conv=nn.Conv2d(pc,pc,kernel_size=3,padding=1,bias=True,groups=pc)
        self.policy_linear=nn.Conv2d(pc,1,kernel_size=1,padding=0,bias=False)
        self.policy_leakyrelu=channelwiseLeakyRelu(1,bias=False)
        self.value_leakyrelu=channelwiseLeakyRelu(vc,bias=False)
        self.value_linear1=nn.Linear(vc,vc)
        self.value_linear2=nn.Linear(vc,vc)
        self.value_linearfinal=nn.Linear(vc,3)

    def forward(self, x):
        map = self.mapping(x)


        if(self.mapmax!=0):
            map=self.mapmax*torch.tanh(map/self.mapmax) # |map|<30

        map = map.mean(1) # |map|<30
        map=self.map_leakyrelu(map) # |map|<300 if slope<10
        p=map[:,:self.pc]
        p=self.policy_conv(p)
        p=torch.relu(p)
        p=self.policy_linear(p)
        p=self.policy_leakyrelu(p)
        v=map[:,self.pc:].mean((2,3))
        v0=self.value_leakyrelu(v)
        v=self.value_linear1(v0)
        v=torch.relu(v)
        v=self.value_linear2(v)
        v=torch.relu(v)+v0
        v=self.value_linearfinal(v)
        return v, p


class Model_mix6m3(nn.Module):
    #using Mapping3
    #different directions use different mapping
    def __init__(self, b, f, pc=16, vc=32, mapmax=30):  # 1d卷积通道，policy通道，胜负和通道
        super().__init__()
        self.model_type = "mix6m3"
        self.model_param = (b, f, pc, vc, mapmax)
        self.pc=pc
        self.vc=vc

        self.mapmax=mapmax

        self.mapping = Mapping3(b, f, pc+vc)
        self.map_leakyrelu=channelwiseLeakyRelu(pc+vc,bound6=True)
        self.policy_conv=nn.Conv2d(pc,pc,kernel_size=3,padding=1,bias=True,groups=pc)
        self.policy_linear=nn.Conv2d(pc,1,kernel_size=1,padding=0,bias=False)
        self.policy_leakyrelu=channelwiseLeakyRelu(1,bias=False)
        self.value_leakyrelu=channelwiseLeakyRelu(vc,bias=False)
        self.value_linear1=nn.Linear(vc,vc)
        self.value_linear2=nn.Linear(vc,vc)
        self.value_linearfinal=nn.Linear(vc,3)

    def forward(self, x):
        map = self.mapping(x)


        if(self.mapmax!=0):
            map=self.mapmax*torch.tanh(map/self.mapmax) # |map|<30

        map = map.mean(1) # |map|<30
        map=self.map_leakyrelu(map) # |map|<300 if slope<10
        p=map[:,:self.pc]
        p=self.policy_conv(p)
        p=torch.relu(p)
        p=self.policy_linear(p)
        p=self.policy_leakyrelu(p)
        v=map[:,self.pc:].mean((2,3))
        v0=self.value_leakyrelu(v)
        v=self.value_linear1(v0)
        v=torch.relu(v)
        v=self.value_linear2(v)
        v=torch.relu(v)+v0
        v=self.value_linearfinal(v)
        return v, p

class Model_mix6m3conv(nn.Module):
    #可调节最后policy卷积的mix6m3
    #using Mapping3
    #different directions use different mapping
    def __init__(self, b, f, kernelsize, groups, pc=16, vc=32, mapmax=30):  # 1d卷积通道，policy通道，胜负和通道
        super().__init__()
        self.model_type = "mix6m3conv"
        self.model_param = (b, f, kernelsize, groups, pc, vc, mapmax)
        self.pc=pc
        self.vc=vc

        self.mapmax=mapmax

        self.mapping = Mapping3(b, f, pc+vc)
        self.map_leakyrelu=channelwiseLeakyRelu(pc+vc,bound6=True)
        self.policy_conv=nn.Conv2d(pc,pc,kernel_size=kernelsize,padding=(kernelsize-1)//2,bias=True,groups=groups)
        self.policy_linear=nn.Conv2d(pc,1,kernel_size=1,padding=0,bias=False)
        self.policy_leakyrelu=channelwiseLeakyRelu(1,bias=False)
        self.value_leakyrelu=channelwiseLeakyRelu(vc,bias=False)
        self.value_linear1=nn.Linear(vc,vc)
        self.value_linear2=nn.Linear(vc,vc)
        self.value_linearfinal=nn.Linear(vc,3)

    def forward(self, x):
        map = self.mapping(x)


        if(self.mapmax!=0):
            map=self.mapmax*torch.tanh(map/self.mapmax) # |map|<30

        map = map.mean(1) # |map|<30
        map=self.map_leakyrelu(map) # |map|<300 if slope<10
        p=map[:,:self.pc]
        p=self.policy_conv(p)
        p=torch.relu(p)
        p=self.policy_linear(p)
        p=self.policy_leakyrelu(p)
        v=map[:,self.pc:].mean((2,3))
        v0=self.value_leakyrelu(v)
        v=self.value_linear1(v0)
        v=torch.relu(v)
        v=self.value_linear2(v)
        v=torch.relu(v)+v0
        v=self.value_linearfinal(v)
        return v, p


class CNNLayer(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv=nn.Conv2d(in_c,
                      out_c,
                      3,
                      stride=1,
                      padding=1,
                      dilation=1,
                      groups=1,
                      bias=False,
                      padding_mode='zeros')
        self.bn= nn.BatchNorm2d(out_c)

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        y = torch.relu(y)
        return y


class ResnetLayer(nn.Module):
    def __init__(self, inout_c, mid_c):
        super().__init__()
        self.conv_net = nn.Sequential(
            CNNLayer(inout_c, mid_c),
            CNNLayer(mid_c, inout_c)
        )

    def forward(self, x):
        x = self.conv_net(x) + x
        return x

class Outputhead_v1(nn.Module):

    def __init__(self,out_c,head_mid_c):
        super().__init__()
        self.cnn=CNNLayer(out_c, head_mid_c)
        self.valueHeadLinear = nn.Linear(head_mid_c, 3)
        self.policyHeadLinear = nn.Conv2d(head_mid_c, 1, 1)

    def forward(self, h):
        x=self.cnn(h)

        # value head
        value = x.mean((2, 3))
        value = self.valueHeadLinear(value)

        # policy head
        policy = self.policyHeadLinear(x)
        policy = policy.squeeze(1)

        return value, policy



class Model_ResNet(nn.Module):

    def __init__(self,b,f):
        super().__init__()
        self.model_type = "res"
        self.model_param=(b,f)

        self.inputhead=CNNLayer(input_c, f)
        self.trunk=nn.ModuleList()
        for i in range(b):
            self.trunk.append(ResnetLayer(f,f))
        self.outputhead=Outputhead_v1(f,f)

    def forward(self, x):
        if(x.shape[1]==2):#global feature is none
            x=torch.cat((x,torch.zeros((1,input_c-2,boardH,boardW)).to(x.device)),dim=1)
        h=self.inputhead(x)

        for block in self.trunk:
            h=block(h)

        return self.outputhead(h)

class Model_v2a(nn.Module):

    def __init__(self, midb=5, midc=128, groupc=16, mlpc=32, mapmax=30):  # 1d卷积通道，policy通道，胜负和通道
        super().__init__()
        self.model_type = "v2a"
        self.model_param = (midb, midc, groupc, mlpc, mapmax)
        self.groupc=groupc

        self.mapmax=mapmax

        self.mapping = Mapping1big(midb, midc, 2*groupc)
        self.g1lr=channelwiseLeakyRelu(groupc,bound6=True,bias=False)
        self.h1conv=Conv1dGroupLayerTupleOp(groupc,groupc,L=11,groups=groupc)
        self.h1lr1=channelwiseLeakyRelu(groupc,bound6=True,bias=False)
        self.h1lr2=channelwiseLeakyRelu(groupc,bound6=True,bias=True)

        self.h3plr=channelwiseLeakyRelu(groupc,bound6=True,bias=False)
        self.policy_linear=nn.Conv2d(groupc,1,kernel_size=1,padding=0,bias=True)
        self.policylr=channelwiseLeakyRelu(1,bias=False)

        self.h3vlr=channelwiseLeakyRelu(groupc,bound6=True,bias=True)
        self.valuelr=channelwiseLeakyRelu(groupc,bias=False)
        self.value_linear1=nn.Linear(groupc,mlpc)
        self.value_linear2=nn.Linear(mlpc,mlpc)
        self.value_linear3=nn.Linear(mlpc,mlpc)
        self.value_linearfinal=nn.Linear(mlpc,3)

    def forward(self, x):
        mapf = self.mapping(x)


        if(self.mapmax!=0):
            mapf=self.mapmax*torch.tanh(mapf/self.mapmax) # |map|<30

        g1=mapf[:,:,:self.groupc,:,:]#第一组通道
        g2=mapf[:,:,self.groupc:,:,:]#第二组通道
        h1=self.g1lr(g1.sum(1,keepdims=True)-g1,dim=2)#三线求和再leakyrelu
        h1=(h1[:,0],h1[:,1],h1[:,2],h1[:,3])
        h1=torch.stack(self.h1conv(h1),dim=1)#沿着另一条线卷积
        h2=self.h1lr2(self.h1lr1(h1,dim=2)+g2,dim=2)
        h3=h2.mean(1)#最后把四条线整合起来

        h3p=self.h3plr(h3)
        p=self.policylr(self.policy_linear(h3p))

        h3v=self.h3vlr(h3)
        v=h3v.mean((2,3))
        v=self.valuelr(v)
        v=self.value_linear1(v)
        v=torch.relu(v)
        v=self.value_linear2(v)
        v=torch.relu(v)
        v=self.value_linear3(v)
        v=torch.relu(v)
        v=self.value_linearfinal(v)
        return v, p

class Model_v2b(nn.Module):
    #每3线求和改成4线求和（节省计算量）
    def __init__(self, midb=5, midc=128, groupc=16, mlpc=32, mapmax=30):  # 1d卷积通道，policy通道，胜负和通道
        super().__init__()
        self.model_type = "v2b"
        self.model_param = (midb, midc, groupc, mlpc, mapmax)
        self.groupc=groupc

        self.mapmax=mapmax

        self.mapping = Mapping1big(midb, midc, 2*groupc)
        self.g1lr=channelwiseLeakyRelu(groupc,bound6=True,bias=False)
        self.h1conv=Conv1dGroupLayerTupleOp(groupc,groupc,L=11,groups=groupc)
        self.h1lr1=channelwiseLeakyRelu(groupc,bound6=True,bias=False)
        self.h1lr2=channelwiseLeakyRelu(groupc,bound6=True,bias=True)

        self.h3plr=channelwiseLeakyRelu(groupc,bound6=True,bias=False)
        self.policy_linear=nn.Conv2d(groupc,1,kernel_size=1,padding=0,bias=True)
        self.policylr=channelwiseLeakyRelu(1,bias=False)

        self.h3vlr=channelwiseLeakyRelu(groupc,bound6=True,bias=True)
        self.valuelr=channelwiseLeakyRelu(groupc,bias=False)
        self.value_linear1=nn.Linear(groupc,mlpc)
        self.value_linear2=nn.Linear(mlpc,mlpc)
        self.value_linear3=nn.Linear(mlpc,mlpc)
        self.value_linearfinal=nn.Linear(mlpc,3)

    def forward(self, x):
        mapf = self.mapping(x)


        if(self.mapmax!=0):
            mapf=self.mapmax*torch.tanh(mapf/self.mapmax) # |map|<30

        g1=mapf[:,:,:self.groupc,:,:]#第一组通道
        g2=mapf[:,:,self.groupc:,:,:]#第二组通道
        h1=self.g1lr(g1.sum(1))#四线求和再leakyrelu
        h1=(h1,h1,h1,h1)
        h1=torch.stack(self.h1conv(h1),dim=1)#沿着另一条线卷积
        h2=self.h1lr2(self.h1lr1(h1,dim=2)+g2,dim=2)
        h3=h2.mean(1)#最后把四条线整合起来

        h3p=self.h3plr(h3)
        p=self.policylr(self.policy_linear(h3p))

        h3v=self.h3vlr(h3)
        v=h3v.mean((2,3))
        v=self.valuelr(v)
        v=self.value_linear1(v)
        v=torch.relu(v)
        v=self.value_linear2(v)
        v=torch.relu(v)
        v=self.value_linear3(v)
        v=torch.relu(v)
        v=self.value_linearfinal(v)
        return v, p

class Model_v2c(nn.Module):
    #加了另一组映射表，用于原地
    def __init__(self, midb=5, midc=128, groupc=16, mlpc=32, mapmax=30):  # 1d卷积通道，policy通道，胜负和通道
        super().__init__()
        self.model_type = "v2c"
        self.model_param = (midb, midc, groupc, mlpc, mapmax)
        self.groupc=groupc

        self.mapmax=mapmax

        self.mapping = Mapping1big(midb, midc, 3*groupc)
        self.g1lr=channelwiseLeakyRelu(groupc,bound6=True,bias=False)
        self.g3lr=channelwiseLeakyRelu(groupc,bound6=True,bias=False)
        self.h1conv=Conv1dGroupLayerTupleOp(groupc,groupc,L=11,groups=groupc)
        self.h1lr1=channelwiseLeakyRelu(groupc,bound6=True,bias=False)
        self.h1lr2=channelwiseLeakyRelu(groupc,bound6=True,bias=True)

        self.h3plr=channelwiseLeakyRelu(groupc,bound6=True,bias=False)
        self.policy_linear=nn.Conv2d(groupc,1,kernel_size=1,padding=0,bias=True)
        self.policylr=channelwiseLeakyRelu(1,bias=False)

        self.h3vlr=channelwiseLeakyRelu(groupc,bound6=True,bias=True)
        self.valuelr=channelwiseLeakyRelu(groupc,bias=False)
        self.value_linear1=nn.Linear(groupc,mlpc)
        self.value_linear2=nn.Linear(mlpc,mlpc)
        self.value_linear3=nn.Linear(mlpc,mlpc)
        self.value_linearfinal=nn.Linear(mlpc,3)

    def forward(self, x):
        mapf = self.mapping(x)


        if(self.mapmax!=0):
            mapf=self.mapmax*torch.tanh(mapf/self.mapmax) # |map|<30

        g1=mapf[:,:,:self.groupc,:,:]#第一组通道
        g2=mapf[:,:,self.groupc:2*self.groupc,:,:]#第二组通道
        g3=mapf[:,:,2*self.groupc:,:,:]#第三组通道
        h1=self.g1lr(g1.sum(1,keepdims=True)-g1,dim=2)#三线求和再leakyrelu
        hg3=self.g3lr(g3.sum(1,keepdims=True)-g3,dim=2)#三线求和再leakyrelu
        h1=(h1[:,0],h1[:,1],h1[:,2],h1[:,3])
        h1=torch.stack(self.h1conv(h1),dim=1)#沿着另一条线卷积
        h2=self.h1lr2(self.h1lr1(h1,dim=2)+g2+hg3,dim=2)
        h3=h2.mean(1)#最后把四条线整合起来

        h3p=self.h3plr(h3)
        p=self.policylr(self.policy_linear(h3p))

        h3v=self.h3vlr(h3)
        v=h3v.mean((2,3))
        v=self.valuelr(v)
        v=self.value_linear1(v)
        v=torch.relu(v)
        v=self.value_linear2(v)
        v=torch.relu(v)
        v=self.value_linear3(v)
        v=torch.relu(v)
        v=self.value_linearfinal(v)
        return v, p


class Model_v2d(nn.Module):
    #policy加了个1x1卷积
    def __init__(self, midb=5, midc=128, groupc=16, mlpc=32, mapmax=30):  # 1d卷积通道，policy通道，胜负和通道
        super().__init__()
        self.model_type = "v2d"
        self.model_param = (midb, midc, groupc, mlpc, mapmax)
        self.groupc=groupc

        self.mapmax=mapmax

        self.mapping = Mapping1big(midb, midc, 2*groupc)
        self.g1lr=channelwiseLeakyRelu(groupc,bound6=True,bias=False)
        self.h1conv=Conv1dGroupLayerTupleOp(groupc,groupc,L=11,groups=groupc)
        self.h1lr1=channelwiseLeakyRelu(groupc,bound6=True,bias=False)
        self.h1lr2=channelwiseLeakyRelu(groupc,bound6=True,bias=True)

        self.h3plr=channelwiseLeakyRelu(groupc,bound6=True,bias=False)
        self.policyconv=nn.Conv2d(groupc,groupc,kernel_size=1,padding=0,bias=True)
        self.policy_linear=nn.Conv2d(groupc,1,kernel_size=1,padding=0,bias=True)
        self.policylr=channelwiseLeakyRelu(1,bias=False)

        self.h3vlr=channelwiseLeakyRelu(groupc,bound6=True,bias=True)
        self.valuelr=channelwiseLeakyRelu(groupc,bias=False)
        self.value_linear1=nn.Linear(groupc,mlpc)
        self.value_linear2=nn.Linear(mlpc,mlpc)
        self.value_linear3=nn.Linear(mlpc,mlpc)
        self.value_linearfinal=nn.Linear(mlpc,3)

    def forward(self, x):
        mapf = self.mapping(x)


        if(self.mapmax!=0):
            mapf=self.mapmax*torch.tanh(mapf/self.mapmax) # |map|<30

        g1=mapf[:,:,:self.groupc,:,:]#第一组通道
        g2=mapf[:,:,self.groupc:,:,:]#第二组通道
        h1=self.g1lr(g1.sum(1,keepdims=True)-g1,dim=2)#三线求和再leakyrelu
        h1=(h1[:,0],h1[:,1],h1[:,2],h1[:,3])
        h1=torch.stack(self.h1conv(h1),dim=1)#沿着另一条线卷积
        h2=self.h1lr2(self.h1lr1(h1,dim=2)+g2,dim=2)
        h3=h2.mean(1)#最后把四条线整合起来

        h3p=self.h3plr(h3)
        p=torch.relu(self.policyconv(h3p))
        p=self.policylr(self.policy_linear(p))

        h3v=self.h3vlr(h3)
        v=h3v.mean((2,3))
        v=self.valuelr(v)
        v=self.value_linear1(v)
        v=torch.relu(v)
        v=self.value_linear2(v)
        v=torch.relu(v)
        v=self.value_linear3(v)
        v=torch.relu(v)
        v=self.value_linearfinal(v)
        return v, p

class Model_v2ba(nn.Module):
    #v2b的基础上，直接去掉group2
    def __init__(self, midb=5, midc=128, groupc=16, mlpc=32, mapmax=30):  # 1d卷积通道，policy通道，胜负和通道
        super().__init__()
        self.model_type = "v2ba"
        self.model_param = (midb, midc, groupc, mlpc, mapmax)
        self.groupc=groupc

        self.mapmax=mapmax

        self.mapping = Mapping1big(midb, midc, groupc)
        self.g1lr=channelwiseLeakyRelu(groupc,bound6=True,bias=False)
        self.h1conv=Conv1dGroupLayerTupleOp(groupc,groupc,L=11,groups=groupc)
        self.h1lr=channelwiseLeakyRelu(groupc,bound6=True,bias=True)

        self.h3plr=channelwiseLeakyRelu(groupc,bound6=True,bias=False)
        self.policy_linear=nn.Conv2d(groupc,1,kernel_size=1,padding=0,bias=True)
        self.policylr=channelwiseLeakyRelu(1,bias=False)

        self.h3vlr=channelwiseLeakyRelu(groupc,bound6=True,bias=True)
        self.valuelr=channelwiseLeakyRelu(groupc,bias=False)
        self.value_linear1=nn.Linear(groupc,mlpc)
        self.value_linear2=nn.Linear(mlpc,mlpc)
        self.value_linear3=nn.Linear(mlpc,mlpc)
        self.value_linearfinal=nn.Linear(mlpc,3)

    def forward(self, x):
        mapf = self.mapping(x)


        if(self.mapmax!=0):
            mapf=self.mapmax*torch.tanh(mapf/self.mapmax) # |map|<30

        #g1=mapf[:,:,:self.groupc,:,:]#第一组通道
        #g2=mapf[:,:,self.groupc:,:,:]#第二组通道
        g1=mapf
        h1=self.g1lr(g1.sum(1))#四线求和再leakyrelu
        h1=(h1,h1,h1,h1)
        h1=torch.stack(self.h1conv(h1),dim=1)#沿着另一条线卷积
        h2=self.h1lr(h1,dim=2)
        h3=h2.mean(1)#最后把四条线整合起来

        h3p=self.h3plr(h3)
        p=self.policylr(self.policy_linear(h3p))

        h3v=self.h3vlr(h3)
        v=h3v.mean((2,3))
        v=self.valuelr(v)
        v=self.value_linear1(v)
        v=torch.relu(v)
        v=self.value_linear2(v)
        v=torch.relu(v)
        v=self.value_linear3(v)
        v=torch.relu(v)
        v=self.value_linearfinal(v)
        return v, p

class Model_v2bsym(nn.Module):
    #v2b+把h1卷积改成对称的
    def __init__(self, midb=5, midc=128, groupc=16, mlpc=32, mapmax=30):  # 1d卷积通道，policy通道，胜负和通道
        super().__init__()
        self.model_type = "v2bsym"
        self.model_param = (midb, midc, groupc, mlpc, mapmax)
        self.groupc=groupc

        self.mapmax=mapmax

        self.mapping = Mapping1big(midb, midc, 2*groupc)
        self.g1lr=channelwiseLeakyRelu(groupc,bound6=True,bias=False)
        self.h1conv=Conv1dGroupLayerTupleOp(groupc,groupc,L=11,groups=groupc)
        self.h1lr1=channelwiseLeakyRelu(groupc,bound6=True,bias=False)
        self.h1lr2=channelwiseLeakyRelu(groupc,bound6=True,bias=True)

        self.h3plr=channelwiseLeakyRelu(groupc,bound6=True,bias=False)
        self.policy_linear=nn.Conv2d(groupc,1,kernel_size=1,padding=0,bias=True)
        self.policylr=channelwiseLeakyRelu(1,bias=False)

        self.h3vlr=channelwiseLeakyRelu(groupc,bound6=True,bias=True)
        self.valuelr=channelwiseLeakyRelu(groupc,bias=False)
        self.value_linear1=nn.Linear(groupc,mlpc)
        self.value_linear2=nn.Linear(mlpc,mlpc)
        self.value_linear3=nn.Linear(mlpc,mlpc)
        self.value_linearfinal=nn.Linear(mlpc,3)

    def forward(self, x):
        mapf = self.mapping(x)


        if(self.mapmax!=0):
            mapf=self.mapmax*torch.tanh(mapf/self.mapmax) # |map|<30

        g1=mapf[:,:,:self.groupc,:,:]#第一组通道
        g2=mapf[:,:,self.groupc:,:,:]#第二组通道
        h1=self.g1lr(g1.mean(1))#四线求和再leakyrelu
        h1sym=torch.flip(h1,[2,3])
        h1=(h1,h1,h1,h1)
        h1sym=(h1sym,h1sym,h1sym,h1sym)
        h1=torch.stack(self.h1conv(h1),dim=1)#沿着另一条线卷积
        h1sym=torch.stack(self.h1conv(h1sym),dim=1)#沿着另一条线卷积
        h1sym=torch.flip(h1sym,[3,4])
        h1=(h1+h1sym)/2
        h2=self.h1lr2(self.h1lr1(h1,dim=2)+g2,dim=2)
        h3=h2.mean(1)#最后把四条线整合起来

        h3p=self.h3plr(h3)
        p=self.policylr(self.policy_linear(h3p))

        h3v=self.h3vlr(h3)
        v=h3v.mean((2,3))
        v=self.valuelr(v)
        v=self.value_linear1(v)
        v=torch.relu(v)
        v=self.value_linear2(v)
        v=torch.relu(v)
        v=self.value_linear3(v)
        v=torch.relu(v)
        v=self.value_linearfinal(v)
        return v, p

class Model_v2bb(nn.Module):
    #v2bsym+去掉h1lr1
    def __init__(self, midb=5, midc=128, groupc=16, mlpc=32, mapmax=30):  # 1d卷积通道，policy通道，胜负和通道
        super().__init__()
        self.model_type = "v2bb"
        self.model_param = (midb, midc, groupc, mlpc, mapmax)
        self.groupc=groupc

        self.mapmax=mapmax

        self.mapping = Mapping1big(midb, midc, 2*groupc)
        self.g1lr=channelwiseLeakyRelu(groupc,bound6=True,bias=False)
        self.h1conv=Conv1dGroupLayerTupleOp(groupc,groupc,L=11,groups=groupc)
        self.h1lr2=channelwiseLeakyRelu(groupc,bound6=True,bias=True)

        self.h3plr=channelwiseLeakyRelu(groupc,bound6=True,bias=False)
        self.policy_linear=nn.Conv2d(groupc,1,kernel_size=1,padding=0,bias=True)
        self.policylr=channelwiseLeakyRelu(1,bias=False)

        self.h3vlr=channelwiseLeakyRelu(groupc,bound6=True,bias=True)
        self.valuelr=channelwiseLeakyRelu(groupc,bias=False)
        self.value_linear1=nn.Linear(groupc,mlpc)
        self.value_linear2=nn.Linear(mlpc,mlpc)
        self.value_linear3=nn.Linear(mlpc,mlpc)
        self.value_linearfinal=nn.Linear(mlpc,3)

    def forward(self, x):
        mapf = self.mapping(x)


        if(self.mapmax!=0):
            mapf=self.mapmax*torch.tanh(mapf/self.mapmax) # |map|<30

        g1=mapf[:,:,:self.groupc,:,:]#第一组通道
        g2=mapf[:,:,self.groupc:,:,:]#第二组通道
        h1=self.g1lr(g1.mean(1))#四线求和再leakyrelu

        #以下几行是对h1进行对称的卷积。把h1卷积一次，再把h1翻转一下再卷积一次，两次的取平均，相当于对称的卷积核
        h1sym=torch.flip(h1,[2,3])
        h1=(h1,h1,h1,h1)
        h1sym=(h1sym,h1sym,h1sym,h1sym)
        h1=torch.stack(self.h1conv(h1),dim=1)#沿着另一条线卷积
        h1sym=torch.stack(self.h1conv(h1sym),dim=1)#等价于卷积核反向
        h1sym=torch.flip(h1sym,[3,4])
        h1=(h1+h1sym)/2#正向和反向取平均，相当于对称卷积核

        h2=self.h1lr2(h1+g2,dim=2)
        h3=h2.mean(1)#最后把四条线整合起来

        h3p=self.h3plr(h3)
        p=self.policylr(self.policy_linear(h3p))

        h3v=self.h3vlr(h3)
        v=h3v.mean((2,3))
        v=self.valuelr(v)
        v=self.value_linear1(v)
        v=torch.relu(v)
        v=self.value_linear2(v)
        v=torch.relu(v)
        v=self.value_linear3(v)
        v=torch.relu(v)
        v=self.value_linearfinal(v)
        return v, p

class Model_v2bc(nn.Module):
    #v2bsym+最后给policy加几个卷积
    def __init__(self, midb=5, midc=128, groupc=16, mlpc=32, mapmax=30):  # 1d卷积通道，policy通道，胜负和通道
        super().__init__()
        self.model_type = "v2bc"
        self.model_param = (midb, midc, groupc, mlpc, mapmax)
        self.groupc=groupc

        self.mapmax=mapmax

        self.mapping = Mapping1big(midb, midc, 2*groupc)
        self.g1lr=channelwiseLeakyRelu(groupc,bound6=True,bias=False)
        self.h1conv=Conv1dGroupLayerTupleOp(groupc,groupc,L=11,groups=groupc)
        self.h1lr1=channelwiseLeakyRelu(groupc,bound6=True,bias=False)
        self.h1lr2=channelwiseLeakyRelu(groupc,bound6=True,bias=True)

        self.h3lr=channelwiseLeakyRelu(groupc,bound6=True,bias=False)
        self.trunkconv1=nn.Conv2d(groupc,groupc,groups=groupc//4,kernel_size=1,padding=0,bias=True)
        self.trunklr1=channelwiseLeakyRelu(groupc,bias=False)
        self.trunkconv2=nn.Conv2d(groupc,groupc,groups=groupc,kernel_size=3,padding=1,bias=True)
        self.trunklr2p=channelwiseLeakyRelu(groupc,bias=False)

        self.policy_linear=nn.Conv2d(groupc,1,kernel_size=1,padding=0,bias=True)
        self.policylr=channelwiseLeakyRelu(1,bias=False)

        self.trunklr2v=channelwiseLeakyRelu(groupc,bias=True)
        self.valuelr=channelwiseLeakyRelu(groupc,bias=False)
        self.value_linear1=nn.Linear(groupc,mlpc)
        self.value_linear2=nn.Linear(mlpc,mlpc)
        self.value_linear3=nn.Linear(mlpc,mlpc)
        self.value_linearfinal=nn.Linear(mlpc,3)

    def forward(self, x):
        mapf = self.mapping(x)


        if(self.mapmax!=0):
            mapf=self.mapmax*torch.tanh(mapf/self.mapmax) # |map|<30

        g1=mapf[:,:,:self.groupc,:,:]#第一组通道
        g2=mapf[:,:,self.groupc:,:,:]#第二组通道
        h1=self.g1lr(g1.mean(1))#四线求和再leakyrelu

        #以下几行是对h1进行对称的卷积。把h1卷积一次，再把h1翻转一下再卷积一次，两次的取平均，相当于对称的卷积核
        h1sym=torch.flip(h1,[2,3])
        h1=(h1,h1,h1,h1)
        h1sym=(h1sym,h1sym,h1sym,h1sym)
        h1=torch.stack(self.h1conv(h1),dim=1)#沿着另一条线卷积
        h1sym=torch.stack(self.h1conv(h1sym),dim=1)#等价于卷积核反向
        h1sym=torch.flip(h1sym,[3,4])
        h1=(h1+h1sym)/2#正向和反向取平均，相当于对称卷积核

        h2=self.h1lr2(self.h1lr1(h1,dim=2)+g2,dim=2)
        h3=h2.mean(1)#最后把四条线整合起来

        trunk=self.h3lr(h3)
        trunk=self.trunkconv1(trunk)
        trunk=self.trunklr1(trunk)
        trunk=self.trunkconv2(trunk)

        p=self.trunklr2p(trunk)
        v=self.trunklr2v(trunk)


        p=self.policylr(self.policy_linear(p))

        v=v.mean((2,3))
        v=self.valuelr(v)
        v=self.value_linear1(v)
        v=torch.relu(v)
        v=self.value_linear2(v)
        v=torch.relu(v)
        v=self.value_linear3(v)
        v=torch.relu(v)
        v=self.value_linearfinal(v)
        return v, p

class Model_v2bd(nn.Module):
    #v2bsym+h1conv长度改成7
    def __init__(self, midb=5, midc=128, groupc=16, mlpc=32, mapmax=30):  # 1d卷积通道，policy通道，胜负和通道
        super().__init__()
        self.model_type = "v2bd"
        self.model_param = (midb, midc, groupc, mlpc, mapmax)
        self.groupc=groupc

        self.mapmax=mapmax

        self.mapping = Mapping1big(midb, midc, 2*groupc)
        self.g1lr=channelwiseLeakyRelu(groupc,bound6=True,bias=False)
        self.h1conv=Conv1dGroupLayerTupleOp(groupc,groupc,L=7,groups=groupc)
        self.h1lr1=channelwiseLeakyRelu(groupc,bound6=True,bias=False)
        self.h1lr2=channelwiseLeakyRelu(groupc,bound6=True,bias=True)

        self.h3plr=channelwiseLeakyRelu(groupc,bound6=True,bias=False)
        self.policy_linear=nn.Conv2d(groupc,1,kernel_size=1,padding=0,bias=True)
        self.policylr=channelwiseLeakyRelu(1,bias=False)

        self.h3vlr=channelwiseLeakyRelu(groupc,bound6=True,bias=True)
        self.valuelr=channelwiseLeakyRelu(groupc,bias=False)
        self.value_linear1=nn.Linear(groupc,mlpc)
        self.value_linear2=nn.Linear(mlpc,mlpc)
        self.value_linear3=nn.Linear(mlpc,mlpc)
        self.value_linearfinal=nn.Linear(mlpc,3)

    def forward(self, x):
        mapf = self.mapping(x)


        if(self.mapmax!=0):
            mapf=self.mapmax*torch.tanh(mapf/self.mapmax) # |map|<30

        g1=mapf[:,:,:self.groupc,:,:]#第一组通道
        g2=mapf[:,:,self.groupc:,:,:]#第二组通道
        h1=self.g1lr(g1.mean(1))#四线求和再leakyrelu

        #以下几行是对h1进行对称的卷积。把h1卷积一次，再把h1翻转一下再卷积一次，两次的取平均，相当于对称的卷积核
        h1sym=torch.flip(h1,[2,3])
        h1=(h1,h1,h1,h1)
        h1sym=(h1sym,h1sym,h1sym,h1sym)
        h1=torch.stack(self.h1conv(h1),dim=1)#沿着另一条线卷积
        h1sym=torch.stack(self.h1conv(h1sym),dim=1)#等价于卷积核反向
        h1sym=torch.flip(h1sym,[3,4])
        h1=(h1+h1sym)/2#正向和反向取平均，相当于对称卷积核

        h2=self.h1lr2(self.h1lr1(h1,dim=2)+g2,dim=2)
        h3=h2.mean(1)#最后把四条线整合起来

        h3p=self.h3plr(h3)
        p=self.policylr(self.policy_linear(h3p))

        h3v=self.h3vlr(h3)
        v=h3v.mean((2,3))
        v=self.valuelr(v)
        v=self.value_linear1(v)
        v=torch.relu(v)
        v=self.value_linear2(v)
        v=torch.relu(v)
        v=self.value_linear3(v)
        v=torch.relu(v)
        v=self.value_linearfinal(v)
        return v, p

class Model_v2bca(nn.Module):
    #v2bc+leakyrelu换成prelu3a
    def __init__(self, midb=5, midc=128, groupc=16, mlpc=32, mapmax=30):  # 1d卷积通道，policy通道，胜负和通道
        super().__init__()
        self.model_type = "v2bca"
        self.model_param = (midb, midc, groupc, mlpc, mapmax)
        self.groupc=groupc

        self.mapmax=mapmax

        self.mapping = Mapping1big(midb, midc, 2*groupc)
        self.g1lr=PRelu3a(groupc)
        self.h1conv=Conv1dGroupLayerTupleOp(groupc,groupc,L=11,groups=groupc)
        self.h1lr1=PRelu3a(groupc)
        self.h1lr2=PRelu3a(groupc)

        self.h3lr=PRelu3a(groupc)
        self.trunkconv1=nn.Conv2d(groupc,groupc,groups=groupc//4,kernel_size=1,padding=0,bias=True)
        self.trunklr1=PRelu3a(groupc)
        self.trunkconv2=nn.Conv2d(groupc,groupc,groups=groupc,kernel_size=3,padding=1,bias=True)
        self.trunklr2p=PRelu3a(groupc)

        self.policy_linear=nn.Conv2d(groupc,1,kernel_size=1,padding=0,bias=True)
        self.policylr=PRelu3a(1)

        self.trunklr2v=PRelu3a(groupc)
        self.valuelr=PRelu3a(groupc)
        self.value_linear1=nn.Linear(groupc,mlpc)
        self.value_linear2=nn.Linear(mlpc,mlpc)
        self.value_linear3=nn.Linear(mlpc,mlpc)
        self.value_linearfinal=nn.Linear(mlpc,3)

    def forward(self, x):
        mapf = self.mapping(x)


        if(self.mapmax!=0):
            mapf=self.mapmax*torch.tanh(mapf/self.mapmax) # |map|<30

        g1=mapf[:,:,:self.groupc,:,:]#第一组通道
        g2=mapf[:,:,self.groupc:,:,:]#第二组通道
        h1=self.g1lr(g1.mean(1))#四线求和再leakyrelu

        #以下几行是对h1进行对称的卷积。把h1卷积一次，再把h1翻转一下再卷积一次，两次的取平均，相当于对称的卷积核
        h1sym=torch.flip(h1,[2,3])
        h1=(h1,h1,h1,h1)
        h1sym=(h1sym,h1sym,h1sym,h1sym)
        h1=torch.stack(self.h1conv(h1),dim=1)#沿着另一条线卷积
        h1sym=torch.stack(self.h1conv(h1sym),dim=1)#等价于卷积核反向
        h1sym=torch.flip(h1sym,[3,4])
        h1=(h1+h1sym)/2#正向和反向取平均，相当于对称卷积核

        h2=self.h1lr2(self.h1lr1(h1,dim=2)+g2,dim=2)
        h3=h2.mean(1)#最后把四条线整合起来

        trunk=self.h3lr(h3)
        trunk=self.trunkconv1(trunk)
        trunk=self.trunklr1(trunk)
        trunk=self.trunkconv2(trunk)

        p=self.trunklr2p(trunk)
        v=self.trunklr2v(trunk)


        p=self.policylr(self.policy_linear(p))

        v=v.mean((2,3))
        v=self.valuelr(v)
        v=self.value_linear1(v)
        v=torch.relu(v)
        v=self.value_linear2(v)
        v=torch.relu(v)
        v=self.value_linear3(v)
        v=torch.relu(v)
        v=self.value_linearfinal(v)
        return v, p

class Model_v2bcb(nn.Module):
    #v2bc+leakyrelu换成bound=1的prelu1
    def __init__(self, midb=5, midc=128, groupc=16, mlpc=32, mapmax=30):  # 1d卷积通道，policy通道，胜负和通道
        super().__init__()
        self.model_type = "v2bcb"
        self.model_param = (midb, midc, groupc, mlpc, mapmax)
        self.groupc=groupc

        self.mapmax=mapmax

        self.mapping = Mapping1big(midb, midc, 2*groupc)
        self.g1lr=PRelu1(groupc,bound=1)
        self.h1conv=Conv1dGroupLayerTupleOp(groupc,groupc,L=11,groups=groupc)
        self.h1lr1=PRelu1(groupc,bound=1)
        self.h1lr2=PRelu1(groupc,bound=1)

        self.h3lr=PRelu1(groupc,bound=1)
        self.trunkconv1=nn.Conv2d(groupc,groupc,groups=groupc//4,kernel_size=1,padding=0,bias=True)
        self.trunklr1=PRelu1(groupc,bound=1)
        self.trunkconv2=nn.Conv2d(groupc,groupc,groups=groupc,kernel_size=3,padding=1,bias=True)
        self.trunklr2p=PRelu1(groupc,bound=1)

        self.policy_linear=nn.Conv2d(groupc,1,kernel_size=1,padding=0,bias=True)
        self.policylr=PRelu1(1)

        self.trunklr2v=PRelu1(groupc,bound=1)
        self.valuelr=PRelu1(groupc)
        self.value_linear1=nn.Linear(groupc,mlpc)
        self.value_linear2=nn.Linear(mlpc,mlpc)
        self.value_linear3=nn.Linear(mlpc,mlpc)
        self.value_linearfinal=nn.Linear(mlpc,3)

    def forward(self, x):
        mapf = self.mapping(x)


        if(self.mapmax!=0):
            mapf=self.mapmax*torch.tanh(mapf/self.mapmax) # |map|<30

        g1=mapf[:,:,:self.groupc,:,:]#第一组通道
        g2=mapf[:,:,self.groupc:,:,:]#第二组通道
        h1=self.g1lr(g1.mean(1))#四线求和再leakyrelu

        #以下几行是对h1进行对称的卷积。把h1卷积一次，再把h1翻转一下再卷积一次，两次的取平均，相当于对称的卷积核
        h1sym=torch.flip(h1,[2,3])
        h1=(h1,h1,h1,h1)
        h1sym=(h1sym,h1sym,h1sym,h1sym)
        h1=torch.stack(self.h1conv(h1),dim=1)#沿着另一条线卷积
        h1sym=torch.stack(self.h1conv(h1sym),dim=1)#等价于卷积核反向
        h1sym=torch.flip(h1sym,[3,4])
        h1=(h1+h1sym)/2#正向和反向取平均，相当于对称卷积核

        h2=self.h1lr2(self.h1lr1(h1,dim=2)+g2,dim=2)
        h3=h2.mean(1)#最后把四条线整合起来

        trunk=self.h3lr(h3)
        trunk=self.trunkconv1(trunk)
        trunk=self.trunklr1(trunk)
        trunk=self.trunkconv2(trunk)

        p=self.trunklr2p(trunk)
        v=self.trunklr2v(trunk)


        p=self.policylr(self.policy_linear(p))

        v=v.mean((2,3))
        v=self.valuelr(v)
        v=self.value_linear1(v)
        v=torch.relu(v)
        v=self.value_linear2(v)
        v=torch.relu(v)
        v=self.value_linear3(v)
        v=torch.relu(v)
        v=self.value_linearfinal(v)
        return v, p

class Model_v2bcc(nn.Module):
    #v2bc+leakyrelu换成prelu4a和4b交替
    def __init__(self, midb=5, midc=128, groupc=16, mlpc=32, mapmax=30):  # 1d卷积通道，policy通道，胜负和通道
        super().__init__()
        self.model_type = "v2bcc"
        self.model_param = (midb, midc, groupc, mlpc, mapmax)
        self.groupc=groupc

        self.mapmax=mapmax

        self.mapping = Mapping1big(midb, midc, 2*groupc)
        self.g1lr=PRelu4a(groupc)
        self.h1conv=Conv1dGroupLayerTupleOp(groupc,groupc,L=11,groups=groupc)
        self.h1lr1=PRelu4b(groupc)
        self.h1lr2=PRelu4a(groupc)

        self.h3lr=PRelu4b(groupc)
        self.trunkconv1=nn.Conv2d(groupc,groupc,groups=groupc//4,kernel_size=1,padding=0,bias=True)
        self.trunklr1=PRelu4a(groupc)
        self.trunkconv2=nn.Conv2d(groupc,groupc,groups=groupc,kernel_size=3,padding=1,bias=True)
        self.trunklr2p=PRelu4b(groupc)

        self.policy_linear=nn.Conv2d(groupc,1,kernel_size=1,padding=0,bias=True)
        self.policylr=PRelu4a(1)

        self.trunklr2v=PRelu4b(groupc)
        self.valuelr=PRelu4a(groupc)
        self.value_linear1=nn.Linear(groupc,mlpc)
        self.value_linear2=nn.Linear(mlpc,mlpc)
        self.value_linear3=nn.Linear(mlpc,mlpc)
        self.value_linearfinal=nn.Linear(mlpc,3)

    def forward(self, x):
        mapf = self.mapping(x)


        if(self.mapmax!=0):
            mapf=self.mapmax*torch.tanh(mapf/self.mapmax) # |map|<30

        g1=mapf[:,:,:self.groupc,:,:]#第一组通道
        g2=mapf[:,:,self.groupc:,:,:]#第二组通道
        h1=self.g1lr(g1.mean(1))#四线求和再leakyrelu

        #以下几行是对h1进行对称的卷积。把h1卷积一次，再把h1翻转一下再卷积一次，两次的取平均，相当于对称的卷积核
        h1sym=torch.flip(h1,[2,3])
        h1=(h1,h1,h1,h1)
        h1sym=(h1sym,h1sym,h1sym,h1sym)
        h1=torch.stack(self.h1conv(h1),dim=1)#沿着另一条线卷积
        h1sym=torch.stack(self.h1conv(h1sym),dim=1)#等价于卷积核反向
        h1sym=torch.flip(h1sym,[3,4])
        h1=(h1+h1sym)/2#正向和反向取平均，相当于对称卷积核

        h2=self.h1lr2(self.h1lr1(h1,dim=2)+g2,dim=2)
        h3=h2.mean(1)#最后把四条线整合起来

        trunk=self.h3lr(h3)
        trunk=self.trunkconv1(trunk)
        trunk=self.trunklr1(trunk)
        trunk=self.trunkconv2(trunk)

        p=self.trunklr2p(trunk)
        v=self.trunklr2v(trunk)


        p=self.policylr(self.policy_linear(p))

        v=v.mean((2,3))
        v=self.valuelr(v)
        v=self.value_linear1(v)
        v=torch.relu(v)
        v=self.value_linear2(v)
        v=torch.relu(v)
        v=self.value_linear3(v)
        v=torch.relu(v)
        v=self.value_linearfinal(v)
        return v, p

class Model_v2bcd(nn.Module):
    #v2bc+leakyrelu换成bound=1.1的prelu1与prelu3
    def __init__(self, midb=5, midc=128, groupc=16, mlpc=32, mapmax=30):  # 1d卷积通道，policy通道，胜负和通道
        super().__init__()
        self.model_type = "v2bcd"
        self.model_param = (midb, midc, groupc, mlpc, mapmax)
        self.groupc=groupc

        self.mapmax=mapmax

        self.mapping = Mapping1big(midb, midc, 2*groupc)
        self.g1lr=PRelu3a(groupc,bound=1.1)
        self.h1conv=Conv1dGroupLayerTupleOp(groupc,groupc,L=11,groups=groupc)
        self.h1lr1=PRelu1(groupc,bound=1.1)
        self.h1lr2=PRelu1(groupc,bound=1.1)

        self.h3lr=PRelu3b(groupc,bound=1.1)
        self.trunkconv1=nn.Conv2d(groupc,groupc,groups=groupc//4,kernel_size=1,padding=0,bias=True)
        self.trunklr1=PRelu3a(groupc,bound=1.1)
        self.trunkconv2=nn.Conv2d(groupc,groupc,groups=groupc,kernel_size=3,padding=1,bias=True)
        self.trunklr2p=PRelu3a(groupc,bound=1.1)

        self.policy_linear=nn.Conv2d(groupc,1,kernel_size=1,padding=0,bias=True)
        self.policylr=PRelu1(1)

        self.trunklr2v=PRelu3a(groupc,bound=1.1)
        self.valuelr=PRelu1(groupc)
        self.value_linear1=nn.Linear(groupc,mlpc)
        self.value_linear2=nn.Linear(mlpc,mlpc)
        self.value_linear3=nn.Linear(mlpc,mlpc)
        self.value_linearfinal=nn.Linear(mlpc,3)

    def forward(self, x):
        mapf = self.mapping(x)


        if(self.mapmax!=0):
            mapf=self.mapmax*torch.tanh(mapf/self.mapmax) # |map|<30

        g1=mapf[:,:,:self.groupc,:,:]#第一组通道
        g2=mapf[:,:,self.groupc:,:,:]#第二组通道
        h1=self.g1lr(g1.mean(1))#四线求和再leakyrelu

        #以下几行是对h1进行对称的卷积。把h1卷积一次，再把h1翻转一下再卷积一次，两次的取平均，相当于对称的卷积核
        h1sym=torch.flip(h1,[2,3])
        h1=(h1,h1,h1,h1)
        h1sym=(h1sym,h1sym,h1sym,h1sym)
        h1=torch.stack(self.h1conv(h1),dim=1)#沿着另一条线卷积
        h1sym=torch.stack(self.h1conv(h1sym),dim=1)#等价于卷积核反向
        h1sym=torch.flip(h1sym,[3,4])
        h1=(h1+h1sym)/2#正向和反向取平均，相当于对称卷积核

        h2=self.h1lr2(self.h1lr1(h1,dim=2)+g2,dim=2)
        h3=h2.mean(1)#最后把四条线整合起来

        trunk=self.h3lr(h3)
        trunk=self.trunkconv1(trunk)
        trunk=self.trunklr1(trunk)
        trunk=self.trunkconv2(trunk)

        p=self.trunklr2p(trunk)
        v=self.trunklr2v(trunk)


        p=self.policylr(self.policy_linear(p))

        v=v.mean((2,3))
        v=self.valuelr(v)
        v=self.value_linear1(v)
        v=torch.relu(v)
        v=self.value_linear2(v)
        v=torch.relu(v)
        v=self.value_linear3(v)
        v=torch.relu(v)
        v=self.value_linearfinal(v)
        return v, p

class Model_v2bce(nn.Module):
    #v2bc+leakyrelu换成bound=1.1的prelu1与prelu3c(全凸)
    def __init__(self, midb=5, midc=128, groupc=16, mlpc=32, mapmax=30):  # 1d卷积通道，policy通道，胜负和通道
        super().__init__()
        self.model_type = "v2bce"
        self.model_param = (midb, midc, groupc, mlpc, mapmax)
        self.groupc=groupc

        self.mapmax=mapmax

        self.mapping = Mapping1big(midb, midc, 2*groupc)
        self.g1lr=PRelu3c(groupc,bound=1.1)
        self.h1conv=Conv1dGroupLayerTupleOp(groupc,groupc,L=11,groups=groupc)
        self.h1lr1=PRelu1(groupc,bound=1.1)
        self.h1lr2=PRelu1(groupc,bound=1.1)

        self.h3lr=PRelu3c(groupc,bound=1.1)
        self.trunkconv1=nn.Conv2d(groupc,groupc,groups=groupc//4,kernel_size=1,padding=0,bias=True)
        self.trunklr1=PRelu3c(groupc,bound=1.1)
        self.trunkconv2=nn.Conv2d(groupc,groupc,groups=groupc,kernel_size=3,padding=1,bias=True)
        self.trunklr2p=PRelu3c(groupc,bound=1.1)

        self.policy_linear=nn.Conv2d(groupc,1,kernel_size=1,padding=0,bias=True)
        self.policylr=PRelu1(1)

        self.trunklr2v=PRelu3c(groupc,bound=1.1)
        self.valuelr=PRelu1(groupc)
        self.value_linear1=nn.Linear(groupc,mlpc)
        self.value_linear2=nn.Linear(mlpc,mlpc)
        self.value_linear3=nn.Linear(mlpc,mlpc)
        self.value_linearfinal=nn.Linear(mlpc,3)

    def forward(self, x):
        mapf = self.mapping(x)


        if(self.mapmax!=0):
            mapf=self.mapmax*torch.tanh(mapf/self.mapmax) # |map|<30

        g1=mapf[:,:,:self.groupc,:,:]#第一组通道
        g2=mapf[:,:,self.groupc:,:,:]#第二组通道
        h1=self.g1lr(g1.mean(1))#四线求和再leakyrelu

        #以下几行是对h1进行对称的卷积。把h1卷积一次，再把h1翻转一下再卷积一次，两次的取平均，相当于对称的卷积核
        h1sym=torch.flip(h1,[2,3])
        h1=(h1,h1,h1,h1)
        h1sym=(h1sym,h1sym,h1sym,h1sym)
        h1=torch.stack(self.h1conv(h1),dim=1)#沿着另一条线卷积
        h1sym=torch.stack(self.h1conv(h1sym),dim=1)#等价于卷积核反向
        h1sym=torch.flip(h1sym,[3,4])
        h1=(h1+h1sym)/2#正向和反向取平均，相当于对称卷积核

        h2=self.h1lr2(self.h1lr1(h1,dim=2)+g2,dim=2)
        h3=h2.mean(1)#最后把四条线整合起来

        trunk=self.h3lr(h3)
        trunk=self.trunkconv1(trunk)
        trunk=self.trunklr1(trunk)
        trunk=self.trunkconv2(trunk)

        p=self.trunklr2p(trunk)
        v=self.trunklr2v(trunk)


        p=self.policylr(self.policy_linear(p))

        v=v.mean((2,3))
        v=self.valuelr(v)
        v=self.value_linear1(v)
        v=torch.relu(v)
        v=self.value_linear2(v)
        v=torch.relu(v)
        v=self.value_linear3(v)
        v=torch.relu(v)
        v=self.value_linearfinal(v)
        return v, p


class Model_v2bcf(nn.Module):
    # v2bc调整了一大堆bias和prelu

    def __init__(self, midb=5, midc=128, groupc=16, mlpc=32, mapmax=30):  # 1d卷积通道，policy通道，胜负和通道
        super().__init__()
        self.model_type = "v2bcf"
        self.model_param = (midb, midc, groupc, mlpc, mapmax)
        self.groupc = groupc

        self.mapmax = mapmax

        self.mapping = Mapping1big(midb, midc, 2 * groupc)
        self.g1lr = PRelu3a(groupc, bound=1.1)
        self.h1conv = Conv1dGroupLayerTupleOp(groupc, groupc, L=11, groups=groupc)
        self.h1lr1 = PRelu1(groupc, bound=1.1,bias=False)
        self.h1lr2 = PRelu1(groupc, bound=1.1,bias=True)

        self.h3lr = PRelu3b(groupc, bound=1.1)
        self.trunkconv1 = nn.Conv2d(groupc, groupc, groups=groupc // 4, kernel_size=1, padding=0, bias=False)
        self.trunklr1 = PRelu3a(groupc, bound=1.1)
        self.trunkconv2 = nn.Conv2d(groupc, groupc, groups=groupc, kernel_size=3, padding=1, bias=False)
        self.trunklr2p = PRelu3a(groupc, bound=1.1)

        self.policy_linear = nn.Conv2d(groupc, 1, kernel_size=1, padding=0, bias=True)
        self.policylr = nn.Identity()  #舍弃policylr

        self.trunklr2v = PRelu3a(groupc, bound=1.1)
        self.valuelr = PRelu3b(groupc, bound=10) #后面全是float了
        self.value_linear1 = nn.Linear(groupc, mlpc)
        self.value_linear2 = nn.Linear(mlpc, mlpc)
        self.value_linear3 = nn.Linear(mlpc, mlpc)
        self.value_linearfinal = nn.Linear(mlpc, 3)

    def forward(self, x):
        mapf = self.mapping(x)

        if (self.mapmax != 0):
            mapf = self.mapmax * torch.tanh(mapf / self.mapmax)  # |map|<30

        g1 = mapf[:, :, :self.groupc, :, :]  # 第一组通道
        g2 = mapf[:, :, self.groupc:, :, :]  # 第二组通道
        h1 = self.g1lr(g1.mean(1))  # 四线求和再leakyrelu

        # 以下几行是对h1进行对称的卷积。把h1卷积一次，再把h1翻转一下再卷积一次，两次的取平均，相当于对称的卷积核
        h1sym = torch.flip(h1, [2, 3])
        h1 = (h1, h1, h1, h1)
        h1sym = (h1sym, h1sym, h1sym, h1sym)
        h1 = torch.stack(self.h1conv(h1), dim=1)  # 沿着另一条线卷积
        h1sym = torch.stack(self.h1conv(h1sym), dim=1)  # 等价于卷积核反向
        h1sym = torch.flip(h1sym, [3, 4])
        h1 = (h1 + h1sym) / 2  # 正向和反向取平均，相当于对称卷积核

        h2 = self.h1lr2(self.h1lr1(h1, dim=2) + g2, dim=2)
        h3 = h2.mean(1)  # 最后把四条线整合起来

        trunk = self.h3lr(h3)
        trunk = self.trunkconv1(trunk)
        trunk = self.trunklr1(trunk)
        trunk = self.trunkconv2(trunk)

        p = self.trunklr2p(trunk)
        v = self.trunklr2v(trunk)

        p = self.policylr(self.policy_linear(p))

        v = v.mean((2, 3))
        v = self.valuelr(v)
        v = self.value_linear1(v)
        v = torch.relu(v)
        v = self.value_linear2(v)
        v = torch.relu(v)
        v = self.value_linear3(v)
        v = torch.relu(v)
        v = self.value_linearfinal(v)
        return v, p

class Model_v2bcg(nn.Module):
    #v2bcb+去掉多余bias
    def __init__(self, midb=5, midc=128, groupc=16, mlpc=32, mapmax=30):  # 1d卷积通道，policy通道，胜负和通道
        super().__init__()
        self.model_type = "v2bcg"
        self.model_param = (midb, midc, groupc, mlpc, mapmax)
        self.groupc=groupc

        self.mapmax=mapmax

        self.mapping = Mapping1big(midb, midc, 2*groupc)
        self.g1lr=PRelu1(groupc,bound=1,bias=False)
        self.h1conv=Conv1dGroupLayerTupleOp(groupc,groupc,L=11,groups=groupc)
        self.h1lr1=PRelu1(groupc,bound=1,bias=False)
        self.h1lr2=PRelu1(groupc,bound=1,bias=False)

        self.h3lr=PRelu1(groupc,bound=1,bias=True)
        self.trunkconv1=nn.Conv2d(groupc,groupc,groups=groupc//4,kernel_size=1,padding=0,bias=True)
        self.trunklr1=PRelu1(groupc,bound=1,bias=False)
        self.trunkconv2=nn.Conv2d(groupc,groupc,groups=groupc,kernel_size=3,padding=1,bias=False)
        self.trunklr2p=PRelu1(groupc,bound=1,bias=True)

        self.policy_linear=nn.Conv2d(groupc,1,kernel_size=1,padding=0,bias=False)

        self.trunklr2v=PRelu1(groupc,bound=1,bias=True)
        self.valuelr=PRelu1(groupc,bound=10,bias=True)
        self.value_linear1=nn.Linear(groupc,mlpc)
        self.value_linear2=nn.Linear(mlpc,mlpc)
        self.value_linear3=nn.Linear(mlpc,mlpc)
        self.value_linearfinal=nn.Linear(mlpc,3)

    def forward(self, x):
        mapf = self.mapping(x)


        if(self.mapmax!=0):
            mapf=self.mapmax*torch.tanh(mapf/self.mapmax) # |map|<30

        g1=mapf[:,:,:self.groupc,:,:]#第一组通道
        g2=mapf[:,:,self.groupc:,:,:]#第二组通道
        h1=self.g1lr(g1.mean(1))#四线求和再leakyrelu

        #以下几行是对h1进行对称的卷积。把h1卷积一次，再把h1翻转一下再卷积一次，两次的取平均，相当于对称的卷积核
        h1sym=torch.flip(h1,[2,3])
        h1=(h1,h1,h1,h1)
        h1sym=(h1sym,h1sym,h1sym,h1sym)
        h1=torch.stack(self.h1conv(h1),dim=1)#沿着另一条线卷积
        h1sym=torch.stack(self.h1conv(h1sym),dim=1)#等价于卷积核反向
        h1sym=torch.flip(h1sym,[3,4])
        h1=(h1+h1sym)/2#正向和反向取平均，相当于对称卷积核

        h2=self.h1lr2(self.h1lr1(h1,dim=2)+g2,dim=2)
        h3=h2.mean(1)#最后把四条线整合起来

        trunk=self.h3lr(h3)
        trunk=self.trunkconv1(trunk)
        trunk=self.trunklr1(trunk)
        trunk=self.trunkconv2(trunk)

        p=self.trunklr2p(trunk)
        v=self.trunklr2v(trunk)


        p=self.policy_linear(p)

        v=v.mean((2,3))
        v=self.valuelr(v)
        v=self.value_linear1(v)
        v=torch.relu(v)
        v=self.value_linear2(v)
        v=torch.relu(v)
        v=self.value_linear3(v)
        v=torch.relu(v)
        v=self.value_linearfinal(v)
        return v, p


ModelDic = {
    "res": Model_ResNet, #resnet对照组
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
    "mix4convep": Model_mix4convep, #实验
    "mix6": Model_mix6,
    "mix6m1big": Model_mix6m1big,
    "mix6m1dif": Model_mix6m1dif,
    "mix6m2": Model_mix6m2,
    "mix6m2c": Model_mix6m2c,
    "mix6m3": Model_mix6m3,
    "mix6m3conv": Model_mix6m3conv,

    "v1": Model_mix6,
    "v2a": Model_v2a,
    "v2b": Model_v2b,
    "v2c": Model_v2c,
    "v2d": Model_v2d,
    "v2ba": Model_v2ba,
    "v2bsym": Model_v2bsym,
    "v2bb": Model_v2bb,
    "v2bc": Model_v2bc,
    "v2bd": Model_v2bd,
    "v2bca": Model_v2bca,
    "v2bcb": Model_v2bcb,
    "v2bcc": Model_v2bcc,
    "v2bcd": Model_v2bcd,
    "v2bce": Model_v2bce,
    "v2bcf": Model_v2bcf,
    "v2bcg": Model_v2bcg,
}
