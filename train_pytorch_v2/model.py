
import torch
import torch.nn as nn
import numpy as np
from config import *
input_c=3



def swish(x):
    return torch.sigmoid(x)*x
def tupleOp(f,x):
    return (f(x[0]),f(x[1]),f(x[2]),f(x[3]))

def conv3x3symmetry(x,w,b,groups=1):

    out_c=w.shape[1]
    in_c=w.shape[2]
    w=torch.stack((w[2],w[1],w[2],
                    w[1],w[0],w[1],
                    w[2],w[1],w[2],),dim=2)
    w=w.view(out_c,in_c,3,3)

    x = torch.conv2d(x,w,b,padding=1,groups=groups)
    return x
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
class Conv3x3LayerSymmetry(nn.Module):
    def __init__(self, in_c, out_c,groups,bias=True):
        super().__init__()
        self.groups=groups
        self.w = nn.Parameter(torch.empty((3,out_c, in_c//groups)), True)
        nn.init.kaiming_uniform_(self.w)
        if(bias):
            self.b = nn.Parameter(torch.zeros((out_c,)), True)
        else:
            self.b=None


    def forward(self, x):
        y=conv3x3symmetry(x,self.w,self.b,groups=self.groups)
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

    def export_slope(self):
        slope = self.slope
        if(self.bound>0):
            slope=torch.tanh(slope/self.bound)*self.bound
        return slope.data.cpu().numpy()


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

class Model_v2(nn.Module):
    # =v2bcge
    def __init__(self, mapb=5, mapc=256, groupc=64, mlpc=64, mapmax=30):  # 1d卷积通道，policy通道，胜负和通道
        super().__init__()
        self.model_type = "v2"
        self.model_param = (mapb, mapc, groupc, mlpc, mapmax)
        self.groupc = groupc

        self.mapmax = mapmax

        self.mapping = Mapping1big(mapb, mapc, 2 * groupc)
        self.g1lr = PRelu1(groupc, bound=0.999, bias=False)
        self.h1conv = Conv1dGroupLayerTupleOp(groupc, groupc, L=11, groups=groupc)
        self.h1lr1 = PRelu1(groupc, bound=0.999, bias=False)
        self.h1lr2 = PRelu1(groupc, bound=0.999, bias=False)

        self.h3lr = PRelu1(groupc, bound=0.999, bias=True)
        self.trunkconv1 = nn.Conv2d(groupc, groupc, groups=groupc // 4, kernel_size=1, padding=0, bias=True)
        self.trunklr1 = PRelu1(groupc, bound=0.999, bias=False)
        self.trunkconv2 = Conv3x3LayerSymmetry(groupc, groupc, groups=groupc, bias=False)
        self.trunklr2p = PRelu1(groupc, bound=0.999, bias=True)

        self.policy_linear = nn.Conv2d(groupc, 1, kernel_size=1, padding=0, bias=False)

        self.trunklr2v = PRelu1(groupc, bound=0.999, bias=True)
        self.valuelr = PRelu1(groupc, bound=10, bias=True)
        self.value_linear1 = nn.Linear(groupc, mlpc)
        self.value_linear2 = nn.Linear(mlpc, mlpc)
        self.value_linear3 = nn.Linear(mlpc, mlpc)
        self.value_linearfinal = nn.Linear(mlpc, 3)

    def forward(self, x, mapnoise=0):
        mapf = self.mapping(x)

        if (self.mapmax != 0):
            mapf = self.mapmax * torch.tanh(mapf / self.mapmax)  # <30

        if(mapnoise!=0):
            mapf=mapf+mapnoise*torch.randn(mapf.shape,device=mapf.device)

        g1 = mapf[:, :, :self.groupc, :, :]  # 第一组通道
        g2 = mapf[:, :, self.groupc:, :, :]  # 第二组通道
        h1 = self.g1lr(g1.mean(1))  # 四线求和再leakyrelu   # avx代码里改用sum，<120


        # 以下几行是对h1进行对称的卷积。把h1卷积一次，再把h1翻转一下再卷积一次，两次的取平均，相当于对称的卷积核
        h1sym = torch.flip(h1, [2, 3])
        h1 = (h1, h1, h1, h1)
        h1sym = (h1sym, h1sym, h1sym, h1sym)
        h1c = torch.stack(self.h1conv(h1), dim=1)  # 沿着另一条线卷积
        h1csym = torch.stack(self.h1conv(h1sym), dim=1)  # 等价于卷积核反向
        h1csym = torch.flip(h1csym, [3, 4])
        h1c = (h1c + h1csym) / 2  # 正向和反向取平均，相当于对称卷积核

        h2 = self.h1lr2(self.h1lr1(h1c, dim=2) + g2, dim=2)
        h3 = h2.mean(1)  # 最后把四条线整合起来

        trunk = self.h3lr(h3)

        # 交换通道分组顺序，方便avx2计算
        trunk = trunk.reshape(trunk.shape[0], -1, 4, 4, trunk.shape[2], trunk.shape[3])
        trunk = torch.transpose(trunk, 2, 3)
        trunk = trunk.reshape(trunk.shape[0], -1, trunk.shape[4], trunk.shape[5])

        trunk = self.trunkconv1(trunk)

        # 交换通道分组顺序，方便avx2计算
        trunk = trunk.reshape(trunk.shape[0], -1, 4, 4, trunk.shape[2], trunk.shape[3])
        trunk = torch.transpose(trunk, 2, 3)
        trunk = trunk.reshape(trunk.shape[0], -1, trunk.shape[4], trunk.shape[5])

        trunk = self.trunklr1(trunk)
        trunk = self.trunkconv2(trunk)

        p = self.trunklr2p(trunk)
        v = self.trunklr2v(trunk)

        p = self.policy_linear(p)

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


    def exportMapTable(self, x, device):  # x.shape=(h,w<=9)
        b = (x == 1)
        w = (x == 2)
        x = np.stack((b, w), axis=0).astype(np.float32)

        x = torch.tensor(x[np.newaxis], dtype=torch.float32, device=device)# x.shape=(n=1,c=2,h,w<=9)

        with torch.no_grad():
            batchsize=1024
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

ModelDic = {
    "res": Model_ResNet, #resnet对照组
    "mix6": Model_mix6,
    "mix6m1big": Model_mix6m1big,

    "v1": Model_mix6,
    "v2": Model_v2,
}
