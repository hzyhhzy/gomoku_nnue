
import torch
import torch.nn as nn
import numpy as np
from config import *

rules_c=30


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



class Conv0dResnetBlockTupleOp(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.conv1=nn.Conv2d(c,c,1,1,0)
        self.conv2=nn.Conv2d(c,c,1,1,0)


    def forward(self, x, mask):
        y=tupleOp(self.conv1,x)
        y=tupleOp(torch.relu,y)
        y=(mask*y[0],mask*y[1],mask*y[2],mask*y[3],)
        y=tupleOp(self.conv2,y)
        y=tupleOp(torch.relu,y)
        y=(y[0]+x[0],y[1]+x[1],y[2]+x[2],y[3]+x[3])
        y=(mask*y[0],mask*y[1],mask*y[2],mask*y[3],)
        return y


class Conv1dResnetBlockTupleOp(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.conv1=Conv1dLayerTupleOp(c,c)
        self.conv2=nn.Conv2d(c,c,1,1,0)


    def forward(self, x, mask):
        y=self.conv1(x)
        y=tupleOp(torch.relu,y)
        y=(mask*y[0],mask*y[1],mask*y[2],mask*y[3],)
        y=tupleOp(self.conv2,y)
        y=tupleOp(torch.relu,y)
        y=(y[0]+x[0],y[1]+x[1],y[2]+x[2],y[3]+x[3])
        y=(mask*y[0],mask*y[1],mask*y[2],mask*y[3],)
        return y


class gfVector(nn.Module):
    def __init__(self,in_c,c):
        super().__init__()
        self.layer1=nn.Linear(in_c,c)
        self.layer2=nn.Linear(c,c)

    def forward(self, rules):
        x=self.layer1(rules)
        #x=torch.relu(x)
        #x=self.layer2(x)
        return x

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



    def forward(self, x,mask):
        x=x[:,0:2]
        y=self.firstConv((x,x,x,x))
        y=tupleOp(torch.relu,y)
        y=(mask*y[0],mask*y[1],mask*y[2],mask*y[3],)
        y=self.conv1(y,mask)
        y=self.conv2(y,mask)
        y=self.conv3(y,mask)
        y=self.conv4(y,mask)
        for block in self.trunk:
            y = block(y,mask)
        y=tupleOp(self.finalconv,y)
        y=(mask*y[0],mask*y[1],mask*y[2],mask*y[3],)
        y=torch.stack(y,dim=1)#shape=(n,4,c,h,w)

        return y





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
        self.conv1 = CNNLayer(inout_c, mid_c)
        self.conv2 = CNNLayer(mid_c, inout_c)

    def forward(self, x, mask):
        y = self.conv1(x)
        y = y * mask
        y = self.conv2(y)
        y = y + x
        y = y * mask
        return y

class Outputhead_v1(nn.Module):

    def __init__(self,out_c,head_mid_c):
        super().__init__()
        self.cnn=CNNLayer(out_c, head_mid_c)
        self.valueHeadLinear = nn.Linear(head_mid_c, 3)
        self.policyHeadLinear = nn.Conv2d(head_mid_c, 1, 1)

    def forward(self, h, mask):
        x=self.cnn(h)
        x=x*mask

        # value head
        value = x.mean((2, 3))/mask.mean((2,3))
        value = self.valueHeadLinear(value)

        # policy head
        policy = mask*self.policyHeadLinear(x)-(1-mask)*100
        policy = policy.squeeze(1)

        return value, policy



class Model_ResNet(nn.Module):

    def __init__(self,b,f):
        super().__init__()
        self.model_type = "res"
        self.model_param=(b,f)

        self.inputhead=CNNLayer(2+rules_c, f)
        self.trunk=nn.ModuleList()
        for i in range(b):
            self.trunk.append(ResnetLayer(f,f))
        self.outputhead=Outputhead_v1(f,f)

    def forward(self, x):
        mask=x[:,0:1,:,:]
        #mask=mask*0+1
        x=x[:,1:,:,:]
        h=self.inputhead(x)
        h=h*mask

        for block in self.trunk:
            h=block(h,mask)

        return self.outputhead(h,mask)

class Model_v2(nn.Module):
    #
    def __init__(self, mapb=5, mapc=256, groupc=64, mlpc=64, mapmax=30):  # 1d卷积通道，policy通道，胜负和通道
        super().__init__()
        self.model_type = "v2"
        self.model_param = (mapb, mapc, groupc, mlpc, mapmax)
        self.groupc = groupc

        self.mapmax = mapmax

        self.mapping = Mapping1big(mapb, mapc, 2 * groupc)
        self.gfVector=gfVector(rules_c+3,groupc)
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
        self.value_linearfinal = nn.Linear(mlpc, 3+1) #including pass

    def forward(self, x, mapnoise=0):
        mask=x[:,0:1,:,:]
        assert(mask.shape[1]==1)
        assert(mask.shape[2]==boardH)
        assert(mask.shape[3]==boardW)
        boardHs=mask.sum(dim=2).max(dim=2)[0].squeeze(1)
        boardWs=mask.sum(dim=3).max(dim=2)[0].squeeze(1)
        boardArea=boardHs*boardWs
        boardAreaInput=torch.stack((
            boardArea/225-1,
            torch.sqrt(boardArea/225)-1,
            (boardHs-boardWs)*(boardHs-boardWs)/boardArea
        ),dim=1)
        #print(boardArea)
        rules=x[:,3:,0,0]
        gf=torch.cat((boardAreaInput,rules),dim=1)
        x=x[:,1:3,:,:]
        mapf = self.mapping(x,mask)

        if (self.mapmax != 0):
            mapf = self.mapmax * torch.tanh(mapf / self.mapmax)  # <30

        if(mapnoise!=0):
            mapf=mapf+mapnoise*torch.randn(mapf.shape,device=mapf.device)

        mapf=mapf*mask.unsqueeze(1)

        g1 = mapf[:, :, :self.groupc, :, :]  # 第一组通道
        g2 = mapf[:, :, self.groupc:, :, :]  # 第二组通道
        rv=self.gfVector(gf)
        max_rv=torch.max(torch.abs(rv)).item()
        if(max_rv>30):
            print("max_rv =",max_rv)
        #print(rv1.shape)
        h1 = self.g1lr(g1.mean(1)+rv.view(rv.shape[0],rv.shape[1],1,1))  # 四线求和再leakyrelu   # avx代码里改用sum，<120


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

        # 交换回来
        trunk = trunk.reshape(trunk.shape[0], -1, 4, 4, trunk.shape[2], trunk.shape[3])
        trunk = torch.transpose(trunk, 2, 3)
        trunk = trunk.reshape(trunk.shape[0], -1, trunk.shape[4], trunk.shape[5])

        trunk = self.trunklr1(trunk)
        trunk = self.trunkconv2(trunk*mask)


        #print("max_trunk =",torch.max(torch.abs(trunk)).item())

        p = self.trunklr2p(trunk)
        v = self.trunklr2v(trunk)

        p = self.policy_linear(p)
        p=p-(1-mask)*100

        v = v.mean((2, 3))/mask.mean((2,3))
        v = self.valuelr(v)
        v = self.value_linear1(v)
        v = torch.relu(v)
        v = self.value_linear2(v)
        v = torch.relu(v)
        v = self.value_linear3(v)
        v = torch.relu(v)
        v = self.value_linearfinal(v)

        p_pass=v[:,3].reshape(-1,1)
        p=torch.cat((torch.flatten(p,start_dim=1),p_pass),dim=1)
        v=v[:,0:3]

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
                mask=torch.ones((x.shape[0],1,end-start,x.shape[3]), dtype=torch.float32, device=device)
                map=self.mapping(x[:, :, start:end, :],mask)[0, 0]
                if(self.mapmax!=0):
                    map=self.mapmax*torch.tanh(map/self.mapmax) # |map|<30

                buf.append(map.to('cpu').numpy())  # map.shape=(n=1,4,c=pc+vc,h,w<=9)
            buf=np.concatenate(buf,axis=1)
            return buf # shape=(c=pc+vc,h,w<=9)

    def exportDefaultGFVector(self, device): # gf vector for 15x15 board, normal rule
        gf=torch.zeros((1,33), dtype=torch.float32, device=device)
        #gf = 0.01*torch.arange(33, dtype=torch.float32, device=device).reshape((1,33))
        with torch.no_grad():
            gfv=self.gfVector(gf)
            gfv=gfv.to('cpu').numpy().reshape(-1)
        return gfv


class Model_v2_noOppVCF(nn.Module):
    #
    def __init__(self, mapb=5, mapc=256, groupc=64, mlpc=64, mapmax=30):  # 1d卷积通道，policy通道，胜负和通道
        super().__init__()
        self.model_type = "v2_noOppVCF"
        self.model_param = (mapb, mapc, groupc, mlpc, mapmax)
        self.groupc = groupc

        self.mapmax = mapmax

        self.mapping = Mapping1big(mapb, mapc, 2 * groupc)
        self.gfVector=gfVector(rules_c+3,groupc)
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
        self.value_linearfinal = nn.Linear(mlpc, 3+1) #including pass

    def forward(self, x, mapnoise=0):
        mask=x[:,0:1,:,:]
        assert(mask.shape[1]==1)
        assert(mask.shape[2]==boardH)
        assert(mask.shape[3]==boardW)
        boardHs=mask.sum(dim=2).max(dim=2)[0].squeeze(1)
        boardWs=mask.sum(dim=3).max(dim=2)[0].squeeze(1)
        boardArea=boardHs*boardWs
        boardAreaInput=torch.stack((
            boardArea/225-1,
            torch.sqrt(boardArea/225)-1,
            (boardHs-boardWs)*(boardHs-boardWs)/boardArea
        ),dim=1)
        #print(boardArea)
        rules=x[:,3:,0,0]
        rules[:,2:5]=0
        gf=torch.cat((boardAreaInput,rules),dim=1)
        x=x[:,1:3,:,:]
        mapf = self.mapping(x,mask)

        if (self.mapmax != 0):
            mapf = self.mapmax * torch.tanh(mapf / self.mapmax)  # <30

        if(mapnoise!=0):
            mapf=mapf+mapnoise*torch.randn(mapf.shape,device=mapf.device)

        mapf=mapf*mask.unsqueeze(1)

        g1 = mapf[:, :, :self.groupc, :, :]  # 第一组通道
        g2 = mapf[:, :, self.groupc:, :, :]  # 第二组通道
        rv=self.gfVector(gf)
        max_rv=torch.max(torch.abs(rv)).item()
        if(max_rv>30):
            print("max_rv =",max_rv)
        #print(rv1.shape)
        h1 = self.g1lr(g1.mean(1)+rv.view(rv.shape[0],rv.shape[1],1,1))  # 四线求和再leakyrelu   # avx代码里改用sum，<120


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

        # 交换回来
        trunk = trunk.reshape(trunk.shape[0], -1, 4, 4, trunk.shape[2], trunk.shape[3])
        trunk = torch.transpose(trunk, 2, 3)
        trunk = trunk.reshape(trunk.shape[0], -1, trunk.shape[4], trunk.shape[5])

        trunk = self.trunklr1(trunk)
        trunk = self.trunkconv2(trunk*mask)

        p = self.trunklr2p(trunk)
        v = self.trunklr2v(trunk)

        p = self.policy_linear(p)
        p=p-(1-mask)*100

        v = v.mean((2, 3))/mask.mean((2,3))
        v = self.valuelr(v)
        v = self.value_linear1(v)
        v = torch.relu(v)
        v = self.value_linear2(v)
        v = torch.relu(v)
        v = self.value_linear3(v)
        v = torch.relu(v)
        v = self.value_linearfinal(v)

        p_pass=v[:,3].reshape(-1,1)
        p=torch.cat((torch.flatten(p,start_dim=1),p_pass),dim=1)
        v=v[:,0:3]

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

                mask=torch.ones((x.shape[0],1,end-start,x.shape[3]), dtype=torch.float32, device=device)
                map=self.mapping(x[:, :, start:end, :],mask)[0, 0]
                if(self.mapmax!=0):
                    map=self.mapmax*torch.tanh(map/self.mapmax) # |map|<30

                buf.append(map.to('cpu').numpy())  # map.shape=(n=1,4,c=pc+vc,h,w<=9)
            buf=np.concatenate(buf,axis=1)
            return buf # shape=(c=pc+vc,h,w<=9)

    def exportDefaultGFVector(self, device):  # gf vector for 15x15 board, normal rule
        gf = torch.zeros((1, 33), dtype=torch.float32, device=device)

        with torch.no_grad():
            gfv = self.gfVector(gf)
            gfv = gfv.to('cpu').numpy().reshape(-1)
        return gfv


class Model_v3(nn.Module):
    #mlp output as policy linear weight
    def __init__(self, mapb=5, mapc=256, groupc=64, mlpc=64, mapmax=30):  # 1d卷积通道，policy通道，胜负和通道
        super().__init__()
        self.model_type = "v3"
        self.model_param = (mapb, mapc, groupc, mlpc, mapmax)
        self.groupc = groupc

        self.mapmax = mapmax

        self.mapping = Mapping1big(mapb, mapc, 2 * groupc)
        self.gfVector=gfVector(rules_c+3,groupc)
        self.g1lr = PRelu1(groupc, bound=0.999, bias=False)
        self.h1conv = Conv1dGroupLayerTupleOp(groupc, groupc, L=11, groups=groupc)
        self.h1lr1 = PRelu1(groupc, bound=0.999, bias=False)
        self.h1lr2 = PRelu1(groupc, bound=0.999, bias=False)

        self.h3lr = PRelu1(groupc, bound=0.999, bias=True)
        self.trunkconv1 = nn.Conv2d(groupc, groupc, groups=groupc // 4, kernel_size=1, padding=0, bias=True)
        self.trunklr1 = PRelu1(groupc, bound=0.999, bias=False)
        self.trunkconv2 = Conv3x3LayerSymmetry(groupc, groupc, groups=groupc, bias=False)
        self.trunklr2 = PRelu1(groupc, bound=0.999, bias=True)

        #self.policy_linear = nn.Conv2d(groupc, 1, kernel_size=1, padding=0, bias=False)

        #self.trunklr2v = PRelu1(groupc, bound=0.999, bias=True)
        self.valuelr = PRelu1(groupc, bound=10, bias=True)
        self.value_linear1 = nn.Linear(groupc, mlpc)
        self.value_linear2 = nn.Linear(mlpc, mlpc)
        self.value_linear3 = nn.Linear(mlpc, mlpc)
        self.value_linearfinal = nn.Linear(mlpc, 3+1) #including pass
        self.mlp_policy_w = nn.Linear(mlpc, groupc) #including pass
        self.mlp_plr = PRelu1(groupc, bound=10, bias=False)

    def forward(self, x, mapnoise=0):
        mask=x[:,0:1,:,:]
        assert(mask.shape[1]==1)
        assert(mask.shape[2]==boardH)
        assert(mask.shape[3]==boardW)
        boardHs=mask.sum(dim=2).max(dim=2)[0].squeeze(1)
        boardWs=mask.sum(dim=3).max(dim=2)[0].squeeze(1)
        boardArea=boardHs*boardWs
        boardAreaInput=torch.stack((
            boardArea/225-1,
            torch.sqrt(boardArea/225)-1,
            (boardHs-boardWs)*(boardHs-boardWs)/boardArea
        ),dim=1)
        #print(boardArea)
        rules=x[:,3:,0,0]
        gf=torch.cat((boardAreaInput,rules),dim=1)
        x=x[:,1:3,:,:]
        mapf = self.mapping(x,mask)

        if (self.mapmax != 0):
            mapf = self.mapmax * torch.tanh(mapf / self.mapmax)  # <30

        if(mapnoise!=0):
            mapf=mapf+mapnoise*torch.randn(mapf.shape,device=mapf.device)

        mapf=mapf*mask.unsqueeze(1)

        g1 = mapf[:, :, :self.groupc, :, :]  # 第一组通道
        g2 = mapf[:, :, self.groupc:, :, :]  # 第二组通道
        rv=self.gfVector(gf)
        max_rv=torch.max(torch.abs(rv)).item()
        if(max_rv>30):
            print("max_rv =",max_rv)
        #print(rv1.shape)
        h1 = self.g1lr(g1.mean(1)+rv.view(rv.shape[0],rv.shape[1],1,1))  # 四线求和再leakyrelu   # avx代码里改用sum，<120


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

        # 交换回来
        trunk = trunk.reshape(trunk.shape[0], -1, 4, 4, trunk.shape[2], trunk.shape[3])
        trunk = torch.transpose(trunk, 2, 3)
        trunk = trunk.reshape(trunk.shape[0], -1, trunk.shape[4], trunk.shape[5])

        trunk = self.trunklr1(trunk)
        trunk = self.trunkconv2(trunk*mask)

        trunk = self.trunklr2(trunk)


        v = trunk.mean((2, 3))/mask.mean((2,3))
        v = self.valuelr(v)
        v = self.value_linear1(v)
        v = torch.relu(v)
        v = self.value_linear2(v)
        v = torch.relu(v)
        v = self.value_linear3(v)
        v = torch.relu(v)
        value = self.value_linearfinal(v)
        mlp_p = self.mlp_policy_w(v)

        mlp_p=self.mlp_plr(mlp_p,dim=1)
        mlp_p=torch.clip(mlp_p,-0.99,0.99)
        p=torch.einsum("nchw,nc->nhw",trunk,mlp_p).reshape((-1,1,boardH,boardW))
        p=p-(1-mask)*100

        p_pass=value[:,3].reshape(-1,1)
        p=torch.cat((torch.flatten(p,start_dim=1),p_pass),dim=1)
        value=value[:,0:3]

        return value, p


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
                mask=torch.ones((x.shape[0],1,end-start,x.shape[3]), dtype=torch.float32, device=device)
                map=self.mapping(x[:, :, start:end, :],mask)[0, 0]
                if(self.mapmax!=0):
                    map=self.mapmax*torch.tanh(map/self.mapmax) # |map|<30

                buf.append(map.to('cpu').numpy())  # map.shape=(n=1,4,c=pc+vc,h,w<=9)
            buf=np.concatenate(buf,axis=1)
            return buf # shape=(c=pc+vc,h,w<=9)

    def exportDefaultGFVector(self, device): # gf vector for 15x15 board, normal rule
        gf=torch.zeros((1,33), dtype=torch.float32, device=device)
        #gf = 0.01*torch.arange(33, dtype=torch.float32, device=device).reshape((1,33))
        with torch.no_grad():
            gfv=self.gfVector(gf)
            gfv=gfv.to('cpu').numpy().reshape(-1)
        return gfv

ModelDic = {
    "res": Model_ResNet, #resnet对照组
    "v2": Model_v2,
    "v2_noOppVCF": Model_v2_noOppVCF,
    "v3": Model_v3,
}
