
import torch
import torch.nn as nn
import numpy as np
from config import *
assert(UseVariousBoardsize) # many V2 model input assert UseVariousBoardsize

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


class LineMapping(nn.Module):

    def __init__(self,L,b,midc,outc):
        super().__init__()
        self.L=L
        self.midc=midc
        self.outc=outc
        self.firstConv=Conv1dLayerTupleOp(2,midc)#len=3
        self.conv1=Conv1dResnetBlockTupleOp(midc)#len=5
        self.conv2=Conv1dResnetBlockTupleOp(midc)#len=7
        self.conv3=Conv1dResnetBlockTupleOp(midc)#len=9
        if(self.L>=11):
            self.conv4=Conv1dResnetBlockTupleOp(midc)#len=11
        if(self.L>=13):
            self.conv5=Conv1dResnetBlockTupleOp(midc)#len=13
        assert(L>=9 and L<=13 and L%2==1)

        self.trunk = nn.ModuleList()
        for i in range(b):
            self.trunk.append(Conv0dResnetBlockTupleOp(midc))
        self.finalconv=nn.Conv2d(midc,outc,1,1,0)



    def forward(self, x,mask):
        assert(x.shape[1]==2) # my stones and opp stones
        y=self.firstConv((x,x,x,x))
        y=tupleOp(torch.relu,y)
        y=(mask*y[0],mask*y[1],mask*y[2],mask*y[3],)
        y=self.conv1(y,mask)
        y=self.conv2(y,mask)
        y=self.conv3(y,mask)
        if(self.L>=11):
            y=self.conv4(y,mask)
        if(self.L>=13):
            y=self.conv5(y,mask)

        for block in self.trunk:
            y = block(y,mask)
        y=tupleOp(self.finalconv,y)
        y=(mask*y[0],mask*y[1],mask*y[2],mask*y[3],)
        y=torch.stack(y,dim=1)#shape=(n,4,c,h,w)

        return y

#mapping with a "legalmap" input: 3^L * 2
class LineMappingC3(nn.Module):

    def __init__(self,L,b,midc,outc):
        super().__init__()
        self.L=L
        self.midc=midc
        self.outc=outc
        self.firstConv=Conv1dLayerTupleOp(2,midc)#len=3
        self.conv1=Conv1dResnetBlockTupleOp(midc)#len=5
        self.conv2=Conv1dResnetBlockTupleOp(midc)#len=7
        self.conv3=Conv1dResnetBlockTupleOp(midc)#len=9
        if(self.L>=11):
            self.conv4=Conv1dResnetBlockTupleOp(midc)#len=11
        if(self.L>=13):
            self.conv5=Conv1dResnetBlockTupleOp(midc)#len=13
        assert(L>=9 and L<=13 and L%2==1)

        self.c3conv=nn.Conv2d(1,midc,1,1,0)
        self.trunk = nn.ModuleList()
        for i in range(b):
            self.trunk.append(Conv0dResnetBlockTupleOp(midc))
        self.finalconv=nn.Conv2d(midc,outc,1,1,0)



    def forward(self, x, c3, mask):
        assert(x.shape[1]==2) # my stones and opp stones
        assert(c3.shape[1]==1) # legalmap
        y=self.firstConv((x,x,x,x))
        y=tupleOp(torch.relu,y)
        y=(mask*y[0],mask*y[1],mask*y[2],mask*y[3],)
        y=self.conv1(y,mask)
        y=self.conv2(y,mask)
        y=self.conv3(y,mask)
        if(self.L>=11):
            y=self.conv4(y,mask)
        if(self.L>=13):
            y=self.conv5(y,mask)

        c3=self.c3conv(c3)
        y=(c3+y[0],c3+y[1],c3+y[2],c3+y[3],)

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

    def forward(self, x, mask):
        y = self.conv(x)
        y = self.bn(y)
        y = torch.relu(y)
        if(mask is not None):
            y = y * mask
        return y


class ResnetLayer(nn.Module):
    def __init__(self, inout_c, mid_c):
        super().__init__()
        self.conv1 = CNNLayer(inout_c, mid_c)
        self.conv2 = CNNLayer(mid_c, inout_c)

    def forward(self, x, mask):
        if(mask is not None):
            x = x * mask
        y = self.conv1(x, mask)
        y = y
        y = self.conv2(y, mask)
        y = y + x
        return y

class Outputhead_resnet(nn.Module):

    def __init__(self,out_c,head_mid_c):
        super().__init__()
        self.cnn=CNNLayer(out_c, head_mid_c)
        self.valueHeadLinear = nn.Linear(head_mid_c, 3 if NoPass else 4)
        self.policyHeadLinear = nn.Conv2d(head_mid_c, 1, 1, bias=False)

    def forward(self, h, mask):
        x=self.cnn(h, mask)

        # value head
        value = x.mean((2, 3))
        if(mask is not None):
            value = value/mask.mean((2,3))
        value = self.valueHeadLinear(value)

        # policy head
        policy = self.policyHeadLinear(x)
        if(mask is not None):
            policy -= (1-mask)*100
        policy = policy.flatten(1)

        if not NoPass:
            passPolicy=value[:,-1:]
            policy=torch.cat((policy,passPolicy), dim=1)
            value=value[:,:-1]

        return value, policy



class Model_ResNet(nn.Module):

    def __init__(self,b,f):
        super().__init__()
        self.model_type = "res"
        self.model_param=(b,f)
        self.input_c = BoardC + GlobalC
        if(UseVariousBoardsize): # 0th channel of BoardC is mask, not input
            self.input_c -= 1
        self.inputhead=CNNLayer(self.input_c, f)
        self.trunk=nn.ModuleList()
        for i in range(b):
            self.trunk.append(ResnetLayer(f,f))
        self.outputhead=Outputhead_resnet(f,f)

    def forward(self, bf, gf):
        mask=None
        if(UseVariousBoardsize): # 0th channel of BoardC is mask of board area, not input
            mask=bf[:,0:1,:,:]
            bf=bf[:,1:,:,:]
        #concat gf
        gf_expanded=gf.view(gf.shape[0], gf.shape[1], 1, 1).expand(-1,-1,bf.shape[2],bf.shape[3])
        x = torch.cat((bf, gf_expanded) , dim=1)

        h=self.inputhead(x, mask)

        for block in self.trunk:
            h=block(h,mask)

        return self.outputhead(h,mask)

def getIllegalMap(bf,gf):
    
    enable_illegalmap=gf[:,0].view(-1,1,1,1) #stage=1 in katago
    illegalmap=enable_illegalmap*(1.0-bf[:,4].unsqueeze(1))*bf[:,0].unsqueeze(1)
    return illegalmap


#tested, worst than getIllegalMap(bf,gf):
def getIllegalMapV2(bf,gf):
    assert(false)
    #only banned location from priority value
    enable_illegalmap=gf[:,0].view(-1,1,1,1) #stage=1 in katago
    board_mask=bf[:,0].unsqueeze(1)
    no_stone=1.0-bf[:,1].unsqueeze(1)-bf[:,2].unsqueeze(1)
    illegal_moves=1.0-bf[:,4].unsqueeze(1)
    illegalmap=enable_illegalmap*board_mask*no_stone*illegal_moves
    if(False):
    #if(gf[0,0]>0.5):
        #print(illegalmap[0,0,3:-3,3:-3])
        debug_var=enable_illegalmap*(bf[:,1].unsqueeze(1)+bf[:,2].unsqueeze(1)+bf[:,3].unsqueeze(1)+illegalmap-(1-bf[:,0].unsqueeze(1)))
        print("--------")
        print(debug_var[0,0,2:-2,5:])
        print(illegalmap[0,0,2:-2,5:])
        print((bf[:,1].unsqueeze(1)+bf[:,2].unsqueeze(1))[0,0,2:-2,5:])
    return illegalmap


# v1 ("mix" series)
class Model_v1(nn.Module):
    def __init__(self, mapb=5, mapc=256, c=128, pc=64, mlpc=48, mapmax=30):  # 1d卷积通道，policy通道，胜负和通道
        super().__init__()
        self.model_type = "v1"
        self.info = "mix6 + some modification from dblue's mix9 + global vector"
        self.model_param = (mapb, mapc, c, pc, mlpc, mapmax)
        self.c = c
        self.pc = pc

        self.mapmax = mapmax

        self.mapping = LineMapping(13, mapb, mapc, c)
        self.illegalVector = nn.Parameter(torch.empty(c)) #add a bias to all illegal locations        
        nn.init.normal_(self.illegalVector, mean=0.0, std=0.02) # 使用正态分布进行初始化

        self.gfVector=gfVector(GlobalC+BoardSizeC, c)
        self.map_leakyrelu = PRelu1(c, bound=1.999, bias=False)
        self.policy_conv=nn.Conv2d(pc,pc,kernel_size=3,padding=1,bias=True,groups=pc)
        
        #self.trunklr2v = PRelu1(groupc, bound=0.999, bias=True)
        self.valuelr = PRelu1(c, bound=10, bias=True)
        self.value_linear1 = nn.Linear(c, mlpc)
        self.value_linear2 = nn.Linear(mlpc, mlpc)
         
        self.value_linearfinal = nn.Linear(mlpc, 3 if NoPass else 4) #including pass
        self.mlp_policy_w = nn.Linear(mlpc, pc) #including pass
        self.mlp_plr = PRelu1(pc, bound=10, bias=False)

    def forward(self, bf, gf, mapnoise=0):
        mask=None
        if UseVariousBoardsize:
            mask=bf[:,0:1,:,:]
            assert(mask.shape[1]==1)
            assert(mask.shape[2]==BoardH)
            assert(mask.shape[3]==BoardW)
        else:
            mask=None
        boardHs=mask.sum(dim=2).max(dim=2)[0].squeeze(1)
        boardWs=mask.sum(dim=3).max(dim=2)[0].squeeze(1)
        boardArea=boardHs*boardWs
        boardAreaInput=torch.stack((
            boardArea/225-1,
            torch.sqrt(boardArea/225)-1,
            (boardHs-boardWs)*(boardHs-boardWs)/boardArea
        ),dim=1)
        assert(boardAreaInput.shape[1]==BoardSizeC)
        #print(boardArea)
        gf=torch.cat((boardAreaInput,gf),dim=1)
        rv=self.gfVector(gf)
        max_rv=torch.max(torch.abs(rv)).item()
        if(max_rv>30):
            print("max_rv =",max_rv)
        

        illegalVector=self.illegalVector
        if (self.mapmax != 0):
            illegalVector = self.mapmax * torch.tanh(illegalVector / self.mapmax)  # <30
        
        #bf[3] is where is last move
        #lastLocVector is a bias applied to last move location
        illegalmap=getIllegalMap(bf,gf)
        lb=illegalmap * illegalVector.view(1,self.c,1,1)


        bw=bf[:,1:3,:,:]
        mapf = self.mapping(bw, mask)
        if (self.mapmax != 0):
            mapf = self.mapmax * torch.tanh(mapf / self.mapmax)  # <30
        if(mapnoise!=0):
            mapf = mapf+mapnoise*torch.randn(mapf.shape,device=mapf.device)
        mapf=mapf*mask.unsqueeze(1)
        mapf = mapf.mean(1) # |map|<30
        mapf = mapf + \
            rv.view(rv.shape[0],rv.shape[1],1,1) + \
            lb 
        mapf=self.map_leakyrelu(mapf) # |map|<60 if slope<2

        

        

        v = mapf.mean((2, 3))/mask.mean((2,3))
        v = self.valuelr(v)
        v = self.value_linear1(v)
        v = torch.relu(v)
        v = self.value_linear2(v)
        v = torch.relu(v)
        value = self.value_linearfinal(v)
        mlp_p = self.mlp_policy_w(v)

        mlp_p=self.mlp_plr(mlp_p,dim=1)
        mlp_p=torch.clip(mlp_p,-0.999,0.999)
        
        p=mapf[:,:self.pc]
        p=self.policy_conv(p)
        p=torch.relu(p)
        
        p=torch.einsum("nchw,nc->nhw",p,mlp_p).reshape((-1,1,BoardH,BoardW))
        p=p-(1-mask)*100
        p=p-illegalmap*100

        p_pass=value[:,3].reshape(-1,1)
        p=torch.cat((torch.flatten(p,start_dim=1),p_pass),dim=1)
        value=value[:,0:3]

        return value, p






# small modifications on "v3a" in Gomoku_multirule
# mapping line length = 13
# baseline
class Model_v2(nn.Module):
    def __init__(self, mapb=5, mapc=256, groupc=64, mlpc=64, mlpc2=64, mapmax=30):  # 1d卷积通道，policy通道，胜负和通道
        super().__init__()
        self.model_type = "v2"
        self.info = "Using legal map instead of first stone's location"
        self.model_param = (mapb, mapc, groupc, mlpc, mlpc2, mapmax)
        self.groupc = groupc

        self.mapmax = mapmax

        self.mapping = LineMapping(13, mapb, mapc, 2 * groupc)
        self.illegalVector = nn.Parameter(torch.empty(2, groupc)) #add a bias to all illegal locations        
        nn.init.normal_(self.illegalVector, mean=0.0, std=0.02) # 使用正态分布进行初始化

        self.gfVector=gfVector(GlobalC+BoardSizeC, groupc)
        self.g1lr = PRelu1(groupc, bound=0.999, bias=False)
        self.h1conv = Conv1dGroupLayerTupleOp(groupc, groupc, L=13, groups=groupc)
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
        self.value_linear4 = nn.Linear(mlpc, mlpc2)
         
        self.value_linearfinal = nn.Linear(mlpc2, 3 if NoPass else 4) #including pass
        self.mlp_policy_w = nn.Linear(mlpc2, groupc) #including pass
        self.mlp_plr = PRelu1(groupc, bound=10, bias=False)

    def forward(self, bf, gf, mapnoise=0):
        illegalmap=getIllegalMap(bf,gf)
        mask=None
        if UseVariousBoardsize:
            mask=bf[:,0:1,:,:]
            assert(mask.shape[1]==1)
            assert(mask.shape[2]==BoardH)
            assert(mask.shape[3]==BoardW)
        else:
            mask=None
        boardHs=mask.sum(dim=2).max(dim=2)[0].squeeze(1)
        boardWs=mask.sum(dim=3).max(dim=2)[0].squeeze(1)
        boardArea=boardHs*boardWs
        boardAreaInput=torch.stack((
            boardArea/225-1,
            torch.sqrt(boardArea/225)-1,
            (boardHs-boardWs)*(boardHs-boardWs)/boardArea
        ),dim=1)
        assert(boardAreaInput.shape[1]==BoardSizeC)
        #print(boardArea)
        gf=torch.cat((boardAreaInput,gf),dim=1)
        bw=bf[:,1:3,:,:]
        mapf = self.mapping(bw, mask)

        illegalVector=self.illegalVector
        if (self.mapmax != 0):
            mapf = self.mapmax * torch.tanh(mapf / self.mapmax)  # <30
            illegalVector = self.mapmax * torch.tanh(illegalVector / self.mapmax)  # <30
        
        #bf[3] is where is last move
        #lastLocVector is a bias applied to last move location
        illegalBias=illegalmap * illegalVector.view(1,2*self.groupc,1,1)
        lb1=illegalBias[:,:self.groupc]
        lb2=illegalBias[:,self.groupc:]


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

        #g1 is sum of four line feature
        #rv is global input
        #llb1 and llb2 are last move feature
        g1sum=g1.mean(1) + \
            rv.view(rv.shape[0],rv.shape[1],1,1) + \
            lb1 
        
        h1 = self.g1lr(g1sum)  # 四线求和再leakyrelu   # avx代码里改用sum，<120


        # 以下几行是对h1进行对称的卷积。把h1卷积一次，再把h1翻转一下再卷积一次，两次的取平均，相当于对称的卷积核
        h1sym = torch.flip(h1, [2, 3])
        h1 = (h1, h1, h1, h1)
        h1sym = (h1sym, h1sym, h1sym, h1sym)
        h1c = torch.stack(self.h1conv(h1), dim=1)  # 沿着另一条线卷积
        h1csym = torch.stack(self.h1conv(h1sym), dim=1)  # 等价于卷积核反向
        h1csym = torch.flip(h1csym, [3, 4])
        h1c = (h1c + h1csym) / 2  # 正向和反向取平均，相当于对称卷积核
        h1sum = self.h1lr1(h1c, dim=2) + g2
        h2 = self.h1lr2(h1sum, dim=2)
        h3 = h2.mean(1)  # 最后把四条线整合起来
        h3 = h3 + lb2

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
        v1 = v
        v = self.value_linear2(v)
        v = torch.relu(v)
        v = self.value_linear3(v)
        v = torch.relu(v)
        v = v + v1 #res
        v = self.value_linear4(v)
        v = torch.relu(v)
        value = self.value_linearfinal(v)
        mlp_p = self.mlp_policy_w(v)

        mlp_p=self.mlp_plr(mlp_p,dim=1)
        mlp_p=torch.clip(mlp_p,-0.999,0.999)
        p=torch.einsum("nchw,nc->nhw",trunk,mlp_p).reshape((-1,1,BoardH,BoardW))
        p=p-(1-mask)*100
        p=p-illegalmap*100

        p_pass=value[:,3].reshape(-1,1)
        p=torch.cat((torch.flatten(p,start_dim=1),p_pass),dim=1)
        value=value[:,0:3]

        return value, p

class Model_v2_nog2(nn.Module):
    def __init__(self, mapb=5, mapc=256, groupc=64, mlpc=64, mlpc2=64, mapmax=30):  # 1d卷积通道，policy通道，胜负和通道
        super().__init__()
        self.model_type = "v2nog2"
        self.info = "from v2, remove 2nd group of mapping"
        self.model_param = (mapb, mapc, groupc, mlpc, mlpc2, mapmax)
        self.groupc = groupc

        self.mapmax = mapmax

        self.mapping = LineMapping(13, mapb, mapc, groupc)
        self.illegalVector = nn.Parameter(torch.empty(2, groupc)) #add a bias to all illegal locations        
        nn.init.normal_(self.illegalVector, mean=0.0, std=0.02) # 使用正态分布进行初始化

        self.gfVector=gfVector(GlobalC+BoardSizeC, groupc)
        self.g1lr = PRelu1(groupc, bound=0.999, bias=False)
        self.h1conv = Conv1dGroupLayerTupleOp(groupc, groupc, L=13, groups=groupc)
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
        self.value_linear4 = nn.Linear(mlpc, mlpc2)
         
        self.value_linearfinal = nn.Linear(mlpc2, 3 if NoPass else 4) #including pass
        self.mlp_policy_w = nn.Linear(mlpc2, groupc) #including pass
        self.mlp_plr = PRelu1(groupc, bound=10, bias=False)

    def forward(self, bf, gf, mapnoise=0):
        illegalmap=getIllegalMap(bf,gf)
        mask=None
        if UseVariousBoardsize:
            mask=bf[:,0:1,:,:]
            assert(mask.shape[1]==1)
            assert(mask.shape[2]==BoardH)
            assert(mask.shape[3]==BoardW)
        else:
            mask=None
        boardHs=mask.sum(dim=2).max(dim=2)[0].squeeze(1)
        boardWs=mask.sum(dim=3).max(dim=2)[0].squeeze(1)
        boardArea=boardHs*boardWs
        boardAreaInput=torch.stack((
            boardArea/225-1,
            torch.sqrt(boardArea/225)-1,
            (boardHs-boardWs)*(boardHs-boardWs)/boardArea
        ),dim=1)
        assert(boardAreaInput.shape[1]==BoardSizeC)
        #print(boardArea)
        gf=torch.cat((boardAreaInput,gf),dim=1)
        bw=bf[:,1:3,:,:]
        mapf = self.mapping(bw, mask)

        illegalVector=self.illegalVector
        if (self.mapmax != 0):
            mapf = self.mapmax * torch.tanh(mapf / self.mapmax)  # <30
            illegalVector = self.mapmax * torch.tanh(illegalVector / self.mapmax)  # <30
        
        #bf[3] is where is last move
        #lastLocVector is a bias applied to last move location
        illegalBias=illegalmap * illegalVector.view(1,2*self.groupc,1,1)
        lb1=illegalBias[:,:self.groupc]
        lb2=illegalBias[:,self.groupc:]


        if(mapnoise!=0):
            mapf=mapf+mapnoise*torch.randn(mapf.shape,device=mapf.device)

        mapf=mapf*mask.unsqueeze(1)

        #g1 = mapf[:, :, :self.groupc, :, :]  # 第一组通道
        #g2 = mapf[:, :, self.groupc:, :, :]  # 第二组通道
        g1 = mapf  # 第一组通道
        rv=self.gfVector(gf)
        max_rv=torch.max(torch.abs(rv)).item()
        if(max_rv>30):
            print("max_rv =",max_rv)
        #print(rv1.shape)

        #g1 is sum of four line feature
        #rv is global input
        #llb1 and llb2 are last move feature
        g1sum=g1.mean(1) + \
            rv.view(rv.shape[0],rv.shape[1],1,1) + \
            lb1 
        
        h1 = self.g1lr(g1sum)  # 四线求和再leakyrelu   # avx代码里改用sum，<120


        # 以下几行是对h1进行对称的卷积。把h1卷积一次，再把h1翻转一下再卷积一次，两次的取平均，相当于对称的卷积核
        h1sym = torch.flip(h1, [2, 3])
        h1 = (h1, h1, h1, h1)
        h1sym = (h1sym, h1sym, h1sym, h1sym)
        h1c = torch.stack(self.h1conv(h1), dim=1)  # 沿着另一条线卷积
        h1csym = torch.stack(self.h1conv(h1sym), dim=1)  # 等价于卷积核反向
        h1csym = torch.flip(h1csym, [3, 4])
        h1c = (h1c + h1csym) / 2  # 正向和反向取平均，相当于对称卷积核
        #h1sum = self.h1lr1(h1c, dim=2) + g2
        h1sum = self.h1lr1(h1c, dim=2)
        h2 = self.h1lr2(h1sum, dim=2)
        h3 = h2.mean(1)  # 最后把四条线整合起来
        h3 = h3 + lb2

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
        v1 = v
        v = self.value_linear2(v)
        v = torch.relu(v)
        v = self.value_linear3(v)
        v = torch.relu(v)
        v = v + v1 #res
        v = self.value_linear4(v)
        v = torch.relu(v)
        value = self.value_linearfinal(v)
        mlp_p = self.mlp_policy_w(v)

        mlp_p=self.mlp_plr(mlp_p,dim=1)
        mlp_p=torch.clip(mlp_p,-0.99,0.99)
        p=torch.einsum("nchw,nc->nhw",trunk,mlp_p).reshape((-1,1,BoardH,BoardW))
        p=p-(1-mask)*100
        p=p-illegalmap*100

        p_pass=value[:,3].reshape(-1,1)
        p=torch.cat((torch.flatten(p,start_dim=1),p_pass),dim=1)
        value=value[:,0:3]

        return value, p
        
class Model_v2_l11(nn.Module):
    def __init__(self, mapb=5, mapc=256, groupc=64, mlpc=64, mlpc2=64, mapmax=30):  # 1d卷积通道，policy通道，胜负和通道
        super().__init__()
        self.model_type = "v2l11"
        self.info = "v2 but mapping length=11"
        self.model_param = (mapb, mapc, groupc, mlpc, mlpc2, mapmax)
        self.groupc = groupc

        self.mapmax = mapmax

        self.mapping = LineMapping(11, mapb, mapc, 2 * groupc)
        self.illegalVector = nn.Parameter(torch.empty(2, groupc)) #add a bias to all illegal locations        
        nn.init.normal_(self.illegalVector, mean=0.0, std=0.02) # 使用正态分布进行初始化

        self.gfVector=gfVector(GlobalC+BoardSizeC, groupc)
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
        self.value_linear4 = nn.Linear(mlpc, mlpc2)
         
        self.value_linearfinal = nn.Linear(mlpc2, 3 if NoPass else 4) #including pass
        self.mlp_policy_w = nn.Linear(mlpc2, groupc) #including pass
        self.mlp_plr = PRelu1(groupc, bound=10, bias=False)

    def forward(self, bf, gf, mapnoise=0):
        illegalmap=getIllegalMap(bf,gf)
        mask=None
        if UseVariousBoardsize:
            mask=bf[:,0:1,:,:]
            assert(mask.shape[1]==1)
            assert(mask.shape[2]==BoardH)
            assert(mask.shape[3]==BoardW)
        else:
            mask=None
        boardHs=mask.sum(dim=2).max(dim=2)[0].squeeze(1)
        boardWs=mask.sum(dim=3).max(dim=2)[0].squeeze(1)
        boardArea=boardHs*boardWs
        boardAreaInput=torch.stack((
            boardArea/225-1,
            torch.sqrt(boardArea/225)-1,
            (boardHs-boardWs)*(boardHs-boardWs)/boardArea
        ),dim=1)
        assert(boardAreaInput.shape[1]==BoardSizeC)
        #print(boardArea)
        gf=torch.cat((boardAreaInput,gf),dim=1)
        bw=bf[:,1:3,:,:]
        mapf = self.mapping(bw, mask)

        illegalVector=self.illegalVector
        if (self.mapmax != 0):
            mapf = self.mapmax * torch.tanh(mapf / self.mapmax)  # <30
            illegalVector = self.mapmax * torch.tanh(illegalVector / self.mapmax)  # <30
        
        #bf[3] is where is last move
        #lastLocVector is a bias applied to last move location
        illegalBias=illegalmap * illegalVector.view(1,2*self.groupc,1,1)
        lb1=illegalBias[:,:self.groupc]
        lb2=illegalBias[:,self.groupc:]


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

        #g1 is sum of four line feature
        #rv is global input
        #llb1 and llb2 are last move feature
        g1sum=g1.mean(1) + \
            rv.view(rv.shape[0],rv.shape[1],1,1) + \
            lb1 
        
        h1 = self.g1lr(g1sum)  # 四线求和再leakyrelu   # avx代码里改用sum，<120


        # 以下几行是对h1进行对称的卷积。把h1卷积一次，再把h1翻转一下再卷积一次，两次的取平均，相当于对称的卷积核
        h1sym = torch.flip(h1, [2, 3])
        h1 = (h1, h1, h1, h1)
        h1sym = (h1sym, h1sym, h1sym, h1sym)
        h1c = torch.stack(self.h1conv(h1), dim=1)  # 沿着另一条线卷积
        h1csym = torch.stack(self.h1conv(h1sym), dim=1)  # 等价于卷积核反向
        h1csym = torch.flip(h1csym, [3, 4])
        h1c = (h1c + h1csym) / 2  # 正向和反向取平均，相当于对称卷积核
        h1sum = self.h1lr1(h1c, dim=2) + g2
        h2 = self.h1lr2(h1sum, dim=2)
        h3 = h2.mean(1)  # 最后把四条线整合起来
        h3 = h3 + lb2

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
        v1 = v
        v = self.value_linear2(v)
        v = torch.relu(v)
        v = self.value_linear3(v)
        v = torch.relu(v)
        v = v + v1 #res
        v = self.value_linear4(v)
        v = torch.relu(v)
        value = self.value_linearfinal(v)
        mlp_p = self.mlp_policy_w(v)

        mlp_p=self.mlp_plr(mlp_p,dim=1)
        mlp_p=torch.clip(mlp_p,-0.99,0.99)
        p=torch.einsum("nchw,nc->nhw",trunk,mlp_p).reshape((-1,1,BoardH,BoardW))
        p=p-(1-mask)*100
        p=p-illegalmap*100

        p_pass=value[:,3].reshape(-1,1)
        p=torch.cat((torch.flatten(p,start_dim=1),p_pass),dim=1)
        value=value[:,0:3]

        return value, p
        
class Model_v2_singlepoint(nn.Module):
    def __init__(self, mapb=5, mapc=256, groupc=64, mlpc=64, mlpc2=32, mapmax=30):  # 1d卷积通道，policy通道，胜负和通道
        super().__init__()
        self.model_type = "v2sp"
        self.info = "add a bias vector to the first stone of every move"
        self.model_param = (mapb, mapc, groupc, mlpc, mlpc2, mapmax)
        self.groupc = groupc

        self.mapmax = mapmax

        self.mapping = LineMapping(13, mapb, mapc, 2 * groupc)
        self.lastLocVector = nn.Parameter(torch.empty(2, groupc)) #add to where played last move        
        nn.init.normal_(self.lastLocVector, mean=0.0, std=0.02) # 使用正态分布进行初始化

        self.gfVector=gfVector(GlobalC+BoardSizeC, groupc)
        self.g1lr = PRelu1(groupc, bound=0.999, bias=False)
        self.h1conv = Conv1dGroupLayerTupleOp(groupc, groupc, L=13, groups=groupc)
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
        self.value_linear4 = nn.Linear(mlpc, mlpc2)
         
        self.value_linearfinal = nn.Linear(mlpc2, 3 if NoPass else 4) #including pass
        self.mlp_policy_w = nn.Linear(mlpc2, groupc) #including pass
        self.mlp_plr = PRelu1(groupc, bound=10, bias=False)

    def forward(self, bf, gf, mapnoise=0):
        illegalmap=getIllegalMap(bf,gf)
        mask=None
        if UseVariousBoardsize:
            mask=bf[:,0:1,:,:]
            assert(mask.shape[1]==1)
            assert(mask.shape[2]==BoardH)
            assert(mask.shape[3]==BoardW)
        else:
            mask=None
        boardHs=mask.sum(dim=2).max(dim=2)[0].squeeze(1)
        boardWs=mask.sum(dim=3).max(dim=2)[0].squeeze(1)
        boardArea=boardHs*boardWs
        boardAreaInput=torch.stack((
            boardArea/225-1,
            torch.sqrt(boardArea/225)-1,
            (boardHs-boardWs)*(boardHs-boardWs)/boardArea
        ),dim=1)
        assert(boardAreaInput.shape[1]==BoardSizeC)
        #print(boardArea)
        gf=torch.cat((boardAreaInput,gf),dim=1)
        bw=bf[:,1:3,:,:]
        mapf = self.mapping(bw, mask)

        lastLocVector=self.lastLocVector
        if (self.mapmax != 0):
            mapf = self.mapmax * torch.tanh(mapf / self.mapmax)  # <30
            lastLocVector = self.mapmax * torch.tanh(lastLocVector / self.mapmax)  # <30
        
        #bf[3] is where is last move
        #lastLocVector is a bias applied to last move location
        lastLocBias=bf[:,3:4] * lastLocVector.view(1,2*self.groupc,1,1)
        llb1=lastLocBias[:,:self.groupc]
        llb2=lastLocBias[:,self.groupc:]


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

        #g1 is sum of four line feature
        #rv is global input
        #llb1 and llb2 are last move feature
        g1sum=g1.mean(1) + \
            rv.view(rv.shape[0],rv.shape[1],1,1) + \
            llb1 
        
        h1 = self.g1lr(g1sum)  # 四线求和再leakyrelu   # avx代码里改用sum，<120


        # 以下几行是对h1进行对称的卷积。把h1卷积一次，再把h1翻转一下再卷积一次，两次的取平均，相当于对称的卷积核
        h1sym = torch.flip(h1, [2, 3])
        h1 = (h1, h1, h1, h1)
        h1sym = (h1sym, h1sym, h1sym, h1sym)
        h1c = torch.stack(self.h1conv(h1), dim=1)  # 沿着另一条线卷积
        h1csym = torch.stack(self.h1conv(h1sym), dim=1)  # 等价于卷积核反向
        h1csym = torch.flip(h1csym, [3, 4])
        h1c = (h1c + h1csym) / 2  # 正向和反向取平均，相当于对称卷积核
        h1sum = self.h1lr1(h1c, dim=2) + g2
        h2 = self.h1lr2(h1sum, dim=2)
        h3 = h2.mean(1)  # 最后把四条线整合起来
        h3 = h3 + llb2

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
        v1 = v
        v = self.value_linear2(v)
        v = torch.relu(v)
        v = self.value_linear3(v)
        v = torch.relu(v)
        v = v + v1 #res
        v = self.value_linear4(v)
        v = torch.relu(v)
        value = self.value_linearfinal(v)
        mlp_p = self.mlp_policy_w(v)

        mlp_p=self.mlp_plr(mlp_p,dim=1)
        mlp_p=torch.clip(mlp_p,-0.99,0.99)
        p=torch.einsum("nchw,nc->nhw",trunk,mlp_p).reshape((-1,1,BoardH,BoardW))
        p=p-(1-mask)*100
        p=p-illegalmap*100

        p_pass=value[:,3].reshape(-1,1)
        p=torch.cat((torch.flatten(p,start_dim=1),p_pass),dim=1)
        value=value[:,0:3]

        return value, p

class Model_v2_nolastmove(nn.Module):
    def __init__(self, mapb=5, mapc=256, groupc=64, mlpc=64, mlpc2=32, mapmax=30):  # 1d卷积通道，policy通道，胜负和通道
        super().__init__()
        self.model_type = "v2n"
        self.info = "No lastmove feature. Others same as v2"
        self.model_param = (mapb, mapc, groupc, mlpc, mlpc2, mapmax)
        self.groupc = groupc

        self.mapmax = mapmax

        self.mapping = LineMapping(13, mapb, mapc, 2 * groupc)

        self.gfVector=gfVector(GlobalC+BoardSizeC, groupc)
        self.g1lr = PRelu1(groupc, bound=0.999, bias=False)
        self.h1conv = Conv1dGroupLayerTupleOp(groupc, groupc, L=13, groups=groupc)
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
        self.value_linear4 = nn.Linear(mlpc, mlpc2)
         
        self.value_linearfinal = nn.Linear(mlpc2, 3 if NoPass else 4) #including pass
        self.mlp_policy_w = nn.Linear(mlpc2, groupc) #including pass
        self.mlp_plr = PRelu1(groupc, bound=10, bias=False)

    def forward(self, bf, gf, mapnoise=0):
        illegalmap=getIllegalMap(bf,gf)
        mask=None
        if UseVariousBoardsize:
            mask=bf[:,0:1,:,:]
            assert(mask.shape[1]==1)
            assert(mask.shape[2]==BoardH)
            assert(mask.shape[3]==BoardW)
        else:
            mask=None
        boardHs=mask.sum(dim=2).max(dim=2)[0].squeeze(1)
        boardWs=mask.sum(dim=3).max(dim=2)[0].squeeze(1)
        boardArea=boardHs*boardWs
        boardAreaInput=torch.stack((
            boardArea/225-1,
            torch.sqrt(boardArea/225)-1,
            (boardHs-boardWs)*(boardHs-boardWs)/boardArea
        ),dim=1)
        assert(boardAreaInput.shape[1]==BoardSizeC)
        #print(boardArea)
        gf=torch.cat((boardAreaInput,gf),dim=1)
        bw=bf[:,1:3,:,:]
        mapf = self.mapping(bw, mask)

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

        #g1 is sum of four line feature
        #rv is global input
        #llb1 and llb2 are last move feature
        g1sum=g1.mean(1) + \
            rv.view(rv.shape[0],rv.shape[1],1,1)
        
        h1 = self.g1lr(g1sum)  # 四线求和再leakyrelu   # avx代码里改用sum，<120


        # 以下几行是对h1进行对称的卷积。把h1卷积一次，再把h1翻转一下再卷积一次，两次的取平均，相当于对称的卷积核
        h1sym = torch.flip(h1, [2, 3])
        h1 = (h1, h1, h1, h1)
        h1sym = (h1sym, h1sym, h1sym, h1sym)
        h1c = torch.stack(self.h1conv(h1), dim=1)  # 沿着另一条线卷积
        h1csym = torch.stack(self.h1conv(h1sym), dim=1)  # 等价于卷积核反向
        h1csym = torch.flip(h1csym, [3, 4])
        h1c = (h1c + h1csym) / 2  # 正向和反向取平均，相当于对称卷积核
        h1sum = self.h1lr1(h1c, dim=2) + g2
        h2 = self.h1lr2(h1sum, dim=2)
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
        v1 = v
        v = self.value_linear2(v)
        v = torch.relu(v)
        v = self.value_linear3(v)
        v = torch.relu(v)
        v = v + v1 #res
        v = self.value_linear4(v)
        v = torch.relu(v)
        value = self.value_linearfinal(v)
        mlp_p = self.mlp_policy_w(v)

        mlp_p=self.mlp_plr(mlp_p,dim=1)
        mlp_p=torch.clip(mlp_p,-0.99,0.99)
        p=torch.einsum("nchw,nc->nhw",trunk,mlp_p).reshape((-1,1,BoardH,BoardW))
        p=p-(1-mask)*100
        p=p-illegalmap*100

        p_pass=value[:,3].reshape(-1,1)
        p=torch.cat((torch.flatten(p,start_dim=1),p_pass),dim=1)
        value=value[:,0:3]

        return value, p

    
  
class Model_v2_pr(nn.Module):
    def __init__(self, mapb=5, mapc=256, groupc=64, mlpc=64, mlpc2=32, mapmax=30):  # 1d卷积通道，policy通道，胜负和通道
        super().__init__()
        self.model_type = "v2pr"
        self.info = "v2lf + priority value input"
        self.model_param = (mapb, mapc, groupc, mlpc, mlpc2, mapmax)
        self.groupc = groupc

        self.mapmax = mapmax

        self.mapping = LineMapping(13, mapb, mapc, 2 * groupc)
        self.priorityVector = nn.Parameter(torch.empty(2, groupc)) #add a bias to all illegal locations   
        self.illegalVector = nn.Parameter(torch.empty(2, groupc)) #add a bias to all illegal locations        
        nn.init.normal_(self.illegalVector, mean=0.0, std=0.02) # 使用正态分布进行初始化
        
        # x and y grid, for priority value calculation
        self.H_grid = nn.Parameter(torch.arange(BoardH, dtype=torch.float32).view(1, BoardH, 1),requires_grad=False)  # shape (1, H, 1)
        self.W_grid = nn.Parameter(torch.arange(BoardW, dtype=torch.float32).view(1, 1, BoardW),requires_grad=False)  # shape (1, 1, W)

        self.gfVector=gfVector(GlobalC+BoardSizeC, groupc)
        self.g1lr = PRelu1(groupc, bound=0.999, bias=False)
        self.h1conv = Conv1dGroupLayerTupleOp(groupc, groupc, L=13, groups=groupc)
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
        self.value_linear4 = nn.Linear(mlpc, mlpc2)
         
        self.value_linearfinal = nn.Linear(mlpc2, 3 if NoPass else 4) #including pass
        self.mlp_policy_w = nn.Linear(mlpc2, groupc) #including pass
        self.mlp_plr = PRelu1(groupc, bound=10, bias=False)

    def getPriorityMap(self, bf):
        N, _, H, W = bf.shape

        if UseVariousBoardsize:
            bf=bf[:,1:]
        board=bf[:,0]+bf[:,1]-bf[:,2] #black and white stone except the new played
        # 重心计算：对H和W方向上以1为权重计算重心
        # 重心的计算是1所在位置的加权平均
        total_mass = board.sum(dim=[1, 2], keepdim=True)  # 计算每个(N,1)上的总质量（1的数量）
        
        # 计算H方向的重心
        H_centroid = (board * self.H_grid).sum(dim=[1, 2], keepdim=True) / (total_mass + 1e-6)
        # 计算W方向的重心
        W_centroid = (board * self.W_grid).sum(dim=[1, 2], keepdim=True) / (total_mass + 1e-6)
        assert(torch.all(H_centroid>-0.001))
        assert(torch.all(W_centroid>-0.001))
        assert(torch.all(H_centroid<BoardH-0.999))
        assert(torch.all(W_centroid<BoardW-0.999))
        
        # H_centroid 和 W_centroid 的形状是 (N, 1, 1)
        
        # 计算每个位置到重心的距离
        distance_H = (self.H_grid - H_centroid)  # (N, H, W) 与重心在H方向的距离
        distance_W = (self.W_grid - W_centroid)  # (N, H, W) 与重心在W方向的距离
        
        # 计算总的欧几里得距离
        distance = torch.sqrt(distance_H*distance_H + distance_W*distance_W)  # (N, H, W) 欧几里得距离

        # 如果空棋盘，返回全0张量
        mask = (total_mass < 0.01).float()  # 如果total_mass < threshold，mask会是1，否则为0
        distance = distance * (1 - mask) 
        distance *= (1/27)
        assert(27>BoardH*(2**0.5))
        assert(torch.all(distance<=1))
        return distance.unsqueeze(1)

    def forward(self, bf, gf, mapnoise=0):
        illegalmap=getIllegalMap(bf,gf)
        mask=None
        if UseVariousBoardsize:
            mask=bf[:,0:1,:,:]
            assert(mask.shape[1]==1)
            assert(mask.shape[2]==BoardH)
            assert(mask.shape[3]==BoardW)
        else:
            mask=None
        boardHs=mask.sum(dim=2).max(dim=2)[0].squeeze(1)
        boardWs=mask.sum(dim=3).max(dim=2)[0].squeeze(1)
        boardArea=boardHs*boardWs
        boardAreaInput=torch.stack((
            boardArea/225-1,
            torch.sqrt(boardArea/225)-1,
            (boardHs-boardWs)*(boardHs-boardWs)/boardArea
        ),dim=1)
        assert(boardAreaInput.shape[1]==BoardSizeC)
        #print(boardArea)
        gf=torch.cat((boardAreaInput,gf),dim=1)
        bw=bf[:,1:3,:,:]
        mapf = self.mapping(bw, mask)

        illegalVector=self.illegalVector
        priorityVector=self.priorityVector
        if (self.mapmax != 0):
            mapf = self.mapmax * torch.tanh(mapf / self.mapmax)  # <30
            illegalVector = self.mapmax * torch.tanh(illegalVector / self.mapmax)  # <30
            priorityVector = self.mapmax * torch.tanh(priorityVector / self.mapmax)  # <30

        #bf[3] is where is last move
        #lastLocVector is a bias applied to last move location
        illegalBias=illegalmap * illegalVector.view(1,2*self.groupc,1,1)
        lb1=illegalBias[:,:self.groupc]
        lb2=illegalBias[:,self.groupc:]

        priorityBias= self.getPriorityMap(bf) * priorityVector.view(1,2*self.groupc,1,1)
        pb1=priorityBias[:,:self.groupc]
        pb2=priorityBias[:,self.groupc:]


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

        #g1 is sum of four line feature
        #rv is global input
        #llb1 and llb2 are last move feature
        g1sum=g1.mean(1) + \
            rv.view(rv.shape[0],rv.shape[1],1,1) + \
            lb1 + pb1
        
        h1 = self.g1lr(g1sum)  # 四线求和再leakyrelu   # avx代码里改用sum，<120


        # 以下几行是对h1进行对称的卷积。把h1卷积一次，再把h1翻转一下再卷积一次，两次的取平均，相当于对称的卷积核
        h1sym = torch.flip(h1, [2, 3])
        h1 = (h1, h1, h1, h1)
        h1sym = (h1sym, h1sym, h1sym, h1sym)
        h1c = torch.stack(self.h1conv(h1), dim=1)  # 沿着另一条线卷积
        h1csym = torch.stack(self.h1conv(h1sym), dim=1)  # 等价于卷积核反向
        h1csym = torch.flip(h1csym, [3, 4])
        h1c = (h1c + h1csym) / 2  # 正向和反向取平均，相当于对称卷积核
        h1sum = self.h1lr1(h1c, dim=2) + g2
        h2 = self.h1lr2(h1sum, dim=2)
        h3 = h2.mean(1)  # 最后把四条线整合起来
        h3 = h3 + lb2 + pb2

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
        v1 = v
        v = self.value_linear2(v)
        v = torch.relu(v)
        v = self.value_linear3(v)
        v = torch.relu(v)
        v = v + v1 #res
        v = self.value_linear4(v)
        v = torch.relu(v)
        value = self.value_linearfinal(v)
        mlp_p = self.mlp_policy_w(v)

        mlp_p=self.mlp_plr(mlp_p,dim=1)
        mlp_p=torch.clip(mlp_p,-0.99,0.99)
        p=torch.einsum("nchw,nc->nhw",trunk,mlp_p).reshape((-1,1,BoardH,BoardW))
        p=p-(1-mask)*100
        p=p-illegalmap*100

        p_pass=value[:,3].reshape(-1,1)
        p=torch.cat((torch.flatten(p,start_dim=1),p_pass),dim=1)
        value=value[:,0:3]

        return value, p


class Model_v2_im(nn.Module):
    def __init__(self, mapb=5, mapc=256, groupc=64, mlpc=64, mlpc2=32, mapmax=30):  # 1d卷积通道，policy通道，胜负和通道
        super().__init__()
        self.model_type = "v2im"
        self.info = "v2pr + illegal feature in mapping"
        self.model_param = (mapb, mapc, groupc, mlpc, mlpc2, mapmax)
        self.groupc = groupc

        self.mapmax = mapmax

        self.mapping = LineMappingC3(13, mapb, mapc, 2 * groupc)
        self.priorityVector = nn.Parameter(torch.empty(2, groupc)) #add a bias to all illegal locations   
        self.illegalVector = nn.Parameter(torch.empty(2, groupc)) #add a bias to all illegal locations        
        nn.init.normal_(self.illegalVector, mean=0.0, std=0.02) # 使用正态分布进行初始化
        
        # x and y grid, for priority value calculation
        self.H_grid = nn.Parameter(torch.arange(BoardH, dtype=torch.float32).view(1, BoardH, 1),requires_grad=False)  # shape (1, H, 1)
        self.W_grid = nn.Parameter(torch.arange(BoardW, dtype=torch.float32).view(1, 1, BoardW),requires_grad=False)  # shape (1, 1, W)

        self.gfVector=gfVector(GlobalC+BoardSizeC, groupc)
        self.g1lr = PRelu1(groupc, bound=0.999, bias=False)
        self.h1conv = Conv1dGroupLayerTupleOp(groupc, groupc, L=13, groups=groupc)
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
        self.value_linear4 = nn.Linear(mlpc, mlpc2)
         
        self.value_linearfinal = nn.Linear(mlpc2, 3 if NoPass else 4) #including pass
        self.mlp_policy_w = nn.Linear(mlpc2, groupc) #including pass
        self.mlp_plr = PRelu1(groupc, bound=10, bias=False)

    def getPriorityMap(self, bf):
        N, _, H, W = bf.shape

        if UseVariousBoardsize:
            bf=bf[:,1:]
        board=bf[:,0]+bf[:,1]-bf[:,2] #black and white stone except the new played
        # 重心计算：对H和W方向上以1为权重计算重心
        # 重心的计算是1所在位置的加权平均
        total_mass = board.sum(dim=[1, 2], keepdim=True)  # 计算每个(N,1)上的总质量（1的数量）
        
        # 计算H方向的重心
        H_centroid = (board * self.H_grid).sum(dim=[1, 2], keepdim=True) / (total_mass + 1e-6)
        # 计算W方向的重心
        W_centroid = (board * self.W_grid).sum(dim=[1, 2], keepdim=True) / (total_mass + 1e-6)
        assert(torch.all(H_centroid>-0.001))
        assert(torch.all(W_centroid>-0.001))
        assert(torch.all(H_centroid<BoardH-0.999))
        assert(torch.all(W_centroid<BoardW-0.999))
        
        # H_centroid 和 W_centroid 的形状是 (N, 1, 1)
        
        # 计算每个位置到重心的距离
        distance_H = (self.H_grid - H_centroid)  # (N, H, W) 与重心在H方向的距离
        distance_W = (self.W_grid - W_centroid)  # (N, H, W) 与重心在W方向的距离
        
        # 计算总的欧几里得距离
        distance = torch.sqrt(distance_H*distance_H + distance_W*distance_W)  # (N, H, W) 欧几里得距离

        # 如果空棋盘，返回全0张量
        mask = (total_mass < 0.01).float()  # 如果total_mass < threshold，mask会是1，否则为0
        distance = distance * (1 - mask) 
        distance *= (1/27)
        assert(27>BoardH*(2**0.5))
        assert(torch.all(distance<=1))
        return distance.unsqueeze(1)

    def forward(self, bf, gf, mapnoise=0):
        illegalmap=getIllegalMap(bf,gf)
        mask=None
        if UseVariousBoardsize:
            mask=bf[:,0:1,:,:]
            assert(mask.shape[1]==1)
            assert(mask.shape[2]==BoardH)
            assert(mask.shape[3]==BoardW)
        else:
            mask=None
        boardHs=mask.sum(dim=2).max(dim=2)[0].squeeze(1)
        boardWs=mask.sum(dim=3).max(dim=2)[0].squeeze(1)
        boardArea=boardHs*boardWs
        boardAreaInput=torch.stack((
            boardArea/225-1,
            torch.sqrt(boardArea/225)-1,
            (boardHs-boardWs)*(boardHs-boardWs)/boardArea
        ),dim=1)
        assert(boardAreaInput.shape[1]==BoardSizeC)
        #print(boardArea)
        gf=torch.cat((boardAreaInput,gf),dim=1)
        bw=bf[:,1:3,:,:]
        mapf = self.mapping(bw, bf[:,4:5], mask)

        illegalVector=self.illegalVector
        priorityVector=self.priorityVector
        if (self.mapmax != 0):
            mapf = self.mapmax * torch.tanh(mapf / self.mapmax)  # <30
            illegalVector = self.mapmax * torch.tanh(illegalVector / self.mapmax)  # <30
            priorityVector = self.mapmax * torch.tanh(priorityVector / self.mapmax)  # <30

        #bf[3] is where is last move
        #lastLocVector is a bias applied to last move location
        illegalBias=illegalmap * illegalVector.view(1,2*self.groupc,1,1)
        lb1=illegalBias[:,:self.groupc]
        lb2=illegalBias[:,self.groupc:]

        priorityBias= self.getPriorityMap(bf) * priorityVector.view(1,2*self.groupc,1,1)
        pb1=priorityBias[:,:self.groupc]
        pb2=priorityBias[:,self.groupc:]


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

        #g1 is sum of four line feature
        #rv is global input
        #llb1 and llb2 are last move feature
        g1sum=g1.mean(1) + \
            rv.view(rv.shape[0],rv.shape[1],1,1) + \
            lb1 + pb1
        
        h1 = self.g1lr(g1sum)  # 四线求和再leakyrelu   # avx代码里改用sum，<120


        # 以下几行是对h1进行对称的卷积。把h1卷积一次，再把h1翻转一下再卷积一次，两次的取平均，相当于对称的卷积核
        h1sym = torch.flip(h1, [2, 3])
        h1 = (h1, h1, h1, h1)
        h1sym = (h1sym, h1sym, h1sym, h1sym)
        h1c = torch.stack(self.h1conv(h1), dim=1)  # 沿着另一条线卷积
        h1csym = torch.stack(self.h1conv(h1sym), dim=1)  # 等价于卷积核反向
        h1csym = torch.flip(h1csym, [3, 4])
        h1c = (h1c + h1csym) / 2  # 正向和反向取平均，相当于对称卷积核
        h1sum = self.h1lr1(h1c, dim=2) + g2
        h2 = self.h1lr2(h1sum, dim=2)
        h3 = h2.mean(1)  # 最后把四条线整合起来
        h3 = h3 + lb2 + pb2

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
        v1 = v
        v = self.value_linear2(v)
        v = torch.relu(v)
        v = self.value_linear3(v)
        v = torch.relu(v)
        v = v + v1 #res
        v = self.value_linear4(v)
        v = torch.relu(v)
        value = self.value_linearfinal(v)
        mlp_p = self.mlp_policy_w(v)

        mlp_p=self.mlp_plr(mlp_p,dim=1)
        mlp_p=torch.clip(mlp_p,-0.99,0.99)
        p=torch.einsum("nchw,nc->nhw",trunk,mlp_p).reshape((-1,1,BoardH,BoardW))
        p=p-(1-mask)*100

        p=p-illegalmap*100

        p_pass=value[:,3].reshape(-1,1)
        p=torch.cat((torch.flatten(p,start_dim=1),p_pass),dim=1)
        value=value[:,0:3]

        return value, p


ModelDic = {
    "res": Model_ResNet, #resnet对照组
    "v1": Model_v1,
    "v2": Model_v2,
    "v2nog2": Model_v2_nog2,
    "v2lf": Model_v2,
    "v2l11": Model_v2_l11,
    "v2sp": Model_v2_singlepoint,
    "v2pr": Model_v2_pr,
    "v2im": Model_v2_im,
    "v2n": Model_v2_nolastmove,
}
