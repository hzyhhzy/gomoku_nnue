
from dataset import trainset
from model import ModelDic
from model import boardH,boardW

import argparse
import glob
import sys
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.functional as F
import torch.nn as nn
import torch
import os
import time
import shutil

try:
    os.mkdir("export")
except:
    pass
else:
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int,
                        default=0, help='which gpu')
    parser.add_argument('--cpu', action='store_true', default=False, help='whether use cpu')
    parser.add_argument('--model', type=str ,default='test', help='model path')
    parser.add_argument('--export', type=str ,default='', help='export path')
    parser.add_argument('--copy', action='store_true', default=False, help='copy a backup for this model, for selfplay training')
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    if(args.cpu):
        device=torch.device('cpu')
    modelname=args.model
    exportname=args.export
    if(exportname==''):
        exportname=modelname






    file_path = f'saved_models/{modelname}.pth'
    model_type=None
    if os.path.exists(file_path):
        data = torch.load(file_path)
        model_type=data['model_name']
        model_param=data['model_size']
        model = ModelDic[model_type](*model_param).to(device)

        model.load_state_dict(data['state_dict'])
        totalstep = data['step']
        print(f"loaded model: type={data['model_name']}, size={model.model_size}, totalstep={totalstep}")
    else:
        print("Invalid Model Path")
        exit(0)


    #copy model file
    if(args.copy):
        modeldir='export/'+modelname
        try:
            os.mkdir(modeldir)
        except:
            pass
        else:
            pass
        modelDestPath=modeldir+'/'+str(totalstep)+'.pth'
        shutil.copy(file_path,modelDestPath)


    model.eval()






    time0=time.time()
    print("Start")
    exportPath='export/'+exportname+'.txt'
    exportfile=open(exportPath,'w')

    pc=model.pc
    vc=model.vc


    scale_now=1

#export featuremap
#-------------------------------------------------------------
    print("Exporting FeatureMap")
    print("featuremap",file=exportfile)
    #prepare data
    def fullData(length):  # 三进制全部排列
        x = np.zeros((1, length), dtype=np.int64)
        for i in range(length):
            x1 = x.copy()
            x1[:, i] = 1
            x2 = x.copy()
            x2[:, i] = 2
            x = np.concatenate((x, x1, x2), axis=0)
        return x

    buf=np.ones((4*(3**11),pc+vc),dtype=np.float64)*114514

    pow3 = np.array([1, 3, 9, 27, 81, 243, 729, 2187, 6561,3**9,3**10,3**11], dtype=np.int64)
    pow3=pow3[:,np.newaxis]
    #无边界和单边边界 长度9

    data=fullData(11)
    label=model.exportMapTable(data,device)

    #无边界与正边界
    for r in range(6):
        ids=np.matmul(data[:,r:],pow3[:11-r])+pow3[12-r:].sum()
        for i in range(ids.shape[0]):
            buf[ids[i]]=label[:,i,5+r]

    #负边界
    for l in range(1,6):
        ids=np.matmul(data[:,:11-l],pow3[l:-1])+2*3**11+pow3[:l-1].sum()
        for i in range(ids.shape[0]):
            buf[ids[i]]=label[:,i,5-l]

    #双边边界
    for left in range(1,6):
        for right in range(1,6):
            L=11-left-right
            data = fullData(L)
            label=model.exportMapTable(data,device)
            idbias=3*3**11+pow3[0:left-1].sum((0,1))+pow3[12-right:-1].sum((0,1))
            ids=np.matmul(data,pow3[left:11-right])+idbias
            for i in range(ids.shape[0]):
                buf[ids[i]]=label[:,i,5-left]

    bound=1#这个变量存储上界，时刻注意int16溢出
    useful=(buf[:,0]!=114514)
    buf=buf*(useful[:,np.newaxis].astype(np.float64))
    usefulcount=useful.sum()
    print("Useful=",usefulcount)
    wmax=np.abs(buf).max()
    bound*=wmax
    print("Max=",wmax)
    print(usefulcount,file=exportfile)




    #np.set_printoptions(suppress=True,precision=3)
    w_scale=6000/wmax
    scale_now*=w_scale #200
    maxint=wmax*w_scale
    if(maxint>32700):
        print("Error! Maxint=",maxint)
        exit(0)
    for i in range(buf.shape[0]):
        if useful[i]:
            print(i,end=' ',file=exportfile)
            for j in range(pc+vc):
                print(int(buf[i,j]*w_scale),end=' ',file=exportfile)
            print('',file=exportfile)
    bound*=w_scale
    print("Bound=",bound)


# export others
    print("Finished featuremap, now exploring others")
# -------------------------------------------------------------
#1 map_leakyrelu
    #prepare
    scale_maplr=scale_now #prepare for value head


    map_lr_slope=model.map_leakyrelu.slope.data.cpu().numpy()
    map_lr_slope=np.tanh(map_lr_slope / 6) * 6
    map_lr_bias=model.map_leakyrelu.bias.data.cpu().numpy()

    #calculate max
    wmax=np.abs(map_lr_slope).max()
    bmax=np.abs(map_lr_bias).max()
    bound_c=bound*(np.abs(map_lr_slope)+1)+np.abs(map_lr_bias)*scale_now
    print("map lr maxslope=",wmax)
    print("map lr maxbias=",bmax)
    map_lr_slope_sub1=(map_lr_slope-1)*0.125  #slope>1会溢出，所以负半轴乘slope*0.125再乘2，正半轴乘0.25
    map_lr_bias=map_lr_bias*scale_now

    maxint=max((wmax+1)*2**15/8,bmax*scale_now) #mulhrs右移15位
    if(maxint>32700):
        print("Error! Maxint=",maxint)
        exit(0)

    #write
    print("map_lr_slope_sub1div8",file=exportfile)
    for i in range(pc+vc):
        print(int(map_lr_slope_sub1[i]*2**15),end=' ',file=exportfile) #mulhrs右移15位
    print('',file=exportfile)

    print("map_lr_bias",file=exportfile)
    for i in range(pc+vc):
        print(int(map_lr_bias[i]),end=' ',file=exportfile)
    print('',file=exportfile)

    #bound

    bound=bound*(wmax+1)+bmax*scale_now
    print("Bound=",bound)

# 2 policyConv


    policyConvWeight=model.policy_conv.weight.data.cpu().numpy()
    policyConvBias=model.policy_conv.bias.data.cpu().numpy()

    #calculate max
    wmax=np.abs(policyConvWeight).max()
    bound_c=(np.abs(policyConvWeight).sum((1,2,3))*bound_c[:pc]+np.abs(policyConvBias)*scale_now)
    bmax=np.abs(policyConvBias).max()
    print("policyConvWeight max=",wmax)
    print("policyConvBias max=",bmax)

    maxint=max(wmax*2**15,bmax*scale_now)

    w_scale=min(32700/maxint,32700/bound_c.max())
    print("policy conv w_scale=",w_scale)
    scale_now*=w_scale
    bound_c*=w_scale


    policyConvWeight=policyConvWeight*w_scale
    policyConvBias=policyConvBias*scale_now

    maxint=max(wmax*w_scale*2**15,bmax*scale_now)
    if(maxint>32750):
        print("Error! Maxint=",maxint)
        exit(0)

    #write
    print("policyConvWeight",file=exportfile)
    for i in range(9):
        for j in range(pc):
            print(int(policyConvWeight[j,0,i//3,i%3]*2**15),end=' ',file=exportfile)
    print('',file=exportfile)

    print("policyConvBias",file=exportfile)
    for i in range(pc):
        print(int(policyConvBias[i]),end=' ',file=exportfile)
    print('',file=exportfile)

    #bound
    print("Boundc=",bound_c.max())
    print("if boundc < 32767, that's ok")

# 2 policyFinalConv
    #prepare
    policyFinalConv=model.policy_linear.weight.data.cpu().numpy()

    #calculate max
    wmax=np.abs(policyFinalConv).max()
    print("policyFinalConv max=",wmax)

    maxint=wmax*2**15
    w_scale=min(32700/maxint,0.5)
    scale_now*=w_scale
    if(maxint*w_scale>32750):
        print("Error! Maxint=",maxint)
        exit(0)

    #write
    policyFinalConv=policyFinalConv*w_scale
    print("policyFinalConv",file=exportfile)
    for i in range(pc):
        print(int(policyFinalConv[0,i,0,0]*2**15),end=' ',file=exportfile)
    print('',file=exportfile)

    #bound
    bound=(np.abs(policyFinalConv)[0,:,0,0]*bound_c).sum()
    print("Bound=",bound)
    print("If this bound is a little bigger than 32767, there's no big problem")



#剩下的都是float 无需担心量化

#policy final leakyrelu
    p_slope=model.policy_leakyrelu.slope.data.cpu().numpy()[0]
    print("policy_neg_slope",file=exportfile)
    print(p_slope/scale_now,file=exportfile)
    print("policy_pos_slope",file=exportfile)
    print(1/scale_now,file=exportfile)

#scale_beforemlp
    print("scale_beforemlp",file=exportfile)
    print(1/scale_maplr/boardH/boardW,file=exportfile)

# 从这里开始 scale就是1了
#value first leakyrelu
    value_lr_slope=model.value_leakyrelu.slope.data.cpu().numpy()
    value_lr_slope_sub1=value_lr_slope-1

    print("value_lr_slope_sub1",file=exportfile)
    for i in range(vc):
        print(value_lr_slope_sub1[i],end=' ',file=exportfile)
    print('',file=exportfile)



#mlp layer 1
    #scale_layer1=1/scale_maplr/boardH/boardW
    w=model.value_linear1.weight.data.cpu().numpy()
    #w=w*scale_layer1
    b=model.value_linear1.bias.data.cpu().numpy()

    print("mlp_w1",file=exportfile)
    for i in range(vc):
        for j in range(vc):
            print(w[j][i],end=' ',file=exportfile)
    print('',file=exportfile)

    print("mlp_b1",file=exportfile)
    for i in range(vc):
        print(b[i],end=' ',file=exportfile)
    print('',file=exportfile)

# mlp layer 2
    w = model.value_linear2.weight.data.cpu().numpy()
    b = model.value_linear2.bias.data.cpu().numpy()

    print("mlp_w2", file=exportfile)
    for i in range(vc):
        for j in range(vc):
            print(w[j][i], end=' ', file=exportfile)
    print('', file=exportfile)

    print("mlp_b2", file=exportfile)
    for i in range(vc):
        print(b[i], end=' ', file=exportfile)
    print('', file=exportfile)

# mlp layer 3
    w = model.value_linearfinal.weight.data.cpu().numpy()
    b = model.value_linearfinal.bias.data.cpu().numpy()

    print("mlp_w3", file=exportfile)
    for i in range(vc):
        for j in range(3):
            print(w[j][i], end=' ', file=exportfile)
    print('', file=exportfile)

    print("mlp_b3", file=exportfile)
    for i in range(3):
        print(b[i], end=' ', file=exportfile)
    print('', file=exportfile)









    exportfile.close()



    #copy txt file
    if(args.copy):
        #modeldir='export/'+modelname
        exportCopyDestPath=modeldir+'/'+str(totalstep)+'.txt'
        shutil.copy(exportPath,exportCopyDestPath)


    print("success")






