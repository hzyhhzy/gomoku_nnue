
from model import Model_v3
from model import boardH,boardW

import argparse
import numpy as np
import torch
import os
import time
import shutil

try:
    os.mkdir("../export")
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
    parser.add_argument('--compat', action='store_true', default=False, help='whether disable global features. if true, use default rules and compat old engines')
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    if(args.cpu):
        device=torch.device('cpu')
    modelname=args.model
    exportname=args.export
    if(exportname==''):
        exportname=modelname






    file_path = f'../saved_models/{modelname}/model.pth'
    model_type=None
    if os.path.exists(file_path):
        data = torch.load(file_path, map_location=device)
        model_type = data['model_type']
        model_param = data['model_param']
        if(model_type != "v3"):
            print(f"Invalid Model Type: {model_type}")
            exit(0)
        model = Model_v3(*model_param).to(device)

        model.load_state_dict(data['state_dict'])
        totalstep = data['totalstep']
        print(f"loaded model: type={model_type}, param={model.model_param}, totalstep={totalstep}")
    else:
        print(f"Invalid Model Path: {file_path}")
        exit(0)


    #copy model file
    if(args.copy):
        modeldir='../export/'+modelname
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
    _, _, groupc, mlpc, _ = model_param
    print(f"Start: groupc={groupc}, mlpc={mlpc}")
    exportPath='../export/'+exportname
    if(args.compat):
        exportPath+="_compat"
    exportPath+='.txt'
    exportfile=open(exportPath,'w')

# file head
# -------------------------------------------------------------
    print("v3" if not args.compat else "v3compat",file=exportfile)
    print(groupc,mlpc,end=' ',file=exportfile)
    print('',file=exportfile)
#export featuremap
#-------------------------------------------------------------
    print("Exporting mapping")
    print("mapping",file=exportfile)
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

    buf=np.ones((4*(3**11),groupc*2),dtype=np.float64)*114514

    pow3 = np.array([1, 3, 9, 27, 81, 243, 729, 2187, 6561,3**9,3**10,3**11], dtype=np.int64)
    pow3=pow3[:,np.newaxis]
    #无边界和单边边界 长度11

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

    useful=(buf[:,0]!=114514)
    buf=buf*(useful[:,np.newaxis].astype(np.float64))
    usefulcount=useful.sum()
    print("Useful=",usefulcount)

    scale_now=1 #这个变量存储int16和float的换算比
    bound=np.abs(buf).max()#这个变量存储上界，时刻注意int16溢出
    bound+=30 # h1 = g1.mean(1)+gfVector(gf),  |gfVector(gf)|<30
    print("Max=",bound)
    print(usefulcount,file=exportfile)


    if args.compat:
        gfVector_bias=model.exportDefaultGFVector(device=device)
    else: #calculate gfVector in engine, export later, not here
        gfVector_bias=np.zeros((groupc,))



    map_maxint=8000
    g2extrascale=0.8
    trunkconv1_extrabound=10 #允许少量超越bound
    trunkconv2_extrabound=30

    w_scale=map_maxint/bound #w_scale表示这一步的倍数

    scale_now*=w_scale
    bound*=w_scale
    g2scale=scale_now*g2extrascale
    g2bound=bound*g2scale/scale_now
    maxint=bound #maxint表示导出权重的最大值,此处恰好为bound
    if(maxint>32760):
        print("Error! Maxint=",maxint)
        exit(0)
    for i in range(buf.shape[0]):
        if useful[i]:
            print(i,end=' ',file=exportfile)
            for j in range(groupc):
                print(int((buf[i,j]+gfVector_bias[j])*w_scale),end=' ',file=exportfile)
            for j in range(groupc):
                print(int(buf[i,j+groupc]*g2scale),end=' ',file=exportfile)
            print('',file=exportfile)
    print("Bound after mapping = ",bound)
    print("Scale after mapping = ", scale_now)
    if args.compat:
        print("4*gfVector_bias =",(4*gfVector_bias*w_scale).astype(np.int64))


# export others
    print("Finished mapping, now exporting others")
# -------------------------------------------------------------
#0  gfvector
    if not args.compat:  # export gfVector mlp
        #assert (False, "Todo: implement for exporting gfVector mlp")

        # gfVector mlp layer 1
        w = model.gfVector.layer1.weight.data.cpu().numpy()
        b = model.gfVector.layer1.bias.data.cpu().numpy()

        # pytorch里面是4线平均，c++里是4线求和，所以需要额外乘4
        w=w*scale_now*4
        b=b*scale_now*4

        #gfscale=0.3  #sometimes gf may larger than 1, so let gf multiply gfscale then convert to int16, then multiply gfweight
        #w=w/gfscale
        #gfscale*=2**15 #mulhrs右移15位

        maxint=np.abs(w).max()
        maxint=max(maxint,np.abs(b).max())
        print(f"gf max = {maxint}")


        #print("gfscale", file=exportfile)
        #print(int(gfscale), end=' ', file=exportfile)
        #print('', file=exportfile)

        print("gfvector_w", file=exportfile)
        for i in range(w.shape[1]):
            for j in range(groupc):
                print(int(w[j][i]), end=' ', file=exportfile)
        print('', file=exportfile)

        print("gfvector_b", file=exportfile)
        for i in range(groupc):
            print(int(b[i]), end=' ', file=exportfile)
        print('', file=exportfile)




#1 g1lr
    #pytorch里面是4线平均，c++里是4线求和
    #g1lr本身对scale和bound无影响，w_scale=1
    scale_now*=4
    bound*=4

    g1lr_w=model.g1lr.export_slope()

    maxint=np.abs(g1lr_w*2**15).max() #mulhrs右移15位
    if(maxint>32760):
        print("Error! Maxint=",maxint)
        exit(0)

    #write
    print("g1lr_w",file=exportfile)
    for i in range(groupc):
        print(int(g1lr_w[i]*2**15),end=' ',file=exportfile) #mulhrs右移15位
    print('',file=exportfile)

    print("Bound after g1lr = ",bound)
    print("Scale after g1lr = ", scale_now)

#2 h1conv

    h1conv_w=model.h1conv.w.data.cpu().numpy()
    h1conv_w=np.squeeze(h1conv_w,2)
    h1conv_w=(h1conv_w+np.flip(h1conv_w,0))/2 #symmetry
    assert(h1conv_w.shape==(11,groupc))
    h1conv_b=model.h1conv.b.data.cpu().numpy()

    h1conv_b*=scale_now
    bound=(np.abs(h1conv_w).sum(0)*bound+np.abs(h1conv_b)).max()

    w_scale=g2scale/scale_now
    scale_now*=w_scale
    bound*=w_scale
    h1conv_w*=w_scale
    h1conv_b*=w_scale

    maxint=max(np.abs(h1conv_w).max()*2**15,np.abs(h1conv_b).max())
    if(maxint>32760):
        print("Error! Maxint=",maxint)
        exit(0)

    #write
    print("h1conv_w",file=exportfile)
    for i in range(6):
        for j in range(groupc):
            print(int(h1conv_w[i+5,j]*2**15),end=' ',file=exportfile)
    print('',file=exportfile)

    print("h1conv_b",file=exportfile)
    for i in range(groupc):
        print(int(h1conv_b[i]),end=' ',file=exportfile)
    print('',file=exportfile)

    print("Bound after h1conv = ",bound)
    print("Scale after h1conv = ", scale_now)

#3 h1lr1
    # h1lr1本身对scale和bound无影响，w_scale=1
    h1lr1_w = model.h1lr1.export_slope()

    maxint = np.abs(h1lr1_w * 2 ** 15).max()  # mulhrs右移15位
    if (maxint > 32760):
        print("Error! Maxint=", maxint)
        exit(0)

    # write
    print("h1lr1_w", file=exportfile)
    for i in range(groupc):
        print(int(h1lr1_w[i] * 2 ** 15), end=' ', file=exportfile)  # mulhrs右移15位
    print('', file=exportfile)

    print("Bound after h1lr1 = ", bound)
    print("Scale after h1lr1 = ", scale_now)

    #+g2
    assert(scale_now==g2scale)
    bound+=g2bound

#4 h1lr2
    # h1lr2本身对scale和bound无影响，w_scale=1
    h1lr2_w = model.h1lr2.export_slope()

    maxint = np.abs(h1lr2_w * 2 ** 15).max()  # mulhrs右移15位
    if (maxint > 32760):
        print("Error! Maxint=", maxint)
        exit(0)

    # write
    print("h1lr2_w", file=exportfile)
    for i in range(groupc):
        print(int(h1lr2_w[i] * 2 ** 15), end=' ', file=exportfile)  # mulhrs右移15位
    print('', file=exportfile)

    print("Bound after h1lr2 = ", bound)
    print("Scale after h1lr2 = ", scale_now)

    # 对于h3，pytorch里面是4线平均，c++里也是4线平均，所以保持不变
    scale_now *= 1
    bound *= 1

#5 h3lr
    # 本身对scale无影响，w_scale=1
    h3lr_w = model.h3lr.export_slope()
    h3lr_b = model.h3lr.bias.data.cpu().numpy()
    h3lr_b*=scale_now

    bmax=np.abs(h3lr_b).max()
    bound+=bmax
    maxint = max(np.abs(h3lr_w * 2 ** 15).max(),bmax)  # mulhrs右移15位
    if (maxint > 32760):
        print("Error! Maxint=", maxint)
        exit(0)

    # write
    print("h3lr_w", file=exportfile)
    for i in range(groupc):
        print(int(h3lr_w[i] * 2 ** 15), end=' ', file=exportfile)  # mulhrs右移15位
    print('', file=exportfile)
    print("h3lr_b", file=exportfile)
    for i in range(groupc):
        print(int(h3lr_b[i]), end=' ', file=exportfile)  # mulhrs右移15位
    print('', file=exportfile)

    print("Bound after h3lr = ", bound)
    print("Scale after h3lr = ", scale_now)

    # 对于h3，pytorch里面是4线平均，c++里也是4线平均，所以保持不变
    scale_now *= 1
    bound *= 1

#6 trunkconv1

    trunkconv1_w = model.trunkconv1.weight.data.cpu().numpy()
    trunkconv1_w = np.squeeze(trunkconv1_w, (2,3))
    assert (trunkconv1_w.shape == (groupc, 4))
    trunkconv1_b = model.trunkconv1.bias.data.cpu().numpy()

    trunkconv1_b *= scale_now
    bound = (np.abs(trunkconv1_w).sum(1) * bound + np.abs(trunkconv1_b)).max()

    maxint = max(np.abs(trunkconv1_w).max() * 2 ** 15, np.abs(trunkconv1_b).max())
    w_scale = min(32700*trunkconv1_extrabound/bound,32700/maxint,200/scale_now)
    print("trunkconv1_w_scale = ",w_scale)
    scale_now *= w_scale
    bound *= w_scale
    trunkconv1_w *= w_scale
    trunkconv1_b *= w_scale

    maxint = max(np.abs(trunkconv1_w).max() * 2 ** 15, np.abs(trunkconv1_b).max())
    if (maxint > 32760):
        print("Error! Maxint=", maxint)
        exit(0)

    # c++代码和pytorch代码的分组顺序不一样
    trunkconv1_w=trunkconv1_w.reshape((-1,4,4,4))
    trunkconv1_w=trunkconv1_w.transpose(3,0,2,1)
    trunkconv1_w=trunkconv1_w.reshape((4,-1))
    trunkconv1_b=trunkconv1_b.reshape((-1,4,4))
    trunkconv1_b=trunkconv1_b.transpose(0,2,1)
    trunkconv1_b=trunkconv1_b.reshape((-1,))

    # write
    print("trunkconv1_w", file=exportfile)
    for i in range(4):
        for j in range(groupc):
            print(int(trunkconv1_w[i , j] * 2 ** 15), end=' ', file=exportfile)
    print('', file=exportfile)

    print("trunkconv1_b", file=exportfile)
    for i in range(groupc):
        print(int(trunkconv1_b[i]), end=' ', file=exportfile)
    print('', file=exportfile)

    print("Bound after trunkconv1 = ", bound)
    print("Scale after trunkconv1 = ", scale_now)

#7 trunklr1
    # 本身对scale无影响，w_scale=1
    trunklr1_w = model.trunklr1.export_slope()

    maxint = np.abs(trunklr1_w * 2 ** 15).max()  # mulhrs右移15位
    if (maxint > 32760):
        print("Error! Maxint=", maxint)
        exit(0)

    # write
    print("trunklr1_w", file=exportfile)
    for i in range(groupc):
        print(int(trunklr1_w[i] * 2 ** 15), end=' ', file=exportfile)  # mulhrs右移15位
    print('', file=exportfile)

    print("Bound after trunklr1 = ", bound)
    print("Scale after trunklr1 = ", scale_now)
    print("实验表明大于100的数字的出现频率小于1/100000，因此scale小于300即可")

#8 trunkconv2 trunklr2

    trunkconv2_w = model.trunkconv2.w.data.cpu().numpy()
    trunkconv2_w = np.squeeze(trunkconv2_w, 2)
    assert (trunkconv2_w.shape == (3, groupc))

    trunklr2_b = model.trunklr2.bias.data.cpu().numpy()
    trunklr2_b*=scale_now

    trunklr2_w = model.trunklr2.export_slope()


    bound = (np.maximum(np.abs(trunklr2_b),np.abs(trunklr2_b))+bound*(np.abs(trunkconv2_w[0])+4*np.abs(trunkconv2_w[1])+4*np.abs(trunkconv2_w[2]))).max()

    maxint = np.abs(trunkconv2_w).max() * 2 ** 15
    w_scale = min(32700*trunkconv2_extrabound / bound ,32700/maxint,200/scale_now)
    print("trunkconv2_w_scale = ", w_scale)
    scale_now *= w_scale
    bound *= w_scale
    trunkconv2_w *= w_scale
    trunklr2_b *= w_scale

    maxint = np.abs(trunkconv2_w).max() * 2 ** 15
    if (maxint > 32760):
        print("Error! Maxint=", maxint)
        exit(0)


    # write
    print("trunkconv2_w", file=exportfile)
    for i in range(3):
        for j in range(groupc):
            print(int(trunkconv2_w[i, j] * 2 ** 15), end=' ', file=exportfile)
    print('', file=exportfile)
    print("trunklr2_w", file=exportfile)
    for i in range(groupc):
        print(int(trunklr2_w[i] * 2 ** 15), end=' ', file=exportfile)
    print('', file=exportfile)
    print("trunklr2_b", file=exportfile)
    for i in range(groupc):
        print(int(trunklr2_b[i]), end=' ', file=exportfile)
    print('', file=exportfile)




    print("Bound after trunkconv2+trunklr2 = ", bound)
    print("Scale after trunkconv2+trunklr2 = ", scale_now)
    print("实验表明大于100的数字的出现频率小于1/100000，因此scale小于300即可")

#10 mlp scale
    print("scale_beforemlpInv", file=exportfile)
    print(1/scale_now, end=' ', file=exportfile)
    print('', file=exportfile)

#从这里往下的都是float 无需担心量化，且scale=1

    # valuelr
    valuelr_w = model.valuelr.export_slope()
    valuelr_b = model.valuelr.bias.data.cpu().numpy()

    print("valuelr_w", file=exportfile)
    for i in range(groupc):
        print(valuelr_w[i], end=' ', file=exportfile)  # mulhrs右移15位
    print('', file=exportfile)
    print("valuelr_b", file=exportfile)
    for i in range(groupc):
        print(valuelr_b[i], end=' ', file=exportfile)  # mulhrs右移15位
    print('', file=exportfile)


    #mlp layer 1
    w=model.value_linear1.weight.data.cpu().numpy()
    b=model.value_linear1.bias.data.cpu().numpy()

    print("mlp_w1",file=exportfile)
    for i in range(groupc):
        for j in range(mlpc):
            print(w[j][i],end=' ',file=exportfile)
    print('',file=exportfile)

    print("mlp_b1",file=exportfile)
    for i in range(mlpc):
        print(b[i],end=' ',file=exportfile)
    print('',file=exportfile)

    #mlp layer 2
    w=model.value_linear2.weight.data.cpu().numpy()
    b=model.value_linear2.bias.data.cpu().numpy()

    print("mlp_w2",file=exportfile)
    for i in range(mlpc):
        for j in range(mlpc):
            print(w[j][i],end=' ',file=exportfile)
    print('',file=exportfile)

    print("mlp_b2",file=exportfile)
    for i in range(mlpc):
        print(b[i],end=' ',file=exportfile)
    print('',file=exportfile)

    #mlp layer 3
    w=model.value_linear3.weight.data.cpu().numpy()
    b=model.value_linear3.bias.data.cpu().numpy()

    print("mlp_w3",file=exportfile)
    for i in range(mlpc):
        for j in range(mlpc):
            print(w[j][i],end=' ',file=exportfile)
    print('',file=exportfile)

    print("mlp_b3",file=exportfile)
    for i in range(mlpc):
        print(b[i],end=' ',file=exportfile)
    print('',file=exportfile)

    #mlpfinal
    w=model.value_linearfinal.weight.data.cpu().numpy()
    b=model.value_linearfinal.bias.data.cpu().numpy()

    mlp_outc=3 if args.compat else 4

    print("mlpfinal_w",file=exportfile)
    for i in range(mlpc):
        for j in range(mlp_outc):
            print(w[j][i],end=' ',file=exportfile)
    print('',file=exportfile)

    print("mlpfinal_b",file=exportfile)
    for i in range(mlp_outc):
        print(b[i],end=' ',file=exportfile)
    print('',file=exportfile)


    #mlp_p layer
    w=2**15*model.mlp_policy_w.weight.data.cpu().numpy() #提前乘32768
    b=2**15*model.mlp_policy_w.bias.data.cpu().numpy()

    print("mlp_p_w",file=exportfile)
    for i in range(mlpc):
        for j in range(groupc):
            print(w[j][i],end=' ',file=exportfile)
    print('',file=exportfile)

    print("mlp_p_b",file=exportfile)
    for i in range(groupc):
        print(b[i],end=' ',file=exportfile)
    print('',file=exportfile)


    # mlp_plr
    mlp_plr_w = model.mlp_plr.export_slope()

    print("mlp_plr_w", file=exportfile)
    for i in range(groupc):
        print(mlp_plr_w[i], end=' ', file=exportfile)  # mulhrs右移15位
    print('', file=exportfile)

    #then convert to int16 and clipped to 32768*0.99
    #now scale is scale_now


    exportfile.close()



    #copy txt file
    if(args.copy):
        exportCopyDestPath=modeldir+'/'+str(totalstep)+'.txt'
        shutil.copy(exportPath,exportCopyDestPath)


    print("success")






