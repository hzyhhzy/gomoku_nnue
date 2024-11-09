
from dataset import trainset
from model import ModelDic
from config import *

import argparse
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch
import os
import time
import random
import copy

backup_checkpoints=[50000*i for i in range(5000)]


def lossSoftFunction(losses):
    #losses=torch.pow(losses+0.25,0.5)-0.5
    return torch.mean(losses,dim=0)


def cross_entropy_loss(output, target):
    t = torch.log_softmax(output,dim=1)
    losses = torch.sum(-t*target, dim=1)+torch.sum(torch.log(target+1e-10)*target, dim=1)
    return lossSoftFunction(losses)


def calculatePolicyLoss(output,pt):
    output=torch.flatten(output,start_dim=1)
    if(output.shape[1]==BoardW * BoardH): #the model does not support pass
        pt = pt[:, 0:BoardW * BoardH]
    pt = pt+1e-10
    wsum = torch.sum(pt, dim=1, keepdims=True)
    pt = pt/wsum

    return cross_entropy_loss(output,pt)

def save_checkpoint(model, optimizer, path):
    savedic={'totalstep': totalstep,
            'state_dict': model.state_dict(),
            'model_type': model.model_type,
            'model_param':model.model_param}
    if(optimizer is not None):
        savedic['optimizer_state_dict']=optimizer.state_dict()

    torch.save(
        savedic,
        path)
    print('Model saved in {}\n'.format(path))

if __name__ == '__main__':
    #把工作目录设为此文件所在的目录
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    os.chdir(script_dir)
    print(f"当前工作目录已更改为: {os.getcwd()}")


    parser = argparse.ArgumentParser()

    #data settings
    parser.add_argument('--tdatadir', type=str, default='../data/tdata_filtered', help='train dataset path: dir include dataset files or single dataset file')
    parser.add_argument('--vdatadir', type=str, default='../data/vdata_filtered/part_0.npz', help='validation dataset path: dir include dataset files or single dataset file')
    parser.add_argument('--maxvalsamp', type=int, default=20000, help='validation sample num')
    parser.add_argument('--maxstep', type=int, default=5000000000, help='max step to train')
    parser.add_argument('--savestep', type=int, default=2000, help='step to save and validation')
    parser.add_argument('--infostep', type=int, default=500, help='step to logger')

    parser.add_argument('--sampling', type=float, default=1, help='sampling rate(to avoid overfitting)')
    parser.add_argument('--valuesampling', type=float, default=1, help='value sampling rate(to avoid overfitting)')

    #model parameters
    parser.add_argument('--modeltype', type=str, default='res',help='model type defined in model.py')
    parser.add_argument('--modelparam', nargs='+',type=int,
                        default=(10,128), help='model size')

    parser.add_argument('--savename', type=str ,default='null', help='model save pth, ""null"" means does not save, ""auto"" means modeltype+modelparam')
    parser.add_argument('--new', action='store_true', default=False, help='whether to retrain')

    #training parameters
    parser.add_argument('--gpu', type=int,
                        default=0, help='which gpu, -1 means cpu')
    parser.add_argument('--batchsize', type=int,
                        default=256, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weightdecay', type=float, default=3e-5, help='weight decay')
    parser.add_argument('--rollbackthreshold', type=float, default=0.08, help='if loss increased this value, roll back 2*infostep steps')
    args = parser.parse_args()

    if(args.gpu==-1):
        device=torch.device('cpu')
    else:
        #print(torch.cuda.device_count())
        #os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        #print(torch.cuda.device_count())
        device = torch.device(f"cuda:{args.gpu}")

    if(args.savename=="auto"):
        args.savename=args.modeltype
        for i in args.modelparam:
            args.savename=args.savename+"_"+str(i)




    print("Counting Data Files.........................................................................................")

    tdata_files=[]
    if(os.path.splitext(args.tdatadir)[-1]=='.npz'): #single file
        tdata_files=[args.tdatadir]
    else:
        for (path,dirnames,filenames) in os.walk(args.tdatadir):
            filenames = [os.path.join(path,filename) for filename in filenames if filename.endswith('.npz')]
            tdata_files.extend(filenames)

    vdata_files=[]
    if(args.vdatadir is not None and os.path.exists(args.vdatadir)):
        if(os.path.splitext(args.vdatadir)[-1]=='.npz'): #single file
            vdata_files=[args.vdatadir]
        else:
            for (path,dirnames,filenames) in os.walk(args.vdatadir):
                filenames = [os.path.join(path,filename) for filename in filenames if filename.endswith('.npz')]
                vdata_files.extend(filenames)
    print("Finished counting data")

    os.makedirs("../saved_models",exist_ok=True)
    basepath = f'../saved_models/{args.savename}/'
    os.makedirs(basepath,exist_ok=True)
    backuppath=os.path.join(basepath,"backup")
    os.makedirs(backuppath,exist_ok=True)

    #tensorboard writer
    tensorboardpath=os.path.join(basepath,"tensorboardData")
    os.makedirs(tensorboardpath,exist_ok=True)
    train_writer=SummaryWriter(os.path.join(tensorboardpath,"train"))
    val_writer=SummaryWriter(os.path.join(tensorboardpath,"val"))

    print("Building model..............................................................................................")
    optimizer_state_dict_initial=None
    modelpath=os.path.join(basepath,"model.pth")
    modelpath_modelonly=os.path.join(basepath,"param.pth")
    if os.path.exists(modelpath) and (not args.new) and (args.savename != 'null'):
        modeldata = torch.load(modelpath,map_location="cpu")
        model_type=modeldata['model_type']
        model_param=modeldata['model_param']
        if("optimizer_state_dict" in modeldata):
            optimizer_state_dict_initial=modeldata["optimizer_state_dict"]
            print("Loaded optimizer state dict")
        model = ModelDic[model_type](*model_param).to(device)

        model.load_state_dict(modeldata['state_dict'])
        totalstep = modeldata['totalstep']
        print(f"Loaded model: type={model_type}, size={model_param}, totalstep={totalstep}")
    else:
        totalstep = 0
        model_type=args.modeltype
        model_param=args.modelparam
        model = ModelDic[model_type](*model_param).to(device)

    startstep=totalstep

    if model_type=='mix6' or model_type=='v1':
        print("Using mix6-like model, MLP uses lower(0.05x) weight decay")
        #lowl2param是一些密集型神经网络参数(mlp,cnn等)，对lr和weightdecay更敏感，使用float32计算，几乎不需要weightdecay
        #otherparam因为在c++代码中需要用int16计算，容易溢出，所以需要高的weightdecay控制范围
        lowl2param = list(map(id, model.mapping.parameters()))+\
                     list(map(id, model.value_linear1.parameters()))+\
                     list(map(id, model.value_linear2.parameters()))+\
                     list(map(id, model.value_linearfinal.parameters()))
        otherparam=list(filter(lambda p:id(p) not in lowl2param,model.parameters()))
        lowl2param=list(filter(lambda p:id(p) in lowl2param,model.parameters()))
        optimizer = optim.AdamW([{'params':otherparam},
                                {'params': lowl2param,'lr':args.lr,'weight_decay':0.05*args.weightdecay}],
                                lr=args.lr,weight_decay=args.weightdecay)
    elif model_type.startswith('v2') or model_type.startswith('v3'):
        print("Using v2-like model, MLP uses lower(0.05x) weight decay")
        highl2param = list(map(id,[model.h1conv.w,
                     model.trunkconv1.weight,
                     model.trunkconv2.w]))
        otherparam=list(filter(lambda p:id(p) not in highl2param,model.parameters()))
        highl2param=list(filter(lambda p:id(p) in highl2param,model.parameters()))

        optimizer = optim.AdamW([{'params':highl2param},
                                {'params': otherparam,'lr':args.lr,'weight_decay':0.05*args.weightdecay}],
                                lr=args.lr,weight_decay=args.weightdecay)
    elif model_type=='res':
        print("Using resnet model, Uses 0.2*default weight decay")
        optimizer = optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.2*args.weightdecay)
    else:
        print("Need weight decay settings for new model type")
        assert(False)

    # 加载优化器状态并恢复学习率和权重衰减
    if optimizer_state_dict_initial is not None:
        optimizer.load_state_dict(optimizer_state_dict_initial)
        # 恢复初始学习率和权重衰减
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = args.lr
            if i == 1 and (model_type == 'mix6' or model_type == 'v1'):
                param_group['weight_decay'] = 0.05 * args.weightdecay
            elif i == 1 and (model_type.startswith('v2') or model_type.startswith('v3')):
                param_group['weight_decay'] = 0.05 * args.weightdecay
            elif model_type == 'res':
                param_group['weight_decay'] = 0.2 * args.weightdecay
            else:
                param_group['weight_decay'] = args.weightdecay
    
    model.train()

    #for rollbacking if loss explodes
    modelbackup1=copy.deepcopy(model.state_dict())
    modelbackup2=copy.deepcopy(model.state_dict())
    modelbackup1_step=startstep
    modelbackup2_step=startstep
    modelbackup1_loss=1e10
    modelbackup2_loss=1e10

    time0=time.time()
    loss_record_init=[0,0,0,1e-30]
    loss_record=loss_record_init.copy()
    print("Start Training..............................................................................................")
    while True:
        tdata_file=random.choice(tdata_files)
        print(f"Selected training file: {tdata_file}")
        tDataset = trainset(tdata_file,randomsym=True,symcopy=False)
        print(f"{tDataset.__len__()} rows")
        tDataloader = DataLoader(tDataset, shuffle=True, batch_size=args.batchsize)

        for _ , (bf, gf, valueTarget, policyTarget) in enumerate(tDataloader):
            if(bf.shape[0]!=args.batchsize): #只要完整的batch
                continue
            if(random.random()>args.sampling): #随机舍去1-args.sampling的数据
                continue
            # data
            bf = bf.to(device)
            gf = gf.to(device)
            valueTarget = valueTarget.to(device)
            policyTarget = policyTarget.to(device)

            # optimize
            optimizer.zero_grad()
            value, policy = model(bf, gf)

            vloss = cross_entropy_loss(value, valueTarget)
            ploss = calculatePolicyLoss(policy, policyTarget)

            loss = 1.0*ploss
            if(random.random()<=args.valuesampling):
                loss=loss+vloss*VlossWeight
            loss_record[0]+=(vloss.detach().item()+ploss.detach().item())
            loss_record[1]+=vloss.detach().item()
            loss_record[2]+=ploss.detach().item()
            loss_record[3]+=1


            loss.backward()
            optimizer.step()

            # logs
            totalstep += 1
            if(totalstep % args.infostep == 0):
                time1=time.time()
                time_used=time1-time0
                time0=time1
                totalloss_train=loss_record[0]/loss_record[3]
                vloss_train=loss_record[1]/loss_record[3]
                ploss_train=loss_record[2]/loss_record[3]
                print("name: {}, time: {:.2f} s, step: {}, totalloss: {:.4f}, vloss: {:.4f}, ploss: {:.4f}"
                      .format(args.savename,time_used,totalstep,totalloss_train,vloss_train,ploss_train))
                train_writer.add_scalar("steps_each_second",loss_record[3]/time_used,global_step=totalstep)
                train_writer.add_scalar("totalloss",totalloss_train,global_step=totalstep)
                train_writer.add_scalar("vloss",vloss_train,global_step=totalstep)
                train_writer.add_scalar("ploss",ploss_train,global_step=totalstep)
                train_writer.add_scalar("lr",args.lr,global_step=totalstep)
                train_writer.add_scalar("batchsize",args.batchsize,global_step=totalstep)

                loss_record = loss_record_init.copy()

                #check whether loss explodes
                if(totalloss_train>modelbackup1_loss+args.rollbackthreshold):
                    print(f"loss explodes, rollback {2*args.infostep} steps to {modelbackup2_step}")
                    model.load_state_dict(modelbackup2)
                    modelbackup1=copy.deepcopy(modelbackup2)
                    totalstep=modelbackup2_step
                    modelbackup1_step=modelbackup2_step
                    modelbackup1_loss=modelbackup2_loss
                else:
                    #update backups
                    modelbackup2=modelbackup1
                    modelbackup2_step=modelbackup1_step
                    modelbackup2_loss=modelbackup1_loss
                    modelbackup1=copy.deepcopy(model.state_dict())
                    modelbackup1_step=totalstep
                    modelbackup1_loss=totalloss_train


            if((totalstep % args.savestep == 0) or (totalstep-startstep==args.maxstep) or (totalstep in backup_checkpoints)):

                print(f"Finished training {totalstep} steps")
                save_checkpoint(model,optimizer,modelpath)
                save_checkpoint(model,None,modelpath_modelonly)

                if(totalstep in backup_checkpoints):
                    modelpath_backup=os.path.join(backuppath,str(totalstep)+".pth")
                    save_checkpoint(model,optimizer,modelpath_backup)
                    print('Backup model saved in {}\n'.format(modelpath_backup))


                if vdata_files:
                    time0=time.time()
                    print("Start validation")
                    vdata_file = random.choice(vdata_files)
                    print(f"Selected validation file: {vdata_file}")
                    vDataset = trainset(vdata_file, randomsym=False, symcopy=False)
                    print(f"{vDataset.__len__()} rows")
                    vDataloader = DataLoader(vDataset, shuffle=False, batch_size=args.batchsize)
                    loss_record_val = loss_record_init.copy()
                    vsamp=0
                    model.eval()
                    with torch.no_grad():
                        for s, (bf, gf, valueTarget, policyTarget) in enumerate(vDataloader):
                            if(bf.shape[0]!=args.batchsize): #只要完整的batch
                                continue
                            vsamp+=args.batchsize
                            bf = bf.to(device)
                            gf = gf.to(device)
                            valueTarget = valueTarget.to(device)
                            policyTarget = policyTarget.to(device)

                            value, policy = model(bf,gf)

                            vloss = cross_entropy_loss(value, valueTarget)
                            ploss = calculatePolicyLoss(policy, policyTarget)
                            loss = VlossWeight*vloss+1.0*ploss

                            loss_record_val[0]+=(vloss.detach().item()+ploss.detach().item())
                            loss_record_val[1]+=vloss.detach().item()
                            loss_record_val[2]+=ploss.detach().item()
                            loss_record_val[3]+=1
                            if(vsamp>=args.maxvalsamp):
                                break

                    time1 = time.time()
                    time_used = time1 - time0
                    time0 = time1
                    totalloss_val = loss_record_val[0] / loss_record_val[3]
                    vloss_val = loss_record_val[1] / loss_record_val[3]
                    ploss_val = loss_record_val[2] / loss_record_val[3]
                    print("Validation: name: {}, time: {:.2f} s, step: {}, totalloss: {:.4f}, vloss: {:.4f}, ploss: {:.4f}"
                          .format(args.savename, time_used, totalstep, totalloss_val, vloss_val, ploss_val))
                    val_writer.add_scalar("steps_each_second", loss_record[3] / time_used, global_step=totalstep)
                    val_writer.add_scalar("totalloss", totalloss_val, global_step=totalstep)
                    val_writer.add_scalar("vloss", vloss_val, global_step=totalstep)
                    val_writer.add_scalar("ploss", ploss_val, global_step=totalstep)

                    model.train()

            if(totalstep - startstep >= args.maxstep):
                break