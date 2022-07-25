
from dataset import trainset
from model import ModelDic
from config import boardH,boardW

import argparse
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch
import os
import time
import random
import copy

backup_checkpoints=[20000*i for i in range(500)]


if not os.path.exists("../saved_models"):
    os.mkdir("../saved_models")

def lossSoftFunction(losses):
    #losses=torch.pow(losses+0.25,0.5)-0.5
    return torch.mean(losses,dim=0)


def cross_entropy_loss(output, target):
    t = torch.log_softmax(output,dim=1)
    losses = torch.sum(-t*target, dim=1)+torch.sum(torch.log(target+1e-10)*target, dim=1)
    return lossSoftFunction(losses)


def calculatePolicyLoss(output,pt):
    output=torch.flatten(output,start_dim=1)
    if(output.shape[1]==boardW * boardH): #no pass
        pt = pt[:, 0:boardW * boardH]
    pt = pt+1e-10
    wsum = torch.sum(pt, dim=1, keepdims=True)
    pt = pt/wsum

    return cross_entropy_loss(output,pt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #data settings
    parser.add_argument('--tdatadir', type=str, default='../data/tdata_blackNormal', help='train dataset path: dir include dataset files or single dataset file')
    parser.add_argument('--vdatadir', type=str, default='../data/vdata_blackNormal/part_0.npz', help='validation dataset path: dir include dataset files or single dataset file')
    parser.add_argument('--maxstep', type=int, default=5000000000, help='max step to train')
    parser.add_argument('--savestep', type=int, default=10000, help='step to save and validation')
    parser.add_argument('--infostep', type=int, default=100, help='step to logger')

    parser.add_argument('--sampling', type=float, default=1, help='sampling rate(to avoid overfitting)')
    parser.add_argument('--valuesampling', type=float, default=1, help='value sampling rate(to avoid overfitting)')

    #model parameters
    parser.add_argument('--modeltype', type=str, default='v2bc',help='model type defined in model.py')
    parser.add_argument('--modelparam', nargs='+',type=int,
                        default=(5,256), help='model size')

    parser.add_argument('--savename', type=str ,default='null', help='model save pth, ""null"" means does not save, ""auto"" means modeltype+modelparam')
    parser.add_argument('--new', action='store_true', default=False, help='whether to retrain')

    #training parameters
    parser.add_argument('--gpu', type=int,
                        default=0, help='which gpu, -1 means cpu')
    parser.add_argument('--batchsize', type=int,
                        default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--weightdecay', type=float, default=1e-7, help='weight decay')
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

    basepath = f'../saved_models/{args.savename}/'
    if not os.path.exists(basepath):
        os.mkdir(basepath)
    backuppath=os.path.join(basepath,"backup")
    if not os.path.exists(backuppath):
        os.mkdir(backuppath)
    tensorboardpath=os.path.join(basepath,"tensorboardData")

    #tensorboard writer
    if not os.path.exists(tensorboardpath):
        os.mkdir(tensorboardpath)
    train_writer=SummaryWriter(os.path.join(tensorboardpath,"train"))
    val_writer=SummaryWriter(os.path.join(tensorboardpath,"val"))

    print("Building model..............................................................................................")
    modelpath=os.path.join(basepath,"model.pth")
    if os.path.exists(modelpath) and (not args.new) and (args.savename != 'null'):
        modeldata = torch.load(modelpath)
        model_type=modeldata['model_type']
        model_param=modeldata['model_param']
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

    if model_type=='mix6':
        #lowl2param是一些密集型神经网络参数(mlp,cnn等)，对lr和weightdecay更敏感，使用float32计算，几乎不需要weightdecay
        #otherparam因为在c++代码中需要用int16计算，容易溢出，所以需要高的weightdecay控制范围
        lowl2param = list(map(id, model.mapping.parameters()))+\
                     list(map(id, model.value_leakyrelu.parameters()))+\
                     list(map(id, model.value_linear1.parameters()))+\
                     list(map(id, model.value_linear2.parameters()))+\
                     list(map(id, model.value_linearfinal.parameters()))
        otherparam=list(filter(lambda p:id(p) not in lowl2param,model.parameters()))
        lowl2param=list(filter(lambda p:id(p) in lowl2param,model.parameters()))

        optimizer = optim.Adam([{'params':otherparam},
                                {'params': lowl2param,'lr':args.lr,'weight_decay':1e-7}],
                                lr=args.lr,weight_decay=args.weightdecay)
    else:
        optimizer = optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weightdecay)
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

        for _ , (board, valueTarget, policyTarget) in enumerate(tDataloader):
            if(board.shape[0]!=args.batchsize): #只要完整的batch
                continue
            if(random.random()>args.sampling): #随机舍去1-args.sampling的数据
                continue
            # data
            board = board.to(device)
            valueTarget = valueTarget.to(device)
            policyTarget = policyTarget.to(device)

            # optimize
            optimizer.zero_grad()
            value, policy = model(board)

            vloss = cross_entropy_loss(value, valueTarget)
            ploss = calculatePolicyLoss(policy, policyTarget)

            loss = 1.0*ploss
            if(random.random()<=args.valuesampling):
                loss=loss+vloss*1.2
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
                torch.save(
                    {'totalstep': totalstep,
                     'state_dict': model.state_dict(),
                     'model_type': model.model_type,
                     'model_param':model.model_param},
                    modelpath)
                print('Model saved in {}\n'.format(modelpath))

                if(totalstep in backup_checkpoints):
                    modelpath_backup=os.path.join(backuppath,str(totalstep)+".pth")
                    torch.save(
                        {'totalstep': totalstep,
                         'state_dict': model.state_dict(),
                         'model_type': model.model_type,
                         'model_param':model.model_param},
                        modelpath_backup)
                    print('Model saved in {}\n'.format(modelpath_backup))


                if vdata_files:
                    time0=time.time()
                    print("Start validation")
                    vdata_file = random.choice(vdata_files)
                    print(f"Selected validation file: {vdata_file}")
                    vDataset = trainset(vdata_file, randomsym=False, symcopy=False)
                    print(f"{vDataset.__len__()} rows")
                    vDataloader = DataLoader(vDataset, shuffle=False, batch_size=args.batchsize)
                    loss_record_val = loss_record_init.copy()
                    model.eval()
                    with torch.no_grad():
                        for s, (board, valueTarget, policyTarget) in enumerate(vDataloader):
                            if(board.shape[0]!=args.batchsize): #只要完整的batch
                                continue
                            board = board.to(device)
                            valueTarget = valueTarget.to(device)
                            policyTarget = policyTarget.to(device)

                            value, policy = model(board)

                            vloss = cross_entropy_loss(value, valueTarget)
                            ploss = calculatePolicyLoss(policy, policyTarget)
                            loss = 1.2*vloss+1.0*ploss

                            loss_record_val[0]+=(vloss.detach().item()+ploss.detach().item())
                            loss_record_val[1]+=vloss.detach().item()
                            loss_record_val[2]+=ploss.detach().item()
                            loss_record_val[3]+=1

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