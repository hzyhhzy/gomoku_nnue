
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
    parser.add_argument('--vdatadir', type=str, default='../data/vdata_blackNormal/part_0.npz', help='validation dataset path: dir include dataset files or single dataset file')
    parser.add_argument('--maxstep', type=int, default=5000000000, help='max step to train')
    parser.add_argument('--modelname', type=str ,required=True, help='model path')

    #training parameters
    parser.add_argument('--gpu', type=int,
                        default=0, help='which gpu, -1 means cpu')
    parser.add_argument('--batchsize', type=int,
                        default=128, help='batch size')
    args = parser.parse_args()

    if(args.gpu==-1):
        device=torch.device('cpu')
    else:
        #print(torch.cuda.device_count())
        #os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        #print(torch.cuda.device_count())
        device = torch.device(f"cuda:{args.gpu}")




    print("Counting Data Files.........................................................................................")


    vdata_files=[]
    if(args.vdatadir is not None and os.path.exists(args.vdatadir)):
        if(os.path.splitext(args.vdatadir)[-1]=='.npz'): #single file
            vdata_files=[args.vdatadir]
        else:
            for (path,dirnames,filenames) in os.walk(args.vdatadir):
                filenames = [os.path.join(path,filename) for filename in filenames if filename.endswith('.npz')]
                vdata_files.extend(filenames)
    else:
        print("No validation files")
        exit(0)
    print("Finished counting data")

    basepath = f'../saved_models/{args.modelname}/'
    if not os.path.exists(basepath):
        os.mkdir(basepath)
    tensorboardpath=os.path.join(basepath,"tensorboardData")

    #tensorboard writer
    if not os.path.exists(tensorboardpath):
        os.mkdir(tensorboardpath)
    val_writer=SummaryWriter(os.path.join(tensorboardpath,"noisetest"))

    print("Building model..............................................................................................")
    modelpath=os.path.join(basepath,"model.pth")
    if os.path.exists(modelpath):
        modeldata = torch.load(modelpath,map_location="cpu")
        model_type=modeldata['model_type']
        model_param=modeldata['model_param']
        model = ModelDic[model_type](*model_param).to(device)

        model.load_state_dict(modeldata['state_dict'])
        totalstep = modeldata['totalstep']
        print(f"Loaded model: type={model_type}, size={model_param}, totalstep={totalstep}")
    else:
        print("No model file")
        exit(0)


    model.eval()

    loss_record_init=[0,0,0,1e-30]

    vdata_file = random.choice(vdata_files)
    print(f"Selected validation file: {vdata_file}")
    vDataset = trainset(vdata_file, randomsym=False, symcopy=False)
    print(f"{vDataset.__len__()} rows")

    for mapnoisei in range(25):
        mapnoise=mapnoisei*0.1
        time0 = time.time()
        vDataloader = DataLoader(vDataset, shuffle=False, batch_size=args.batchsize)
        loss_record_val = loss_record_init.copy()
        with torch.no_grad():
            for s, (board, valueTarget, policyTarget) in enumerate(vDataloader):
                if (board.shape[0] != args.batchsize):  # 只要完整的batch
                    continue
                board = board.to(device)
                valueTarget = valueTarget.to(device)
                policyTarget = policyTarget.to(device)

                value, policy = model(board,mapnoise=mapnoise)

                vloss = cross_entropy_loss(value, valueTarget)
                ploss = calculatePolicyLoss(policy, policyTarget)
                loss = 1.2 * vloss + 1.0 * ploss

                loss_record_val[0] += (vloss.detach().item() + ploss.detach().item())
                loss_record_val[1] += vloss.detach().item()
                loss_record_val[2] += ploss.detach().item()
                loss_record_val[3] += 1
                if(loss_record_val[3]>=args.maxstep):
                    break

        time1 = time.time()
        time_used = time1 - time0
        time0 = time1
        totalloss_val = loss_record_val[0] / loss_record_val[3]
        vloss_val = loss_record_val[1] / loss_record_val[3]
        ploss_val = loss_record_val[2] / loss_record_val[3]
        print("Validation: name: {}, time: {:.2f} s, mapnoise: {:.4f}, totalloss: {:.4f}, vloss: {:.4f}, ploss: {:.4f}"
              .format(args.modelname, time_used, mapnoise, totalloss_val, vloss_val, ploss_val))
        val_writer.add_scalar("totalloss", totalloss_val, global_step=mapnoise*10000)
        val_writer.add_scalar("vloss", vloss_val, global_step=mapnoise*10000)
        val_writer.add_scalar("ploss", ploss_val, global_step=mapnoise*10000)

