import os
import multiprocessing
import numpy as np
import config

filter_singleSize=False #only use 19x19 games

filter_noFPW=True #no "first pass win" rule
filter_noVCN=False #no VCF/VCT/VC2 rule
filter_noMM=False #no MaxMove rule

filter_noVC6=True #no "VC6" (win in this move, or lose)
filter_noSimpleWin=True #no single-move winning positions
filter_noPDA=True #no PDA mode games

bfReserveCh=[0,1,2,3,4] #onboard my opp lastmove islegal
assert(0 in bfReserveCh or filter_singleSize)
gfReserveCh=[i for i in range(39)] #all 
vtReserveCh=[3,4,5] # win loss draw wr_win wr_loss wr_draw

def processSingleFile(loadpath,savepath):

    data = np.load(loadpath)
    bf = data["bf"]
    gf = data["gf"]
    vt = data["vt"]
    pt = data["pt"]

    useful=np.any(bf[:,0],axis=(1,2)) #all-True
    if(filter_singleSize):
        useful=useful & np.all(bf[:,0],axis=(1,2)) 
    if(filter_noFPW):
        useful=useful & (gf[:,17]==0) 
    if(filter_noVCN):
        useful=useful & np.all(gf[:,20:30]==0,axis=1) 
    if(filter_noMM):
        useful=useful & (gf[:,30]==0) 
    if(filter_noVC6):
        useful=useful & (gf[:,24]==0) 
        useful=useful & (gf[:,29]==0) 
    if(filter_noSimpleWin):
        useful=useful&(gf[:,11]==0) 
    if(filter_noPDA):
        useful=useful&(gf[:,15]==0)

    print("File "+loadpath+" "+str(sum(useful))+" of "+str(bf.shape[0])+" is useful")

    bf=bf[useful]
    gf=gf[useful]
    pt=pt[useful]
    vt=vt[useful]

    bf=bf[:,bfReserveCh]
    gf=gf[:,gfReserveCh]
    pt=pt
    vt=vt[:,vtReserveCh]

    if(config.NoPass):
        pt = pt[:, 0, :-1]
        assert(pt.shape[2]==config.boardW * config.boardH)
        pt = pt+1e-10
        wsum = np.sum(pt, axis=(1), keepdims=True)
        pt = pt/wsum

    np.savez_compressed(savepath,bf=bf,gf=gf,vt=vt,pt=pt)

def processDirThread(files,loaddir,savedir):
    i=0
    for f in files:
        loadpath=os.path.join(loaddir,f)
        savepath=os.path.join(savedir,f)
        processSingleFile(loadpath,savepath)
        i=i+1
        print(f"{i} of {len(files)}")

def processDir(loaddir,savedir,num_threads):

    try:
        os.mkdir(savedir)
    except:
        print("Target dir already exists. Please delete it first")
        return
    else:
        pass

    all_files=[]
    for (path,dirnames,filenames) in os.walk(loaddir):
        if(loaddir!=path):  #only root
            continue
        filenames = [filename for filename in filenames if filename.endswith('.npz')]
        all_files.extend(filenames)

    print("Processing-------------------------------------------------------------------------")
    filenum=len(all_files)
    #if(filenum<num_threads):
    #    num_threads=filenum
    file_each_thread=1+(filenum-1)//num_threads
    num_threads=1+(filenum-1)//file_each_thread
    start_ids=list(range(0,num_threads*file_each_thread,file_each_thread))
    end_ids=start_ids[1:]
    end_ids.append(filenum)
    print(start_ids,end_ids)
    all_file_split=[(all_files[start_ids[i]:end_ids[i]],loaddir,savedir) for i in range(num_threads)]
    print(all_file_split)
    with multiprocessing.Pool(num_threads) as pool:
        pool.starmap(processDirThread,all_file_split)




if __name__ == '__main__':
    processDir("vdata_merged","vdata_filtered",num_threads=config.cpuThread)
    processDir("tdata_merged","tdata_filtered",num_threads=config.cpuThread)
