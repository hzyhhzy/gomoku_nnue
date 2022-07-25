
import os
import multiprocessing
import numpy as np



def processSingleFile(loadpath,savepath):

    data = np.load(loadpath)
    bf = data["bf"]
    gf = data["gf"]
    vt = data["vt"]
    pt = data["pt"]

    useful=np.all(bf[:,0],axis=(1,2)) #15路
    useful=useful&(gf[:,3]==0) #有禁
    useful=useful&(gf[:,4]==1) #有禁
    useful=useful&(gf[:,5]==-1) #有禁黑
    useful=useful&(gf[:,7]==0) #不要直接输入vcf胜点的局面
    useful=useful&(gf[:,17]==0) #不要firstpasswin
    useful=useful&(np.all(gf[:,20:30]==0,axis=1)) #不要vcn
    useful=useful&(gf[:,30]==0) #不要maxmoves

    print("File "+loadpath+" "+str(sum(useful))+" of "+str(bf.shape[0])+" is useful")

    bf=bf[useful]
    gf=gf[useful]
    pt=pt[useful]
    vt=vt[useful]

    bf=bf[:,[1,2]]
    gf=gf[:,[0]]*0.0
    pt=pt
    vt=vt[:,[3,4,5]]

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
    if(filenum<num_threads):
        num_threads=filenum
    file_each_thread=filenum//num_threads
    start_ids=list(range(0,num_threads*file_each_thread,file_each_thread))
    end_ids=start_ids[1:]
    end_ids.append(filenum)
    print(start_ids,end_ids)
    all_file_split=[(all_files[start_ids[i]:end_ids[i]],loaddir,savedir) for i in range(num_threads)]
    print(all_file_split)
    with multiprocessing.Pool(num_threads) as pool:
        pool.starmap(processDirThread,all_file_split)




if __name__ == '__main__':
    processDir("vdata_processed","vdata_blackNormal",12)
    processDir("tdata_processed","tdata_blackNormal",12)
