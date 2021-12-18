
import os
import multiprocessing
import numpy as np

boardH =15
boardW = 15
dataL =15
assert(dataL == boardW)


def unpackBoardFeatures(packedData):
    # 原始的训练数据是二进制uint8格式
    # 棋盘100个格点（前70个有效，后30个为0），每8个数合成一个uint8，所以最后一维是100/8=13，需要拆开
    dataNum, featureNum, dataLen = np.shape(packedData)

    # 原来22个通道，大部分没用
    usefulChannels = [1, 2]  # my,opp
    packedData = packedData[:, usefulChannels, :]
    featureNum = len(usefulChannels)

    unpackedData = np.zeros(
        [dataNum, featureNum, dataLen, 8], dtype=np.float32)
    for i in range(8):
        unpackedData[:, :, :, 7 - i] = packedData % 2
        packedData = packedData // 2
    unpackedData = np.reshape(
        unpackedData, ([dataNum, featureNum, dataLen * 8]))
    unpackedData = unpackedData[:, :, 0:boardW * boardH]
    unpackedData = np.reshape(
        unpackedData, ([dataNum, featureNum, boardH, boardW]))
    return unpackedData




def unpackGlobalFeatures(packedData):

    # 原来19个通道，大部分没用
    usefulChannels = []  # 5: black(-1) or white(1)
    if(not usefulChannels):
        return 0.0*packedData[:, [0]]
    return packedData[:, usefulChannels]


def unpackValueTarget(packedData):

    return packedData[:, [0, 1, 2]]  # win loss draw

def unpackPolicyTarget(packedData):
    #print("unpacking PolicyTarget")
    dataNum, featureNum, dataLen = np.shape(packedData)
    packedData = packedData[:, 0, 0:boardW * boardH]
    packedData = np.reshape(packedData, ([dataNum, boardH, boardW]))
    packedData = packedData+1e-7
    wsum = np.sum(packedData, axis=(1,2), keepdims=True)
    #print(f"\twsum.shape = {wsum.shape}")
    packedData = packedData/wsum
    return packedData.astype(np.float32)

def processData(loadpath):
    #处理单个文件
    data = np.load(loadpath)
    bf= unpackBoardFeatures(data["binaryInputNCHWPacked"])
    gf= unpackGlobalFeatures(data["globalInputNC"])
    vt = unpackValueTarget(data["globalTargetsNC"])
    pt = unpackPolicyTarget(data["policyTargetsNCMove"])

    print(f"total rows {data.shape[0]}")
    return bf,gf,vt,pt

def processAndSave(loadpath,savepath):

    bf,gf,vt,pt=processData(loadpath)
    np.savez_compressed(savepath,bf=bf,gf=gf,vt=vt,pt=pt)

def processDirThread(files,savedir,startID):
    i=0
    for f in files:
        savename=f"data_{i+startID}.npz"
        savename=os.path.join(savedir, savename)
        processAndSave(f,savename)
        i=i+1
        print(f"{i} of {len(files)}")

def processDir(loaddir,savedir,num_threads):

    try:
        os.mkdir(savedir)
    except:
        pass
    else:
        pass

    all_files=[]
    for (path,dirnames,filenames) in os.walk(loaddir):
        filenames = [os.path.join(path,filename) for filename in filenames if filename.endswith('.npz')]
        all_files.extend(filenames)

    print("Processing-------------------------------------------------------------------------")
    filenum=len(all_files)
    file_each_thread=filenum//num_threads
    start_ids=list(range(0,num_threads*file_each_thread,file_each_thread))
    end_ids=start_ids[1:]
    end_ids.append(filenum)
    print(start_ids,end_ids)
    all_file_split=[(all_files[start_ids[i]:end_ids[i]],savedir ,start_ids[i]) for i in range(num_threads)]
    print(all_file_split)
    with multiprocessing.Pool(num_threads) as pool:
        pool.starmap(processDirThread,all_file_split)




if __name__ == '__main__':
    processDir("data_p0","data_p1",8)
