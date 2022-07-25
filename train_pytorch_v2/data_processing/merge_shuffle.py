
import os
import multiprocessing
import numpy as np

def joint_shuffle(arrs):
  rand_state = np.random.get_state()
  for arr in arrs:
    assert(len(arr) == len(arrs[0]))
  for arr in arrs:
    np.random.set_state(rand_state)
    np.random.shuffle(arr)


keys = [
  "bf",
  "gf",
  "vt",
  "pt"
]

def mergeShuffleSingleDir(loaddir,savepath):
    print("Processing file "+savepath)
    all_files = []
    for (path, dirnames, filenames) in os.walk(loaddir):
        filenames = [os.path.join(path, filename) for filename in filenames if filename.endswith('.npz')]
        all_files.extend(filenames)
    #print(all_files)
    bfs=[]
    gfs=[]
    pts=[]
    vts=[]


    for f in all_files:
        np.random.seed([int.from_bytes(os.urandom(4), byteorder='little') for i in range(4)])
        npz = np.load(f)
        assert (set(npz.keys()) == set(keys))

        bfs.append(npz["bf"])
        gfs.append(npz["gf"])
        vts.append(npz["vt"])
        pts.append(npz["pt"])

    bfs = np.concatenate(bfs)
    gfs = np.concatenate(gfs)
    vts = np.concatenate(vts)
    pts = np.concatenate(pts)

    #print(savepath+" vts[0:10,:] before shuffle")
    #print(vts[0:10,:])
    joint_shuffle([bfs,gfs,vts,pts])
    #print(savepath+" vts[0:10,:] after shuffle")
    #print(vts[0:10,:])
    np.savez_compressed(savepath,
        bf = bfs,
        gf = gfs,
        vt = vts,
        pt = pts)


def processDirThread(loadbasedir,savedir,filesplitnums):

    for fn in filesplitnums:
        subdirname='part_'+str(fn)
        loaddir =os.path.join(loadbasedir,subdirname)
        savepath=os.path.join(savedir,subdirname+'.npz')
        mergeShuffleSingleDir(loaddir,savepath)

def processDir(loadbasedir,savedir,num_threads,filesplitnum):

    try:
        os.mkdir(savedir)
    except:
        print("Target dir already exists. Please delete it first")
        return
    else:
        pass

    if(num_threads>filesplitnum):
        num_threads=filesplitnum

    print("Processing-------------------------------------------------------------------------")
    filesplitnums_eachThread=[[i for i in range(filesplitnum) if i%num_threads == j] for j in range(num_threads)]
    inputparam__eachThread=[(loadbasedir,savedir,i) for i in filesplitnums_eachThread]

    with multiprocessing.Pool(num_threads) as pool:
        pool.starmap(processDirThread,inputparam__eachThread)




if __name__ == '__main__':
    processDir("vdata_1","vdata_processed",8,8)
    processDir("tdata_1","tdata_processed",8,256)
