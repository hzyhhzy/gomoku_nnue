#!/usr/bin/python3
#This is a combination of shuffle_newest.py and dataset_p0_to_p1.py
#To directly shuffle katago data
import sys
import os
import argparse
import traceback
import math
import time
import logging
import zipfile
import shutil
import json

import multiprocessing

import numpy as np

from data_convert import unpackBoardFeatures,unpackGlobalFeatures,unpackValueTarget,unpackPolicyTarget




def joint_shuffle(arrs):
  rand_state = np.random.get_state()
  for arr in arrs:
    assert(len(arr) == len(arrs[0]))
  for arr in arrs:
    np.random.set_state(rand_state)
    np.random.shuffle(arr)



def shardify( input_file,  keep_prob):
  np.random.seed([int.from_bytes(os.urandom(4), byteorder='little') for i in range(4)])

  #print("Shardify reading: " + input_file)
  npz = np.load(input_file)
  #assert(set(npz.keys()) == set(keys))

  ###
  #WARNING - if adding anything here, also add it to joint_shuffle below!
  ###
  bf = npz["binaryInputNCHWPacked"]
  gf = npz["globalInputNC"]
  vt = npz["globalTargetsNC"]
  pt = npz["policyTargetsNCMove"]

  rows_all=bf.shape[0]
  keep_ids=np.where(np.random.uniform(size=rows_all)<keep_prob)[0]

  #if(keep_ids.size==0):
  #  keep_ids=[0]

  bf=bf[keep_ids]
  gf=gf[keep_ids]
  vt=vt[keep_ids]
  pt=pt[keep_ids]

  bf=unpackBoardFeatures(bf)
  gf=unpackGlobalFeatures(gf)
  vt=unpackValueTarget(vt)
  pt=unpackPolicyTarget(pt)
  #joint_shuffle((bf,gf,vt,pt))

  #num_rows_to_keep = bf.shape[0]
  #assert(gf.shape[0] == num_rows_to_keep)
  #assert(vt.shape[0] == num_rows_to_keep)
  #assert(pt.shape[0] == num_rows_to_keep)

  return bf,gf,vt,pt

def merge_shards(data,filepath):

  #print(1)

  bfs=[i[0] for i in data]
  gfs=[i[1] for i in data]
  vts=[i[2] for i in data]
  pts=[i[3] for i in data]

  #print(2)
  bfs = np.concatenate(bfs)
  gfs = np.concatenate(gfs)
  vts = np.concatenate(vts)
  pts = np.concatenate(pts)

  #print(3)

  # datasize=len(bfs)
  # rand_id=np.random.permutation(datasize)
  #
  # bfs=bfs.take(rand_id,axis=0)
  # gfs=gfs.take(rand_id,axis=0)
  # vts=vts.take(rand_id,axis=0)
  # pts=pts.take(rand_id,axis=0)

  joint_shuffle((bfs,gfs,vts,pts))

  #print(4)


  np.savez_compressed(filepath,
      bf = bfs,
      gf = gfs,
      vt = vts,
      pt = pts)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Shuffle data files')
  parser.add_argument('-dir', type=str,default='dir/selfplay',help='Directories of training data files')
  parser.add_argument('-keep', type=int, default=1000000, help='Target number of rows to actually keep in the final data set, default 1.2M')
  parser.add_argument('-out-file', type=str,default="data_shuffled.npz", help='Dir to output training files')
  parser.add_argument('-num-processes', type=int, default=10, help='Number of multiprocessing processes')

  args = parser.parse_args()
  dir = args.dir
  keep_target_rows = args.keep
  out_file = args.out_file
  num_processes = args.num_processes


  all_files = []
  for (path,dirnames,filenames) in os.walk(dir):
    filenames = [os.path.join(path,filename) for filename in filenames if filename.endswith('.npz')]
    filenames = [(filename,os.path.getmtime(filename)) for filename in filenames]
    all_files.extend(filenames)

  all_files.sort(key=(lambda x: x[1]), reverse=False)

  def get_numpy_npz_headers(filename):
    with zipfile.ZipFile(filename) as z:
      wasbad = False
      numrows = 0
      npzheaders = {}
      for subfilename in z.namelist():
        npyfile = z.open(subfilename)
        try:
          version = np.lib.format.read_magic(npyfile)
        except ValueError:
          wasbad = True
          print("WARNING: bad file, skipping it: %s (bad array %s)" % (filename,subfilename))
        else:
          (shape, is_fortran, dtype) = np.lib.format._read_array_header(npyfile,version)
          npzheaders[subfilename] = (shape, is_fortran, dtype)
      if wasbad:
        return None
      return npzheaders


  files_with_row_range = []
  num_rows_total = 0 #Number of data rows

  filenum=len(all_files)
  usefilenum=min(math.floor(1.0*filenum**0.6),filenum)
  all_files=all_files[-usefilenum:]
  print("Using files:",all_files);

  #i=0
  for (filename,mtime) in all_files:
    #i=i+1
    #print(i)
    npheaders = get_numpy_npz_headers(filename)
    if npheaders is None or len(npheaders) <= 0:
      continue
    (shape, is_fortran, dtype) = npheaders["binaryInputNCHWPacked"]
    num_rows = shape[0]
    row_range = (num_rows_total, num_rows_total + num_rows)
    num_rows_total += num_rows

    files_with_row_range.append((filename,row_range))



  if num_rows_total <= 0:
    print("No rows found")
    sys.exit(0)

  print("Total rows found: %d" % (num_rows_total))



  files = [i[0] for i in all_files]


  np.random.seed()
  np.random.shuffle(files)

  approx_rows_to_keep = num_rows_total
  if keep_target_rows is not None:
    approx_rows_to_keep = min(approx_rows_to_keep, keep_target_rows)
  keep_prob = approx_rows_to_keep / num_rows_total


  with multiprocessing.Pool(num_processes) as pool:
    print("Beginning sharding",flush=True)
    t0 = time.time()
    shard_results = pool.starmap(shardify, [(f,keep_prob) for f in files])
    t1 = time.time()
    print("Done sharding, number of shards by input file:",flush=True)
    # print(list(zip(desired_input_files,shard_results)),flush=True)
    print("Time taken: " + str(t1-t0),flush=True)
    sys.stdout.flush()

    print("Beginning merging",flush=True)
    t0 = time.time()
    merge_shards(shard_results,out_file)
    t1 = time.time()
    print("Done merging",flush=True)
    print("Time taken: " + str(t1-t0),flush=True)
    sys.stdout.flush()
