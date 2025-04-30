#!/usr/bin/python
#-*- coding: utf-8 -*-

import time, pdb, argparse, subprocess, pickle, os, gzip, glob

from SyncNetInstance import *

# ==================== PARSE ARGUMENT ====================

parser = argparse.ArgumentParser(description = "SyncNet");
parser.add_argument('--initial_model', type=str, default="data/syncnet_v2.model", help='');
parser.add_argument('--batch_size', type=int, default='20', help='');
parser.add_argument('--vshift', type=int, default='15', help='');
parser.add_argument('--data_dir', type=str, default='data/work', help='');
parser.add_argument('--videofile', type=str, default='', help='');
parser.add_argument('--reference', type=str, default='', help='');
opt = parser.parse_args();

setattr(opt,'avi_dir',os.path.join(opt.data_dir,'pyavi'))
setattr(opt,'tmp_dir',os.path.join(opt.data_dir,'pytmp'))
setattr(opt,'work_dir',os.path.join(opt.data_dir,'pywork'))
setattr(opt,'crop_dir',os.path.join(opt.data_dir,'pycrop'))


# ==================== LOAD MODEL AND FILE LIST ====================

s = SyncNetInstance();

s.loadParameters(opt.initial_model);
print("Model %s loaded."%opt.initial_model);
# s.crop_video(opt)

flist = glob.glob(os.path.join(opt.crop_dir,opt.reference,'0*.avi'))
flist.sort()



# https://github.com/joonson/syncnet_python/issues/71
# bug 修正
# ==================== GET OFFSETS ====================

dists = []
confs=[]
offsets=[]
# breakpoint()
for idx, fname in enumerate(flist):
    offset, conf, dist = s.evaluate(opt,videofile=fname)
    dists.append(dist)
    offsets.append(offset)
    confs.append(conf)





# ==================== PRINT RESULTS TO FILE ====================
# input(os.path.join(opt.work_dir,opt.reference,'tracks.pckl'))
with open(os.path.join(opt.work_dir,opt.reference,'tracks.pckl'), 'rb') as fil:
     tracks = pickle.load(fil, encoding='latin1')

with open(os.path.join(opt.work_dir,opt.reference,'offsets.txt'), 'w') as fil:
     fil.write('FILENAME\tOFFSET\tCONF\n')
     for ii, track in enumerate(tracks):
       fil.write('%05d.avi\t%d\t%.3f\n'%(ii, offsets[ii], confs[ii]))


save_dir=os.path.join(opt.work_dir,opt.reference)
os.makedirs(save_dir,exist_ok=True)
save_file = os.path.join(opt.work_dir,opt.reference,'activesd.pckl')
with open(save_file, 'wb') as fil:
      pickle.dump(dists, fil)
import sys
sys.exit()


# ==================== GET OFFSETS ====================

dists = []
for idx, fname in enumerate(flist):
    offset, conf, dist = s.evaluate(opt,videofile=fname)
    dists.append(dist)
      
# ==================== PRINT RESULTS TO FILE ====================
save_file = os.path.join(opt.work_dir,opt.reference,'activesd.pckl')
with open(save_file, 'wb') as fil:
    pickle.dump(dists, fil)
# with open(os.path.join(opt.work_dir,opt.reference,'activesd.pckl'), 'wb') as fil:
#     pickle.dump(dists, fil)

# # GPT建议
# print("Mean LSE-D:", sum(dists) / len(dists))
# offset, conf, dist = s.evaluate(opt,videofile=fname)
# print("LSE-C:", conf)
# print("LSE-D (avg):", sum(dist)/len(dist))
