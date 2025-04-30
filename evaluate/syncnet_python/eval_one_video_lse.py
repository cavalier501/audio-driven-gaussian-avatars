
import os
import argparse
from SyncNetInstance import *
"""
date:2025.04.09
note:
GPT给出的版本
"""



parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='data/syncnet_v2.model')
parser.add_argument('--videofile', type=str, required=True, help='Path to video file (.avi)')
parser.add_argument('--reference', type=str, default='default', help='Name for temp directory')
args = parser.parse_args()

# 设置默认的工作目录结构
args.data_dir = 'data/work'
args.avi_dir = os.path.join(args.data_dir, 'pyavi')
args.tmp_dir = os.path.join(args.data_dir, 'pytmp')
args.work_dir = os.path.join(args.data_dir, 'pywork')
args.crop_dir = os.path.join(args.data_dir, 'pycrop')

# 创建目录
for d in [args.avi_dir, args.tmp_dir, args.work_dir, args.crop_dir]:
    os.makedirs(d, exist_ok=True)

# 初始化 SyncNet
s = SyncNetInstance()
s.loadParameters(args.model_path)
s.crop_video(args)  # 强制裁剪成嘴部视频

# 读取裁剪后的视频
import glob
flist = glob.glob(os.path.join(args.crop_dir, args.reference, '0*.avi'))
flist.sort()

# 没有检测到人脸
if len(flist) == 0:
    print("[ERROR] No face found in video after cropping.")
    exit()

# 对每一帧嘴部视频跑 evaluate
all_dists = []
all_confs = []
for fname in flist:
    offset, conf, dist = s.evaluate(args, videofile=fname)
    all_dists.extend(dist)
    all_confs.append(conf)

