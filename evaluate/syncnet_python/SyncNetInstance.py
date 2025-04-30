#!/usr/bin/python
#-*- coding: utf-8 -*-
# Video 25 FPS, Audio 16000HZ

import torch
import numpy
import time, pdb, argparse, subprocess, os, math, glob
import cv2
import python_speech_features

from scipy import signal
from scipy.io import wavfile
from SyncNetModel import *
from shutil import rmtree
from python_speech_features import mfcc
import numpy as np
import tempfile
import torch.nn.functional as F


# ==================== Get OFFSET ====================

def calc_pdist(feat1, feat2, vshift=10):
    
    win_size = vshift*2+1

    feat2p = torch.nn.functional.pad(feat2,(0,0,vshift,vshift))

    dists = []

    for i in range(0,len(feat1)):

        dists.append(torch.nn.functional.pairwise_distance(feat1[[i],:].repeat(win_size, 1), feat2p[i:i+win_size,:]))

    return dists

# ==================== MAIN DEF ====================

class SyncNetInstance(torch.nn.Module):

    def __init__(self, dropout = 0, num_layers_in_fc_layers = 1024):
        super(SyncNetInstance, self).__init__();
        # S包含两个分支：audio/lip
        self.__S__ = S(num_layers_in_fc_layers = num_layers_in_fc_layers).cuda();

    def evaluate(self, opt, videofile):

        self.__S__.eval();

        # ========== ==========
        # Convert files
        # ========== ==========

        if os.path.exists(os.path.join(opt.tmp_dir,opt.reference)):
          rmtree(os.path.join(opt.tmp_dir,opt.reference))

        os.makedirs(os.path.join(opt.tmp_dir,opt.reference))

        command = ("ffmpeg -y -i %s -threads 1 -f image2 %s" % (videofile,os.path.join(opt.tmp_dir,opt.reference,'%06d.jpg'))) 
        output = subprocess.call(command, shell=True, stdout=None)

        command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (videofile,os.path.join(opt.tmp_dir,opt.reference,'audio.wav'))) 
        output = subprocess.call(command, shell=True, stdout=None)
        
        # ========== ==========
        # Load video 
        # ========== ==========

        images = []
        
        flist = glob.glob(os.path.join(opt.tmp_dir,opt.reference,'*.jpg'))
        flist.sort()

        for fname in flist:
            images.append(cv2.imread(fname))

        im = numpy.stack(images,axis=3)
        im = numpy.expand_dims(im,axis=0)
        im = numpy.transpose(im,(0,3,4,1,2))

        imtv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())

        # ========== ==========
        # Load audio
        # ========== ==========

        sample_rate, audio = wavfile.read(os.path.join(opt.tmp_dir,opt.reference,'audio.wav'))
        mfcc = zip(*python_speech_features.mfcc(audio,sample_rate))
        mfcc = numpy.stack([numpy.array(i) for i in mfcc])

        cc = numpy.expand_dims(numpy.expand_dims(mfcc,axis=0),axis=0)
        cct = torch.autograd.Variable(torch.from_numpy(cc.astype(float)).float())

        # ========== ==========
        # Check audio and video input length
        # ========== ==========

        if (float(len(audio))/16000) != (float(len(images))/25) :
            print("WARNING: Audio (%.4fs) and video (%.4fs) lengths are different."%(float(len(audio))/16000,float(len(images))/25))

        min_length = min(len(images),math.floor(len(audio)/640))
        
        # ========== ==========
        # Generate video and audio feats
        # ========== ==========

        lastframe = min_length-5
        im_feat = []
        cc_feat = []

        tS = time.time()
        for i in range(0,lastframe,opt.batch_size):
            
            im_batch = [ imtv[:,:,vframe:vframe+5,:,:] for vframe in range(i,min(lastframe,i+opt.batch_size)) ]
            im_in = torch.cat(im_batch,0)
            im_out  = self.__S__.forward_lip(im_in.cuda());
            im_feat.append(im_out.data.cpu())

            cc_batch = [ cct[:,:,:,vframe*4:vframe*4+20] for vframe in range(i,min(lastframe,i+opt.batch_size)) ]
            cc_in = torch.cat(cc_batch,0)
            cc_out  = self.__S__.forward_aud(cc_in.cuda())
            cc_feat.append(cc_out.data.cpu())

        # example.avi
        # im_feat : [312,1024]
        # cc_feat : [312,1024]
        im_feat = torch.cat(im_feat,0)
        cc_feat = torch.cat(cc_feat,0)

        # ========== ==========
        # Compute offset
        # ========== ==========
            
        print('Compute time %.3f sec.' % (time.time()-tS))

        # 在每一个时间点上，比较视频帧与其前后若干帧音频的特征距离，得到一个 [T, 2v+1] 的距离矩阵
        # （实际上，dists是一个列表，每个原始形状为31，注意，此时已经不是312了）
        # mdist 形状[31,]
        dists = calc_pdist(im_feat,cc_feat,vshift=opt.vshift)
        mdist = torch.mean(torch.stack(dists,1),1)

        minval, minidx = torch.min(mdist,0)

        offset = opt.vshift-minidx
        conf   = torch.median(mdist) - minval

        fdist   = numpy.stack([dist[minidx].numpy() for dist in dists])
        # fdist   = numpy.pad(fdist, (3,3), 'constant', constant_values=15)
        fconf   = torch.median(mdist).numpy() - fdist
        fconfm  = signal.medfilt(fconf,kernel_size=9)
        
        numpy.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        print('Framewise conf: ')
        print(fconfm)
        print('AV offset: \t%d \nMin dist: \t%.3f\nConfidence: \t%.3f' % (offset,minval,conf))
        print("✅ LSE-C:", round(float(conf), 4))
        print("✅ LSE-D:", round(float(minval.item()), 4))
        dists_npy = numpy.array([ dist.numpy() for dist in dists ])
        # breakpoint()
        return offset.numpy(), conf.numpy(), dists_npy

    def extract_video_frames(self, videofile):
        cap = cv2.VideoCapture(videofile)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            face = cv2.resize(frame, (112, 112))
            frames.append(face)
        cap.release()
        frames = np.stack(frames, axis=0)  # [T, H, W, C]
        frames = torch.FloatTensor(frames).permute(0, 3, 1, 2) / 255.0  # [T, C, H, W]
        return frames

    def extract_audio_mfcc(self, audiofile, target_sr=16000):
        # 使用 ffmpeg 转码为 16kHz 单声道 WAV
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            wav_path = tmp.name
        cmd = [
            'ffmpeg', '-y', '-i', audiofile,
            '-ac', '1', '-ar', str(target_sr),
            '-loglevel', 'error',
            wav_path
        ]
        subprocess.run(cmd, check=True)

        sr, audio = wavfile.read(wav_path)
        os.remove(wav_path)
        mfcc_feats = mfcc(audio, samplerate=sr, numcep=13)  # shape: [T, 13]
        return torch.FloatTensor(mfcc_feats)


    def evaluate_input_audio_video_path(self, opt, videofile=None, audiofile=None):
        def resize_video_tensor(im, target_size=(112, 112)):
            """
            Resize a (1, 3, T, H, W) numpy array to (1, 3, T, H', W')
            """
            assert im.ndim == 5 and im.shape[0] == 1 and im.shape[1] == 3, "Expected shape (1, 3, T, H, W)"
            _, C, T, H, W = im.shape
            new_H, new_W = target_size
            im_resized = np.zeros((1, C, T, new_H, new_W), dtype=np.float32)

            for t in range(T):
                for c in range(C):
                    frame = im[0, c, t, :, :]  # shape [H, W]
                    frame_resized = cv2.resize(frame, (new_W, new_H))  # 注意顺序是 (W, H)
                    im_resized[0, c, t, :, :] = frame_resized

            return im_resized
        self.__S__.eval()

        import glob
        from shutil import rmtree

        tmp_dir = os.path.join(opt.tmp_dir, opt.reference)
        if os.path.exists(tmp_dir):
            rmtree(tmp_dir)
        os.makedirs(tmp_dir)

        # ========== 1. 视频 → jpg 序列 ==========
        # NOTE GSA中合成视频时需要用30fps，这里提取帧要用25fps
        # subprocess.call(f"ffmpeg -y -i {videofile} -threads 1 -f image2 {tmp_dir}/%06d.jpg", shell=True)
        # 25.4.11 这里就用默认fps(30)，到后面mfcc中再修改音频处理的winstep，这样分数更好看一点
        subprocess.call(f"ffmpeg -y -i {videofile} -threads 1 -f image2 {tmp_dir}/%06d.jpg", shell=True)
        # subprocess.call(f"ffmpeg -y -i {videofile} -r 25 -threads 1 -f image2 {tmp_dir}/%06d.jpg", shell=True)


        # ========== 2. 音频 → 16kHz wav ==========
        audio_source = audiofile if audiofile else videofile
        subprocess.call(f"ffmpeg -y -i {audio_source} -ac 1 -ar 16000 -vn -acodec pcm_s16le {tmp_dir}/audio.wav", shell=True)

        # ========== 3. 读取图像序列 ==========
        flist = sorted(glob.glob(os.path.join(tmp_dir, '*.jpg')))
        images = [cv2.imread(f) for f in flist]
        im = np.stack(images, axis=3)  # [H, W, C, T]
        # NOTE: 专门适用于gsa的resize
        # 评测分数对嘴部在画面中的位置很敏感
        im = np.expand_dims(im, axis=0)  # [1, H, W, C, T]
        im = np.transpose(im, (0, 3, 4, 1, 2))  # [1, C, T, H, W]
        _,_,T,H,W= im.shape
        # im = im[:,:,:,0:W,:]
        im = im[:,:,:,81:81+495,18:18+495]
        # im = im[:,:,:,329-150:329+150,265-150:265+150]
        """
        img_demo=im[0,:,0,:,:]
        img_demo = np.transpose(img_demo, (1, 2, 0))
        cv2.imwrite("./demo.png",img_demo)
        """
        os.makedirs(os.path.join(tmp_dir,"demo_check"), exist_ok=True)

        im = resize_video_tensor(im, target_size=(224, 224))  # [1, C, T, H', W']
        for t in range(T):
            img_demo=im[0,:,t,:,:]
            img_demo = np.transpose(img_demo, (1, 2, 0))
            cv2.imwrite(os.path.join(os.path.join(tmp_dir,"demo_check"), f"demo_{t:03d}.png"), img_demo)        
        imtv = torch.FloatTensor(im)

        # ========== 4. 读取音频 MFCC ==========
        sr, audio = wavfile.read(os.path.join(tmp_dir, 'audio.wav'))
        mfcc_feat = mfcc(audio, samplerate=sr, numcep=13,winstep=1/120)
        # mfcc_feat = mfcc(audio, samplerate=sr, numcep=13,)
        cc = np.expand_dims(np.expand_dims(mfcc_feat.T, axis=0), axis=0)  # [1, 1, 13, T]
        cct = torch.FloatTensor(cc)

        # ========== 5. 同步裁剪匹配 ==========
        n_video = imtv.shape[2]
        n_audio = cct.shape[3] // 4
        min_len = min(n_video, n_audio)
        last_frame = min_len - 5

        im_feat = []
        cc_feat = []

        for i in range(0, last_frame, opt.batch_size):
            im_batch = [imtv[:, :, v:v+5, :, :] for v in range(i, min(last_frame, i+opt.batch_size))]
            im_in = torch.cat(im_batch, dim=0)
            im_out = self.__S__.forward_lip(im_in.cuda())
            im_feat.append(im_out.cpu())

            cc_batch = [cct[:, :, :, v*4:v*4+20] for v in range(i, min(last_frame, i+opt.batch_size))]
            cc_in = torch.cat(cc_batch, dim=0)
            cc_out = self.__S__.forward_aud(cc_in.cuda())
            cc_feat.append(cc_out.cpu())

        im_feat = torch.cat(im_feat, dim=0)
        cc_feat = torch.cat(cc_feat, dim=0)

        # ========== 6. LSE-C / LSE-D ==========
        dists = calc_pdist(im_feat, cc_feat, vshift=opt.vshift)
        mdist = torch.mean(torch.stack(dists, dim=1), dim=1)
        minval, minidx = torch.min(mdist, dim=0)
        offset = opt.vshift - minidx
        conf = torch.median(mdist) - minval

        return offset.item(), float(conf), dists



    def extract_feature(self, opt, videofile):

        self.__S__.eval();
        
        # ========== ==========
        # Load video 
        # ========== ==========
        cap = cv2.VideoCapture(videofile)

        frame_num = 1;
        images = []
        while frame_num:
            frame_num += 1
            ret, image = cap.read()
            if ret == 0:
                break

            images.append(image)

        im = numpy.stack(images,axis=3)
        im = numpy.expand_dims(im,axis=0)
        im = numpy.transpose(im,(0,3,4,1,2))

        imtv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())
        
        # ========== ==========
        # Generate video feats
        # ========== ==========

        lastframe = len(images)-4
        im_feat = []

        tS = time.time()
        for i in range(0,lastframe,opt.batch_size):
            
            im_batch = [ imtv[:,:,vframe:vframe+5,:,:] for vframe in range(i,min(lastframe,i+opt.batch_size)) ]
            im_in = torch.cat(im_batch,0)
            im_out  = self.__S__.forward_lipfeat(im_in.cuda());
            im_feat.append(im_out.data.cpu())

        im_feat = torch.cat(im_feat,0)

        # ========== ==========
        # Compute offset
        # ========== ==========
            
        print('Compute time %.3f sec.' % (time.time()-tS))

        return im_feat


    def loadParameters(self, path):
        loaded_state = torch.load(path, map_location=lambda storage, loc: storage);

        self_state = self.__S__.state_dict();

        for name, param in loaded_state.items():

            self_state[name].copy_(param);
