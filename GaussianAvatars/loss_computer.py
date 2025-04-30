import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
from PIL import Image
import pickle
import sys
from tqdm import tqdm
from typing import Dict
import matplotlib.pyplot as plt
import pylab as pl
from utils.loss_utils import l1_loss, ssim
from lpipsPyTorch import lpips
from utils.image_utils import psnr, error_map
import glob
import torchvision.transforms as transforms
"""
date:2025.03.24
note:
测试算法/模型的loss，用于论文中的对比实验
格式：
给定两个文件夹，文件夹内是渲染好的图片（如模型渲染结果和gt），
    两个文件夹内图片数量相等，相同名字的图片对应是一组
loss指标：
l1 psnr ssim
注意，基于[0,1]的归一化值域来计算这些loss
l1 loss越小表示图片越相似；
psnr ssim越大表示图片越相似
"""
def load_image_as_tensor(image_path):
    transform = transforms.ToTensor()
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image)  # [C, H, W]
    return tensor.unsqueeze(0)  # ➜ [1, C, H, W]


def loss_computer(
    source_path:str = "",
    target_path:str = "",        
):
    valid_exts = {'.png', '.jpg', '.jpeg'}
    image_files = []
    for filename in sorted(os.listdir(source_path)):
        if os.path.splitext(filename)[-1] in valid_exts:
            image_files.append(filename)
    num_of_images = len(image_files)    
    l1_loss_list=[]
    psnr_list=[]
    ssim_list=[]
    # lpips_list=[]
    for img_iter in tqdm(range(num_of_images),desc="compting loss for images"):
        source_img_path = os.path.join(source_path, image_files[img_iter])
        target_img_path = os.path.join(target_path, image_files[img_iter])
        source_tensor = load_image_as_tensor(source_img_path)
        target_tensor = load_image_as_tensor(target_img_path)
        l1_loss_list.append(l1_loss(source_tensor, target_tensor).item())
        psnr_list.append(psnr(source_tensor, target_tensor).item())
        ssim_list.append(ssim(source_tensor, target_tensor).item())
    print(f"l1_loss: {np.mean(l1_loss_list):.4f}")
    print(f"psnr   : {np.mean(psnr_list):.4f}")
    print(f"ssim   : {np.mean(ssim_list):.4f}")
    return
def main():
    loss_computer(
        source_path = "output/UNION10EMOEXP_306_eval_600k/test_6/ours_600000/renders",
        target_path = "output/UNION10EMOEXP_306_eval_600k/test_6/ours_600000/gt",
    )
    return
if __name__ == '__main__':
    main()
