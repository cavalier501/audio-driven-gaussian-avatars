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
# import matplotlib.pyplot as plt
# import pylab as pl



def test_io():
    demo_npz_path="/home/zh/master_thesis_supplementary/UniTalker/test_results/demo.npz"
    demo_npz=np.load(demo_npz_path, allow_pickle=True)
    # 查看键值
    print(demo_npz.files)
    print(type(demo_npz["angry1.wav"]))
    data_dict=demo_npz["angry1.wav"].item()
    for key in data_dict.keys():
        print(key)
        print(data_dict[key].shape)
    return

def test_zh():
    # demo_npz_path="/home/zh/master_thesis_supplementary/UniTalker/test_results/demo.npz"
    demo_npz_path="/home/zh/master_thesis_supplementary/UniTalker/test_results/demo_check_flame.npz"
    demo_npz=np.load(demo_npz_path, allow_pickle=True)
    data_dict=demo_npz["angry1.wav"].item()
    flame_vert=data_dict["D1"]
    print(type(flame_vert))
    # np.save("./flame_vert_test.npy", flame_vert)
    torch.save(flame_vert, "./gaussian_avatar_flame_vert_test.pt")
    return

def replace_flame_from_unitalker_to_gsa():
    # date:2025.03.13
    # note:
    # Unitalker和gaussian avatar中的平均脸形状不太一样
    # 修改一下unitalker_data_release_V1/D1_vocaset/id_template.npy中的平均脸
    template_path="unitalker_data_release_V1/D1_vocaset/id_template.npy"
    template=np.load(template_path)
    print(template.shape)
    model=torch.from_numpy(template)
    is_same = torch.all(model == model[0], dim=(1, 2))
    print(is_same)
    return


def main():
    # test_io()
    # test_zh()
    replace_flame_from_unitalker_to_gsa()
    return
if __name__ == '__main__':
    main()
