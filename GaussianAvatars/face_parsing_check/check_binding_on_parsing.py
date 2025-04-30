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
current_path = os.getcwd()
sys.path.insert(0, current_path)
from utils.obj_io import *
from scene.flame_gaussian_model import FlameGaussianModel
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
"""
date:2025.03.21
note:
以subject=306为例，检测每个分区，平均每个面片有多少个gaussian
"""
def load_flame_faces_parsing_index():
    flame_faces_parsing_index_dict=defaultdict(torch.Tensor)
    lip_faces_index         = np.loadtxt("face_parsing_check/face_seg/lip.txt",dtype=int)
    left_check_faces_index  = np.loadtxt("face_parsing_check/face_seg/left_check.txt",dtype=int)
    right_check_faces_index = np.loadtxt("face_parsing_check/face_seg/right_check.txt",dtype=int)
    upper_chin_faces_index  = np.loadtxt("face_parsing_check/face_seg/upper_chin.txt",dtype=int)
    lower_chin_faces_index  = np.loadtxt("face_parsing_check/face_seg/lower_chin.txt",dtype=int)
    lower_nose_faces_index  = np.loadtxt("face_parsing_check/face_seg/lower_nose.txt",dtype=int)
    teeth_faces_index       = np.loadtxt("face_parsing_check/face_seg/teeth.txt",dtype=int)
    flame_faces_parsing_index_dict["lip_faces_index"]=torch.from_numpy(lip_faces_index)
    flame_faces_parsing_index_dict["left_check_faces_index"]=torch.from_numpy(left_check_faces_index)
    flame_faces_parsing_index_dict["right_check_faces_index"]=torch.from_numpy(right_check_faces_index)
    flame_faces_parsing_index_dict["upper_chin_faces_index"]=torch.from_numpy(upper_chin_faces_index)
    flame_faces_parsing_index_dict["lower_chin_faces_index"]=torch.from_numpy(lower_chin_faces_index)
    flame_faces_parsing_index_dict["lower_nose_faces_index"]=torch.from_numpy(lower_nose_faces_index)
    flame_faces_parsing_index_dict["teeth_faces_index"]=torch.from_numpy(teeth_faces_index)
    return flame_faces_parsing_index_dict



def check_binding_on_parsing(
    ply_path      :str = "output/UNION10EMOEXP_306_eval_600k/point_cloud/iteration_600000/point_cloud.ply",
):
    # 1 加载gaussian
    gaussians=FlameGaussianModel(
        sh_degree=3,n_shape=300,n_expr=100,
    )
    gaussians.load_ply(ply_path,has_target=False)
    # print(gaussians.binding.shape)
    # print(gaussians.binding)
    binding=gaussians.binding.cpu()
    # 2 加载face_parsing_index
    flame_faces_parsing_index_dict=load_flame_faces_parsing_index()
    # 3 统计每个分区的gaussian数量
    for key in flame_faces_parsing_index_dict:
        face_parsing_name=key
        faces_index=flame_faces_parsing_index_dict[key]
        # 通过一个1d tensor
        # 统计face_parsing_name中的每个面片，被绑定了多少个gaussian
        face_counts = torch.zeros_like(faces_index, dtype=torch.int64)
        # 朴素写法避免出错
        for i, face in enumerate(faces_index):
            face_counts[i] = (binding == face).sum()  # 计算 binding 中有多少元素等于当前 face 的值  
        mean_count = face_counts.float().mean().item()  # 平均值
        max_count = face_counts.max().item()  # 最大值
        min_count = face_counts.min().item()  # 最小值  
        print(f"分区{face_parsing_name},每个面片被绑定高斯数量的平均值:{mean_count:.3f},最大值:{max_count},最小值:{min_count}")           
    return


def save_obj_with_colors(filename, vertices, faces, face_colors):
    """
    保存带有面片颜色的 .obj 文件
    faces : np [N_face,3]  (索引从 1 开始)
    face_colors : np [N_face, 3] (RGB 颜色)
    """
    import os

    if np.min(faces) == 0:
        faces = faces + 1  # 确保索引从 1 开始

    # 生成 .mtl 文件
    mtl_filename = filename.replace(".obj", ".mtl")
    with open(mtl_filename, "w") as mtl_file:
        mtl_file.write("# Material file\n")
        unique_colors = np.unique(face_colors, axis=0)  # 找到唯一的颜色
        color_names = {}

        for i, color in enumerate(unique_colors):
            color_name = f"color_{i}"
            color_names[tuple(color)] = color_name
            mtl_file.write(f"newmtl {color_name}\n")
            mtl_file.write(f"Kd {color[0] / 255.0} {color[1] / 255.0} {color[2] / 255.0}\n\n")  # 归一化

    # 生成 .obj 文件
    with open(filename, "w") as f:
        f.write(f"mtllib {os.path.basename(mtl_filename)}\n")

        # 写入顶点
        for vertex in vertices:
            f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")

        # 写入面片及其材质
        for i, face in enumerate(faces):
            color = tuple(face_colors[i])
            material = color_names[color]
            f.write(f"usemtl {material}\n")
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")


def check_binding_on_full_head(
    ply_path      :str = "output/UNION10EMOEXP_306_eval_600k/point_cloud/iteration_600000/point_cloud.ply",
    mesh_path="test_mesh_1000_vert.obj"
):
    # date:2025.03.21
    # note:
    # 检测整个头部的绑定情况
    # 1 加载gaussian
    gaussians=FlameGaussianModel(
        sh_degree=3,n_shape=300,n_expr=100,
    )
    gaussians.load_ply(ply_path,has_target=False)
    binding=gaussians.binding.cpu()
    # 2 加载用于可视化的mesh
    verts,_,faces,_=read_obj(mesh_path)
    # 3 统计每个面片绑定的gaussian数量
    binding_counts_for_each_face=torch.zeros(len(faces),dtype=torch.int64)
    for i, face in enumerate(faces):
        binding_counts_for_each_face[i] = (binding == i).sum()
    # 3.0 打印、保存查看
    print(torch.sum(binding_counts_for_each_face))
    print(binding_counts_for_each_face.shape)
    with open("face_parsing_check/tensor_output.txt", "w") as f:
        for item in binding_counts_for_each_face:
            f.write(f"{item.item()}\n")
    print("1",binding_counts_for_each_face[3396])
    print("2",binding_counts_for_each_face[[8452,3385,8447,3381,8443,8456,8458,3310,8371,3309,8459,3397]])
    sys.exit()
    # 3.1 存直方图
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
    matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题 
    plt.hist(binding_counts_for_each_face.to(int).numpy(), bins=500, edgecolor='black')  # bins 可调节条数
    top_values, top_indices = torch.topk(binding_counts_for_each_face, k=10)
    
    plt.title("每个面片被绑定的高斯数量")
    plt.xlabel("面片被绑定的高斯个数")
    plt.ylabel("被绑定高斯个数相同的面片个数")
    plt.grid(True)
    # plt.show()
    plt.savefig("face_parsing_check/tensor_histogram.png")
    # 3.2 相关查看
    # 计算大于等于5的面片数量
    # greater_than_5=(binding_counts_for_each_face>=5).sum().item()
    # print(f"大于等于5的面片数量:{greater_than_5},{greater_than_5/len(faces):.3f}")

    # # 3.3 统计每个绑定数出现的次数
    # counts = {}
    # for num in binding_counts_for_each_face.tolist():
    #     if num in counts:
    #         counts[num] += 1
    #     else:
    #         counts[num] = 1
    # for key in sorted(counts.keys(), reverse=True):
    #     print(f"{key}: {counts[key]}")
    # sys.exit()


    # 4 设置归一化颜色：
    # 点数最少: [0,0,255](蓝); 点数最多: [255,255,0](黄)
    # 4.1 归一化
    binding_counts_for_each_face=binding_counts_for_each_face.float()
    binding_counts_for_each_face_normalize=(binding_counts_for_each_face-torch.min(binding_counts_for_each_face))/(torch.max(binding_counts_for_each_face)-torch.min(binding_counts_for_each_face))
    # 4.2 设置颜色
    colors=torch.stack(
        (
            binding_counts_for_each_face_normalize,
            binding_counts_for_each_face_normalize,
            1-binding_counts_for_each_face_normalize
        )
    )*255
    colors=colors.to(torch.int).numpy().T
    print(colors)
    # 4.3 保存
    save_obj_with_colors(
        f"face_parsing_check/check_binding_result.obj",
        verts,faces,colors
    )
    # 5 高亮大于等于5的面片
    # 5.1 设置颜色
    binding_counts_for_each_face_np=binding_counts_for_each_face.numpy()
    print(binding_counts_for_each_face_np)
    colors=np.zeros((binding_counts_for_each_face_np.shape[0],3))
    for i in range(10144):
        if binding_counts_for_each_face_np[i]==1:
            colors[i]=[128,128,128]
        elif binding_counts_for_each_face_np[i] in [2,3,4]:
            colors[i]=[255,0,0]
        elif binding_counts_for_each_face_np[i] in range(5,11):
            colors[i]=[0,255,0]
        elif binding_counts_for_each_face_np[i] in range(11,51):
            colors[i]=[0,0,255]
        else:
            colors[i]=[255,255,255]

    # 5.2 保存
    save_obj_with_colors(
        f"face_parsing_check/check_binding_result_highlight_5.obj",
        verts,faces,colors
    )

    return


def main():
    # check_binding_on_parsing()
    check_binding_on_full_head()
    return
if __name__ == '__main__':
    main()
