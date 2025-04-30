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
from flame_model.flame import *
from utils.obj_io import read_obj
# sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

"""
date:2025.03.21
note:
尝试加载flame model中的face parsing信息
"""

""" Available part masks from the FLAME model: 
        face, neck, scalp, boundary, right_eyeball, left_eyeball, 
        right_ear, left_ear, forehead, eye_region, nose, lips,
        right_eye_region, left_eye_region.
"""

def save_obj_with_vertex_colors(filename, vertices, faces, vert_colors):
    """
    直接在 OBJ 文件中存储带颜色的顶点信息
    :param filename: 保存的文件路径
    :param vertices: 顶点坐标，形状 [N, 3]
    :param faces: 面片索引，形状 [M, 3]，索引从 0 开始
    :param vert_colors: 每个顶点的 RGB 颜色，形状 [N, 3]，值范围 [0, 255]
    """
    if np.min(faces) == 0:
        faces = faces + 1  # 确保 OBJ 的索引从 1 开始

    with open(filename, "w") as f:
        f.write("# OBJ file with vertex colors\n\n")

        # 写入顶点 + 颜色
        for i, vertex in enumerate(vertices):
            color = vert_colors[i]  # 获取该顶点的颜色
            f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f} {color[0]} {color[1]} {color[2]}\n")

        f.write("\n")

        # 写入面片
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")

    print(f"✅ OBJ 文件已保存: {filename}")

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


def load_parsing_from_model_test():
    flame_model = FlameHead(shape_params=300,expr_params=100)
    flame_mask  = FlameMask() # 牙齿本来就是靠后的面片和顶点，应该无所谓

    vars_=vars(flame_mask.v)
    parts=vars_["_buffers"]
    for part in parts:
        print(part)
        print(parts[part].shape)
    indexs=[]
    len_indexs=[]
    # 语义区间名字
    part_names=[]
    for part in parts:
        indexs.append(torch.max(parts[part]))
        len_indexs.append(len(parts[part]))
        part_names.append(part)
    
    # 检查顶点
    # for iter in range(0,len(part_names)):
    #     v_indices=parts[part_names[iter]]
    #     # print(v_indices)
    #     # sys.exit()
    #     v_colors=torch.zeros([5143,3],dtype=int)+128
    #     v_colors[v_indices,:]=torch.tensor([255,0,0],dtype=int)
    #     v_colors=v_colors.numpy()
    #     mesh_path="test_mesh_1000_vert.obj"
    #     verts,_,faces,_=read_obj(mesh_path)
    #     save_obj_with_vertex_colors(f"face_parsing_check/{part_names[iter]}.obj",verts,faces,v_colors)  
    #     # sys.exit()

    # 取出面片（基于无teeth 及5023点） 检查后，只取出lips对应的面片
    mesh_path="test_mesh_1000_vert.obj"
    verts,_,faces,_=read_obj(mesh_path)    
    valid_faces_mask = np.all((faces >= 0) & (faces <= 5022), axis=1)
    valid_faces = faces[valid_faces_mask]
    print(valid_faces.shape)
    lips_v_indices_np=parts["lips"].numpy()
    contains_lips_v_indices_mask = np.any(np.isin(valid_faces, lips_v_indices_np), axis=1)
    faces_indices = np.where(contains_lips_v_indices_mask)[0]
    print(faces_indices.shape)
    face_colors = np.full((len(faces), 3), 128, dtype=np.uint8)  # 默认灰色
    face_colors[faces_indices,] = [0,0,0]  # 选中的面片设为黑色
    np.savetxt("face_parsing_check/face_seg/lip.txt", faces_indices, fmt="%d")
    # save_obj_with_colors(
    #     f"face_parsing_check/face_seg/lips.obj", verts, faces, face_colors
    # )

    return

def process_txt():
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
    mesh_path="test_mesh_1000_vert.obj"
    verts,_,faces,_=read_obj(mesh_path)

    left_check=np.loadtxt("face_parsing_check/face_seg/tmp/left_check.txt",dtype=int)
    left_and_right_check=np.loadtxt("face_parsing_check/face_seg/tmp/left_and_right_check.txt",dtype=int)
    left_and_right_check_and_upper_chin=np.loadtxt(
        "face_parsing_check/face_seg/tmp/left_and_right_check_and_upper_chin.txt",dtype=int
    )
    left_and_right_check_and_upper_and_lower_chin=np.loadtxt(
        "face_parsing_check/face_seg/tmp/left_and_right_check_and_upper_and_lower_chin.txt",dtype=int
    )
    left_and_right_check_and_upper_and_lower_chin_and_lower_nose=np.loadtxt(
        "face_parsing_check/face_seg/tmp/left_and_right_check_and_upper_and_lower_chin_and_lower_nose.txt",dtype=int
    )

    right_check=np.setdiff1d(left_and_right_check,left_check)
    upper_chin=np.setdiff1d(left_and_right_check_and_upper_chin,left_and_right_check)
    lower_chin=np.setdiff1d(left_and_right_check_and_upper_and_lower_chin,left_and_right_check_and_upper_chin)
    lower_nose=np.setdiff1d(left_and_right_check_and_upper_and_lower_chin_and_lower_nose,left_and_right_check_and_upper_and_lower_chin)
    face_colors = np.full((len(faces), 3), 128, dtype=np.uint8)  # 默认灰色
    left_check=np.append(left_check,8146)
    face_colors[left_check,] = [255,255,0]
    face_colors[right_check,] = [255,0,0]  
    face_colors[upper_chin,] = [0,255,0]  
    face_colors[lower_chin,] = [0,0,255]
    face_colors[lower_nose,] = [0,255,255]
    # save_obj_with_colors(
    #     f"face_parsing_check/face_seg/face_seg_test.obj", verts, faces, face_colors
    # )
    np.savetxt("face_parsing_check/face_seg/left_check.txt",left_check,fmt="%d")
    np.savetxt("face_parsing_check/face_seg/right_check.txt",right_check,fmt="%d")
    np.savetxt("face_parsing_check/face_seg/upper_chin.txt",upper_chin,fmt="%d")
    np.savetxt("face_parsing_check/face_seg/lower_chin.txt",lower_chin,fmt="%d")
    np.savetxt("face_parsing_check/face_seg/lower_nose.txt",lower_nose,fmt="%d")



    return

def teeth_faces():
    # 读取flame模型
    flame_model = FlameHead(shape_params=300,expr_params=100,add_teeth=False)
    faces_ori=flame_model.faces.shape[0]
    # print(f"faces_ori:{faces_ori}")
    flame_model.add_teeth()
    faces_teeth=flame_model.faces.shape[0]
    print(f"faces_teeth:{faces_teeth}")
    teeth_faces_index=np.arange(faces_ori,faces_teeth)
    np.savetxt("face_parsing_check/face_seg/teeth.txt",teeth_faces_index,fmt="%d")
    return


def flame_face_parsing_check():
    # 通过可视化检查
    # 1 读取flame模型
    flame_model = FlameHead(shape_params=300,expr_params=100)
    # 2 加载用于可视化的mesh
    mesh_path="test_mesh_1000_vert.obj"
    verts,_,faces,_=read_obj(mesh_path)
    # 3 读取parsing区域对应的面片索引
    lip_faces_index         = np.loadtxt("face_parsing_check/face_seg/lip.txt",dtype=int)
    left_check_faces_index  = np.loadtxt("face_parsing_check/face_seg/left_check.txt",dtype=int)
    right_check_faces_index = np.loadtxt("face_parsing_check/face_seg/right_check.txt",dtype=int)
    upper_chin_faces_index  = np.loadtxt("face_parsing_check/face_seg/upper_chin.txt",dtype=int)
    lower_chin_faces_index  = np.loadtxt("face_parsing_check/face_seg/lower_chin.txt",dtype=int)
    lower_nose_faces_index  = np.loadtxt("face_parsing_check/face_seg/lower_nose.txt",dtype=int)
    teeth_faces_index       = np.loadtxt("face_parsing_check/face_seg/teeth.txt",dtype=int)
    print(f"lip_faces_index:{lip_faces_index.shape}")
    print(f"left_check_faces_index:{left_check_faces_index.shape}")
    print(f"right_check_faces_index:{right_check_faces_index.shape}")
    print(f"upper_chin_faces_index:{upper_chin_faces_index.shape}")
    print(f"lower_chin_faces_index:{lower_chin_faces_index.shape}")
    print(f"lower_nose_faces_index:{lower_nose_faces_index.shape}")
    print(f"teeth_faces_index:{teeth_faces_index.shape}")
    # 4 根据面片索引，给面片上色
    face_colors = np.full((len(faces), 3), 128, dtype=np.uint8)  # 默认灰色
    face_colors[lip_faces_index,] = [255,0,255]
    face_colors[left_check_faces_index,] = [255,255,0]
    face_colors[right_check_faces_index,] = [255,0,0]
    face_colors[upper_chin_faces_index,] = [0,255,0]
    face_colors[lower_chin_faces_index,] = [0,0,255]
    face_colors[lower_nose_faces_index,] = [0,255,255]
    face_colors[teeth_faces_index,] = [255,255,255]
    # 5 保存
    save_obj_with_colors(
        f"face_parsing_check/face_seg/face_parsing_test.obj", verts, faces, face_colors
    )

    return


def main():
    # load_parsing_from_model_test()
    # process_txt()
    flame_face_parsing_check()
    # teeth_faces()
    return
if __name__ == '__main__':
    main()
