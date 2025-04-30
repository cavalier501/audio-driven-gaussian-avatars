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
from plyfile import PlyData, PlyElement
import flame_model.flame as flame_model_pkg
from flame_model.flame import FlameHead
import numpy
from utils.obj_io import *
from scene.flame_gaussian_model import FlameGaussianModel
from scene.gaussian_model import GaussianModel

"""
date:2025.03.09
note:
测试gaussians的读取
"""


def gaussian_io_test():
    # 参考scene.flame_gaussian_model.FlameGaussianModel.load_ply
    # 基于flame的不是scene.gaussian_model.GaussianModel.load_ply
    PLY_path="/home/zh/master_thesis_supplementary/GaussianAvatars/output/UNION10EMOEXP_306_eval_600k/point_cloud/iteration_600000/point_cloud.ply"
    plydata = PlyData.read(PLY_path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
    print(xyz.shape)

    return

def flame_io_test():
    # 参考scene.gaussian_model.GaussianModel.load_meshes
    flame_path="/home/zh/master_thesis_supplementary/GaussianAvatars/output/UNION10EMOEXP_306_eval_600k/point_cloud/iteration_600000/flame_param.npz"
    data=np.load(flame_path)
    # print(data.files)
    print(data["shape"].shape)
    print(data["expr"].shape)
    return


def avarage_face_test():
    # date:2025.03.09
    # note:
    # 写一个生成平均脸mesh帧序列的代码用于测试
    flame_path="/home/zh/master_thesis_supplementary/GaussianAvatars/output/UNION10EMOEXP_306_eval_600k/point_cloud/iteration_600000/flame_param.npz"
    data=np.load(flame_path)
    # print(data.files)
    frames=data["expr"].shape[0]
    # print(frames)
    # print(data["shape"].shape)
    flame_model=FlameHead(shape_params=300,expr_params=100,add_teeth=True).cuda()
    # 平均脸
    # print(flame_model.v_template.shape)
    average_face=flame_model.v_template.clone().unsqueeze(0).expand(frames,-1,-1)
    print(average_face.shape)
    torch.save(
        average_face.cpu(),
        "output/UNION10EMOEXP_306_eval_600k/point_cloud/iteration_600000/average_face_meshes.pt"
    )
    # 有身份 无表情人脸
    shape=torch.tensor(data["shape"]).expand(frames,-1)
    expr=torch.zeros_like(torch.tensor(data["expr"]))
    betas = torch.cat([shape, expr], dim=1).cuda()
    print(betas.shape)
    template_vertices = flame_model.v_template.unsqueeze(0).expand(frames, -1, -1)
    
    v_shaped = template_vertices + flame_model_pkg.blend_shapes(betas, flame_model.shapedirs)
    print(v_shaped.shape)
    torch.save(
        v_shaped.cpu(),
        "output/UNION10EMOEXP_306_eval_600k/point_cloud/iteration_600000/id_only_face_meshes.pt"
    )    

    return


def ply_check():
    ply_path="/home/zh/master_thesis_supplementary/GaussianAvatars/output/UNION10EMOEXP_306_eval_600k/point_cloud/iteration_600000/point_cloud.ply"
    plydata = PlyData.read(ply_path)
    # print(plydata.elements)
    # for element in plydata.elements:
    #     print(f"Element name: {element.name}")
    #     print("Keys:", [prop.name for prop in element.properties])
    #     print()
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]    
    print(xyz.shape)
    print(xyz)
    print(opacities.shape)
    binding=np.asarray(plydata.elements[0]["binding_0"])
    print(binding.shape)
    print(binding)
    print(max(binding),min(binding))
    return

def check_vert():
    a="/home/zh/master_thesis_supplementary/GaussianAvatars/output/UNION10EMOEXP_306_eval_600k/point_cloud/iteration_600000/average_face_meshes.pt"
    a=torch.load(a)
    print(a.shape)
    return


def flame_vert_add_teeth():
    # date:2025.03.12
    # note:
    # 测试add_teeth对顶点数的影响
    flame_model=FlameHead(shape_params=300,expr_params=100,add_teeth=True)
    # vert_meshes_path="/home/zh/master_thesis_supplementary/UniTalker/flame_vert_test.pt"
    vert_meshes_path="/home/zh/master_thesis_supplementary/UniTalker/gaussian_avatar_flame_vert_test.pt"
    vert_meshes=torch.load(vert_meshes_path)

    vert_meshes=flame_model.add_teeth_to_mesh(vert_meshes)
    print(vert_meshes.shape)
    torch.save(vert_meshes,vert_meshes_path.replace(".pt","_with_teeth.pt"))
    return


def flame_vert_add_teeth_test():
    # model_path="/home/zh/master_thesis_supplementary/UniTalker/flame_vert_test_with_teeth.pt"
    model_path="/home/zh/master_thesis_supplementary/UniTalker/flame_vert_test_with_teeth.pt"
    model=torch.load(model_path)
    is_same = torch.all(model == model[0], dim=(1, 2))
    print(is_same.all().item())
    print(model.shape)
    return


def check_obj():
    # model_path="/home/zh/master_thesis_supplementary/UniTalker/flame_vert_test_with_teeth.pt"
    # model=torch.load(model_path)
    # shape_from_unitalker=model[0].cpu().numpy()
    # save_obj_pointcloud(
    #     "./shape_from_unitalker.obj",
    #     shape_from_unitalker,
    # )
    vert_gaussian_template,_,_,_=read_obj(
        "/home/zh/master_thesis_supplementary/GaussianAvatars/flame_model/assets/flame/head_template_mesh.obj"
    )
    vert_unitalker_flame_template,_,_,_=read_obj(
        "/home/zh/master_thesis_supplementary/UniTalker/resources/obj_template/FLAME_5023_vertices.obj"
    )

    x_mean_vert_gaussian_template=np.mean(vert_gaussian_template[:,0])
    x_mean_vert_unitalker_flame_template=np.mean(vert_unitalker_flame_template[:,0])
    y_mean_vert_gaussian_template=np.mean(vert_gaussian_template[:,1])
    y_mean_vert_unitalker_flame_template=np.mean(vert_unitalker_flame_template[:,1])
    z_mean_vert_gaussian_template=np.mean(vert_gaussian_template[:,2])
    z_mean_vert_unitalker_flame_template=np.mean(vert_unitalker_flame_template[:,2])

    x_range_vert_gaussian_template=np.max(vert_gaussian_template[:,0])-np.min(vert_gaussian_template[:,0])
    x_range_vert_unitalker_flame_template=np.max(vert_unitalker_flame_template[:,0])-np.min(vert_unitalker_flame_template[:,0])
    y_range_vert_gaussian_template=np.max(vert_gaussian_template[:,1])-np.min(vert_gaussian_template[:,1])
    y_range_vert_unitalker_flame_template=np.max(vert_unitalker_flame_template[:,1])-np.min(vert_unitalker_flame_template[:,1])
    z_range_vert_gaussian_template=np.max(vert_gaussian_template[:,2])-np.min(vert_gaussian_template[:,2])
    z_range_vert_unitalker_flame_template=np.max(vert_unitalker_flame_template[:,2])-np.min(vert_unitalker_flame_template[:,2])

    print("mean")
    print("x",x_mean_vert_gaussian_template,x_mean_vert_unitalker_flame_template)
    print("y",y_mean_vert_gaussian_template,y_mean_vert_unitalker_flame_template)
    print("z",z_mean_vert_gaussian_template,z_mean_vert_unitalker_flame_template)
    print("range")
    print("x",x_range_vert_gaussian_template,x_range_vert_unitalker_flame_template)
    print("y",y_range_vert_gaussian_template,y_range_vert_unitalker_flame_template)
    print("z",z_range_vert_gaussian_template,z_range_vert_unitalker_flame_template)
    


    return

def audio_drive_id_template_vert_tmp():
    # date:2025.03.13
    # note:
    # Unitalker得到的flame顶点形变向量（基于gaussian avatar中的平均脸），
    # 加到身份对应的静息（无表情）脸上
    gsa_vert_with_teeth_path="/home/zh/master_thesis_supplementary/UniTalker/gaussian_avatar_flame_vert_test_with_teeth.pt"
    gsa_vert_with_teeth=torch.load(gsa_vert_with_teeth_path)

    flame_path="/home/zh/master_thesis_supplementary/GaussianAvatars/output/UNION10EMOEXP_306_eval_600k/point_cloud/iteration_600000/flame_param.npz"
    data=np.load(flame_path)
    frames=gsa_vert_with_teeth.shape[0]
    # print(frames)
    # print(data["shape"].shape)
    flame_model=FlameHead(shape_params=300,expr_params=100,add_teeth=True).cuda()
    # audio drive 结果
    # 有身份人脸(add teeth后的结果)
    shape=torch.tensor(data["shape"]).expand(frames,-1)
    expr=torch.zeros((frames,data["expr"].shape[-1]))
    betas = torch.cat([shape, expr], dim=1).cuda()
    print(betas.shape)
    deformation_by_id_flame=flame_model_pkg.blend_shapes(betas, flame_model.shapedirs).cpu()
    print(deformation_by_id_flame.shape)
    
    
    v_shaped =gsa_vert_with_teeth  + deformation_by_id_flame
    print(v_shaped.shape)
    torch.save(
        v_shaped,
        gsa_vert_with_teeth_path.replace("gaussian_avatar_","gaussian_avatar_ID_306_"),
    )

    return

def audio_drive_id_template_vert():
    # date:2025.03.13
    # note:
    # Unitalker得到的flame顶点形变向量（基于gaussian avatar中的平均脸），
    # 加到身份对应的静息（无表情）脸上
    deformation_no_teeth=np.load(
        "/home/zh/master_thesis_supplementary/UniTalker/audio_driven_D1_deformation/D1_angry1_f_111.npy"
    ) # [111,5023,3]
    deformation_no_teeth=torch.tensor(deformation_no_teeth).float()
    frames=deformation_no_teeth.shape[0]

    # no teeth ID flame
    flame_path="/home/zh/master_thesis_supplementary/GaussianAvatars/output/UNION10EMOEXP_306_eval_600k/point_cloud/iteration_600000/flame_param.npz"
    data=np.load(flame_path)
    flame_model_no_teeth=FlameHead(shape_params=300,expr_params=100,add_teeth=False)
    shape = torch.tensor(data["shape"]).expand(frames,-1)
    expr  = torch.zeros((frames,data["expr"].shape[-1]))
    betas = torch.cat([shape, expr], dim=1)
    deformation_by_id_flame=flame_model_pkg.blend_shapes(betas, flame_model_no_teeth.shapedirs).cpu()
    # print(deformation_by_id_flame.shape) # [111,5023,3]

    gsa_average_face=flame_model_no_teeth.v_template.unsqueeze(0).expand(frames, -1, -1)
    # print(gsa_average_face.shape) # [111,5023,3]
    gsa_vert_id_5023=gsa_average_face+deformation_by_id_flame
    torch.save(
        gsa_vert_id_5023,
        "/home/zh/master_thesis_supplementary/UniTalker/gsa_vert_id_5023.pt"
    )
    # with teeth ID flame
    flame_model_with_teeth=FlameHead(shape_params=300,expr_params=100,add_teeth=True)
    gsa_vert_id_5143=flame_model_with_teeth.add_teeth_to_mesh(gsa_vert_id_5023)
    torch.save(
        gsa_vert_id_5143,
        "/home/zh/master_thesis_supplementary/UniTalker/gsa_vert_id_5143.pt"
    )
    # with ID flame, with audio deformation, with teeth
    gsa_vert_id_audio_5023=gsa_vert_id_5023+deformation_no_teeth
    gsa_vert_id_audio_5143=flame_model_with_teeth.add_teeth_to_mesh(gsa_vert_id_audio_5023)
    torch.save(
        gsa_vert_id_audio_5143,
        "/home/zh/master_thesis_supplementary/UniTalker/gsa_vert_id_audio_5143.pt"
    )

    # gsa_shape_no_audio=

    return

def check_template_mesh():
    # date:2025.03.14
    # note:
    # 检查一下gsa和Unitalker中使用的flame 平均脸模板是否一致
    # 结论:
    # gsa中有“平均脸”
    # Unitalker中没有“平均脸”的概念，其中的template其实是“某种身份的静息脸”
    gsa_flame_model_no_teeth=FlameHead(shape_params=300,expr_params=100,add_teeth=False)
    gsa_template_face=gsa_flame_model_no_teeth.v_template.cpu().numpy()
    print(gsa_template_face.shape)
    unitalker_flame_model_no_teeth=np.load(
        "/home/zh/master_thesis_supplementary/UniTalker/unitalker_data_release_V1/D1_vocaset/id_template.npy"
    )

    print(unitalker_flame_model_no_teeth.shape)
    return

def check_xyz():
    # date:2025.03.18
    # note:
    # 检查PLY文件中的xyz元素是不是高斯的global中心位置
    ply_path="/home/zh/master_thesis_supplementary/GaussianAvatars/output/UNION10EMOEXP_306_eval_600k/point_cloud/iteration_600000/point_cloud.ply"
    plydata = PlyData.read(ply_path)

    gs_model=GaussianModel(sh_degree=3)
    gs_model.load_ply(ply_path)

    # xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
    #                 np.asarray(plydata.elements[0]["y"]),
    #                 np.asarray(plydata.elements[0]["z"])),  axis=1)
    # opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]  
    # print(xyz.shape) 
    # print(opacities.shape)
    xyz=gs_model._xyz
    print(xyz)
    print(xyz.shape)
    save_obj_pointcloud(
        "./check_xyz_2.obj",
        xyz,
    )
    return

def check_face_parsing():
    # date:2025.03.20
    # note:
    # face parsing check 基于面片
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
    # 1 读txt
    mouth_faces=np.loadtxt("face_parsing_check/selected_faces_mouth.txt",dtype=int)
    # 2 设置颜色
    face_colors = np.full((len(faces), 3), 128, dtype=np.uint8)  # 默认灰色
    face_colors[mouth_faces,] = [0,0,0]  # 选中的面片设为黑色
    save_obj_with_colors("colored_mesh.obj", verts, faces, face_colors)
    return


def check_flame_error():
    from collections import defaultdict
    flame_param_path = "data/306_EMO-2_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine/flame_param/00026.npz"
    flame_para=np.load(flame_param_path)
    bs=flame_para["expr"].shape[0]
    flame_param_dict=defaultdict(lambda:None)
    for key in flame_para.files:
        flame_param_dict[key]=torch.tensor(flame_para[key])
    print("shape\n",flame_param_dict["shape"])
    print("expr\n",flame_param_dict["expr"])
    print("rot\n",flame_param_dict["rotation"])
    print("beck\n",flame_param_dict["neck_pose"])
    return



def check_flame_error_2():
    from collections import defaultdict
    # flame_param_path = "data/306_EMO-2_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine/flame_param/00026.npz"
    flame_param_path = "data/UNION10_306_dataset_redivide/canonical_flame_param.npz"
    flame_para=np.load(flame_param_path)
    bs=flame_para["expr"].shape[0]
    flame_param_dict=defaultdict(lambda:None)
    for key in flame_para.files:
        flame_param_dict[key]=torch.tensor(flame_para[key])
    print("shape\n",flame_param_dict["shape"])
    print("expr\n",flame_param_dict["expr"])
    print("rot\n",flame_param_dict["rotation"])
    print("beck\n",flame_param_dict["neck_pose"])
    return

def check_flame_error_3():
    from collections import defaultdict
    # flame_param_path = "data/306_EMO-2_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine/flame_param/00026.npz"
    # flame_param_path = "./output/UNION10EMOEXP_306_dataset_redivide/point_cloud/iteration_600000/flame_param.npz"
    flame_param_path = "output/UNION10EMOEXP_306_dataset_redivide_demo_04012039/point_cloud/best_ckpt/flame_param.npz"
    flame_para=np.load(flame_param_path)
    bs=flame_para["expr"].shape[0]
    flame_param_dict=defaultdict(lambda:None)
    for key in flame_para.files:
        flame_param_dict[key]=torch.tensor(flame_para[key])
    print("shape\n",flame_param_dict["shape"].shape)
    print("expr\n",flame_param_dict["expr"].shape)
    print("rot\n",flame_param_dict["rotation"].shape)
    print("beck\n",flame_param_dict["neck_pose"].shape)
    return


def main():
    # gaussian_io_test()
    # flame_io_test()
    # avarage_face_test()
    # ply_check()
    # check_vert()
    # flame_vert_add_teeth()
    # flame_vert_add_teeth_test()
    # check_obj()
    # audio_drive_id_template_vert()
    # gsa_check()
    # check_template_mesh()
    # check_xyz()
    # check_face_parsing()
    check_flame_error_3()
    return
if __name__ == '__main__':
    main()
