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
import flame_model.flame as flame_model_pkg
from flame_model.flame import FlameHead
from flame_model.flame_advanced import FlameHead_advanced
from collections import defaultdict
from utils.obj_io import *
from scene.flame_gaussian_model import FlameGaussianModel
from flame_model.lbs import lbs, vertices2landmarks, blend_shapes, vertices2joints

"""
date:2025.03.14
note:
与几何顶点处理相关的工具函数
如：将audio-driven deformation添加到flame拓扑上，并add teeth
"""

all_audio_deformation_files=['D1_ted1_f_643', 'D1_malaya_f_302', 
                             'D1_sad_f_127', 'D1_bazongname_f_672', 
                             'D1_disgust_f_104', 'D1_angry2_f_120', 
                             'D1_bazongnew_f_432', 'D1_ted2_f_1076', 
                             'D1_fearful_f_113', 'D1_can_you_feel_the_love_tonight_clip_f_866', 
                             'D1_happy_f_116', 'D1_angry1_f_111']



def ID_audio_deformation_convert(
    audio_deformation_path:str,
    flame_path:str,
    add_teeth_flag:bool,
    save_path:str,
):
    # date:2025.03.14
    # note:
    # 1.flame template(平均脸)
    # 2.(+) ID deformation(flame中的身份基) [bs,5023,3] 目前通过gsa中的flame path传入
    # 3.(+) audio-driven deformation [bs,5023,3]
    # 4.(+) add_teeth [bs,5143,3]
    # update:2025.03.17
    # 3.14的版本有点问题，只考虑了flame参数中的id-shape基，漏了static_offset
    # 恢复出来的有点问题（尤其是头发处）
    
    # 0.相关加载
    flame_model_no_teeth   = FlameHead(shape_params=300,expr_params=100,add_teeth=False)
    flame_model_with_teeth = FlameHead(shape_params=300,expr_params=100,add_teeth=True)
    if audio_deformation_path is not None:
        audio_deformation_no_teeth = torch.from_numpy(np.load(audio_deformation_path)).float()
        frames = audio_deformation_no_teeth.shape[0]
    else:
        frames = 1
        
    # 1.平均脸
    gsa_average_face=flame_model_no_teeth.v_template.unsqueeze(0).expand(frames, -1, -1)
    # 2.身份基
    if flame_path is not None:
        flame_data=np.load(flame_path)
        shape = torch.tensor(flame_data["shape"]).expand(frames,-1)
        expr  = torch.zeros((frames,flame_data["expr"].shape[-1]))
    else:
        shape = torch.zeros((frames,300))
        expr  = torch.zeros((frames,100))
    gsa_vert_id_5023,verts_cano=flame_model_no_teeth(
        shape,
        expr,
        torch.zeros_like(torch.tensor(flame_data["rotation"][0]).unsqueeze(0).expand(frames,-1)),
        torch.zeros_like(torch.tensor(flame_data["neck_pose"][0]).unsqueeze(0).expand(frames,-1)),
        torch.zeros_like(torch.tensor(flame_data["jaw_pose"][0]).unsqueeze(0).expand(frames,-1)),
        torch.zeros_like(torch.tensor(flame_data["eyes_pose"][0]).unsqueeze(0).expand(frames,-1)),
        torch.zeros_like(torch.tensor(flame_data["translation"][0]).unsqueeze(0).expand(frames,-1)),
        zero_centered_at_root_node=False,
        return_landmarks=False,
        return_verts_cano=True,
        static_offset=torch.zeros((frames,5023,3)), # 注意，flame_data读出来的结果是基于5143点（add_teeth）的
        dynamic_offset=torch.zeros((frames,5023,3)), # flame_data中的dynamic_offset都是0
    )

    # 3.audio-driven deformation
    if audio_deformation_path is not None:
        gsa_vert_id_5023+=audio_deformation_no_teeth*1
    else:
        pass
    # 4.add_teeth
    if add_teeth_flag:
        gsa_vert_id_5143 = flame_model_with_teeth.add_teeth_to_mesh(gsa_vert_id_5023)
        if flame_path is not None:
            gsa_vert_id_5143+=torch.tensor(flame_data["static_offset"]).expand(frames,-1,-1)
            gsa_vert_id_5143+=torch.tensor(flame_data["dynamic_offset"][0]).unsqueeze(0).expand(frames,-1,-1)
    else:
        pass

    # 5.保存
    if save_path is not None:
        if add_teeth_flag:
            torch.save(gsa_vert_id_5143,save_path)
        else:
            torch.save(gsa_vert_id_5023,save_path)

    return gsa_vert_id_5143 if add_teeth_flag else gsa_vert_id_5023


def ID_audio_deformation_convert_v1(
    audio_deformation_path:str,
    flame_path:str,
    add_teeth_flag:bool,
    save_path:str,
):
    # date:2025.04.18
    # note:
    # 为了做消融实验，重新生成一份“基于插值”的add teeth结果
    # ID_audio_deformation_convert的add teeth是基于插值的，但pose不太对
    
    
    # 0.相关加载
    flame_v1_add_teeth     = FlameHead(shape_params=300,expr_params=100,add_teeth=True)
    flame_model_no_teeth   = FlameHead_advanced(shape_params=300,expr_params=100,add_teeth=False)
    flame_model_with_teeth = FlameHead_advanced(shape_params=300,expr_params=100,add_teeth=True)
    if audio_deformation_path is not None:
        audio_deformation_no_teeth = torch.from_numpy(np.load(audio_deformation_path)).float()
        frames = audio_deformation_no_teeth.shape[0]
    else:
        frames = 1
        
    # 1.平均脸
    gsa_average_face=flame_model_no_teeth.v_template.unsqueeze(0).expand(frames, -1, -1)
    # 2.身份基
    if flame_path is not None:
        flame_data=np.load(flame_path)
        shape = torch.tensor(flame_data["shape"]).expand(frames,-1)
        expr  = torch.zeros((frames,flame_data["expr"].shape[-1]))
    else:
        shape = torch.zeros((frames,300))
        expr  = torch.zeros((frames,100))
    gsa_vert_id_5023,verts_cano=flame_model_no_teeth(
        shape,
        expr,
        torch.zeros_like(torch.tensor(flame_data["rotation"][0]).unsqueeze(0).expand(frames,-1)),
        torch.zeros_like(torch.tensor(flame_data["neck_pose"][0]).unsqueeze(0).expand(frames,-1)),
        torch.zeros_like(torch.tensor(flame_data["jaw_pose"][0]).unsqueeze(0).expand(frames,-1)),
        torch.zeros_like(torch.tensor(flame_data["eyes_pose"][0]).unsqueeze(0).expand(frames,-1)),
        torch.zeros_like(torch.tensor(flame_data["translation"][0]).unsqueeze(0).expand(frames,-1)),
        zero_centered_at_root_node=False,
        return_landmarks=False,
        return_verts_cano=True,
        static_offset=torch.zeros((frames,5023,3)), # 注意，flame_data读出来的结果是基于5143点（add_teeth）的
        dynamic_offset=torch.zeros((frames,5023,3)), # flame_data中的dynamic_offset都是0
    )

    # 3.audio-driven deformation
    if audio_deformation_path is not None:
        gsa_vert_id_5023+=audio_deformation_no_teeth*1
    else:
        pass
    # 4.add_teeth、static_offset
    if add_teeth_flag:
        gsa_vert_id_5143 = flame_v1_add_teeth.add_teeth_to_mesh(gsa_vert_id_5023)
        if flame_path is not None:
            gsa_vert_id_5143+=torch.tensor(flame_data["static_offset"]).expand(frames,-1,-1)
            gsa_vert_id_5143+=torch.tensor(flame_data["dynamic_offset"][0]).unsqueeze(0).expand(frames,-1,-1)
    else:
        pass
    
    # 5.旋转、平移
    # 从forward中copy
    rotation=torch.tensor(flame_data["rotation"][0]).unsqueeze(0).expand(frames,-1)
    translation=torch.tensor(flame_data["translation"][0]).unsqueeze(0).expand(frames,-1)
    jaw_pose  = torch.tensor(flame_data["jaw_pose"][0]).unsqueeze(0).expand(frames, -1)
    neck_pose = torch.tensor(flame_data["neck_pose"][0]).unsqueeze(0).expand(frames, -1)
    eyes_pose = torch.tensor(flame_data["eyes_pose"][0]).unsqueeze(0).expand(frames, -1)
    full_pose = torch.cat([rotation, neck_pose, jaw_pose, eyes_pose], dim=1)


    gsa_vert_id_5143, J, mat_rot = lbs(
        full_pose,
        gsa_vert_id_5143,
        flame_model_with_teeth.posedirs,
        flame_model_with_teeth.J_regressor,
        flame_model_with_teeth.parents,
        flame_model_with_teeth.lbs_weights,
        dtype=flame_model_with_teeth.dtype,
    )
    gsa_vert_id_5143=gsa_vert_id_5143 + translation.unsqueeze(1)
    

    # if add_teeth_flag:
    #     jaw_pose  = torch.tensor(flame_data["jaw_pose"][0]).unsqueeze(0).expand(frames, -1)
    #     neck_pose = torch.tensor(flame_data["neck_pose"][0]).unsqueeze(0).expand(frames, -1)
    #     gsa_vert_id_5143 = flame_model_with_teeth.apply_flame_global_transform(
    #         gsa_vert_id_5143,
    #         rotation,
    #         translation,
    #         jaw_pose=jaw_pose,
    #         neck_pose=neck_pose
    #     )
    # else:
    #     jaw_pose  = torch.tensor(flame_data["jaw_pose"][0]).unsqueeze(0).expand(frames, -1)
    #     neck_pose = torch.tensor(flame_data["neck_pose"][0]).unsqueeze(0).expand(frames, -1)
    #     gsa_vert_id_5023 = flame_model_no_teeth.apply_flame_global_transform(
    #         gsa_vert_id_5023,
    #         rotation,
    #         translation,
    #         jaw_pose=jaw_pose,
    #         neck_pose=neck_pose
    #     )

    # 5.保存
    if save_path is not None:
        if add_teeth_flag:
            torch.save(gsa_vert_id_5143,save_path)
        else:
            torch.save(gsa_vert_id_5023,save_path)

    return gsa_vert_id_5143 if add_teeth_flag else gsa_vert_id_5023






def ID_audio_deformation_convert_v2(
    audio_deformation_path:str,
    flame_path:str,
    add_teeth_flag:bool,
    save_path:str,
):
    # date:2025.04.08
    # note:
    # 1.flame template(平均脸)
    # 2.(+) ID deformation(flame中的身份基、static_offset) [bs,5023,3] 
    # 3.(+) audio-driven deformation [bs,5023,3]
    # 4.(+) add_teeth [bs,5143,3]
    # 5.(+) 加RT
    # 默认add_teeth_flag为True
    
    # 0.相关加载
    flame_model_no_teeth   = FlameHead_advanced(shape_params=300,expr_params=100,add_teeth=False)
    flame_model_with_teeth = FlameHead_advanced(shape_params=300,expr_params=100,add_teeth=True)
    if audio_deformation_path is not None:
        audio_deformation_no_teeth = torch.from_numpy(np.load(audio_deformation_path)).float()
        frames = audio_deformation_no_teeth.shape[0]
    else:
        frames = 1
        
    # 1.平均脸
    gsa_average_face=flame_model_no_teeth.v_template.unsqueeze(0).expand(frames, -1, -1)
    # 2.身份基
    if flame_path is not None:
        flame_data=np.load(flame_path)
        shape = torch.tensor(flame_data["shape"]).expand(frames,-1)
        expr  = torch.zeros((frames,flame_data["expr"].shape[-1]))
    else:
        shape = torch.zeros((frames,300))
        expr  = torch.zeros((frames,100))
    gsa_vert_id_5023,verts_cano=flame_model_no_teeth(
        shape,
        expr,
        torch.zeros_like(torch.tensor(flame_data["rotation"][0]).unsqueeze(0).expand(frames,-1)),
        torch.zeros_like(torch.tensor(flame_data["neck_pose"][0]).unsqueeze(0).expand(frames,-1)),
        torch.zeros_like(torch.tensor(flame_data["jaw_pose"][0]).unsqueeze(0).expand(frames,-1)),
        torch.zeros_like(torch.tensor(flame_data["eyes_pose"][0]).unsqueeze(0).expand(frames,-1)),
        torch.zeros_like(torch.tensor(flame_data["translation"][0]).unsqueeze(0).expand(frames,-1)),
        zero_centered_at_root_node=False,
        return_landmarks=False,
        return_verts_cano=True,
        static_offset=torch.zeros((frames,5023,3)), # 注意，flame_data读出来的结果是基于5143点（add_teeth）的
        dynamic_offset=torch.zeros((frames,5023,3)), # flame_data中的dynamic_offset都是0
    )

    # 3.audio-driven deformation
    if audio_deformation_path is not None:
        gsa_vert_id_5023+=audio_deformation_no_teeth*1
    else:
        pass
    # 4.add_teeth、static_offset
    if add_teeth_flag:
        gsa_vert_id_5143 = flame_model_with_teeth.add_teeth_to_mesh(gsa_vert_id_5023)
        if flame_path is not None:
            gsa_vert_id_5143+=torch.tensor(flame_data["static_offset"]).expand(frames,-1,-1)
            gsa_vert_id_5143+=torch.tensor(flame_data["dynamic_offset"][0]).unsqueeze(0).expand(frames,-1,-1)
    else:
        pass
    
    # 5.旋转、平移
    # 从forward中copy
    rotation=torch.tensor(flame_data["rotation"][0]).unsqueeze(0).expand(frames,-1)
    translation=torch.tensor(flame_data["translation"][0]).unsqueeze(0).expand(frames,-1)
    jaw_pose  = torch.tensor(flame_data["jaw_pose"][0]).unsqueeze(0).expand(frames, -1)
    neck_pose = torch.tensor(flame_data["neck_pose"][0]).unsqueeze(0).expand(frames, -1)
    eyes_pose = torch.tensor(flame_data["eyes_pose"][0]).unsqueeze(0).expand(frames, -1)
    full_pose = torch.cat([rotation, neck_pose, jaw_pose, eyes_pose], dim=1)


    gsa_vert_id_5143, J, mat_rot = lbs(
        full_pose,
        gsa_vert_id_5143,
        flame_model_with_teeth.posedirs,
        flame_model_with_teeth.J_regressor,
        flame_model_with_teeth.parents,
        flame_model_with_teeth.lbs_weights,
        dtype=flame_model_with_teeth.dtype,
    )
    gsa_vert_id_5143=gsa_vert_id_5143 + translation.unsqueeze(1)
    

    # if add_teeth_flag:
    #     jaw_pose  = torch.tensor(flame_data["jaw_pose"][0]).unsqueeze(0).expand(frames, -1)
    #     neck_pose = torch.tensor(flame_data["neck_pose"][0]).unsqueeze(0).expand(frames, -1)
    #     gsa_vert_id_5143 = flame_model_with_teeth.apply_flame_global_transform(
    #         gsa_vert_id_5143,
    #         rotation,
    #         translation,
    #         jaw_pose=jaw_pose,
    #         neck_pose=neck_pose
    #     )
    # else:
    #     jaw_pose  = torch.tensor(flame_data["jaw_pose"][0]).unsqueeze(0).expand(frames, -1)
    #     neck_pose = torch.tensor(flame_data["neck_pose"][0]).unsqueeze(0).expand(frames, -1)
    #     gsa_vert_id_5023 = flame_model_no_teeth.apply_flame_global_transform(
    #         gsa_vert_id_5023,
    #         rotation,
    #         translation,
    #         jaw_pose=jaw_pose,
    #         neck_pose=neck_pose
    #     )

    # 5.保存
    if save_path is not None:
        if add_teeth_flag:
            torch.save(gsa_vert_id_5143,save_path)
        else:
            torch.save(gsa_vert_id_5023,save_path)

    return gsa_vert_id_5143 if add_teeth_flag else gsa_vert_id_5023



def flame_para_to_vert(
    flame_path:str,
    add_teeth_flag:bool,
    vert_save_path:str,
    obj_save_path:str,
):
    # date:2025.03.15
    # note:
    # 将flame参数转换为顶点坐标
    # 0.相关加载
    flame_model_no_teeth   = FlameHead(shape_params=300,expr_params=100,add_teeth=False)
    flame_model_with_teeth = FlameHead(shape_params=300,expr_params=100,add_teeth=True)
    flame_param_path = flame_path
    flame_para=np.load(flame_param_path)
    # 1.处理
    # print(flame_para.files)
    bs=flame_para["expr"].shape[0]
    flame_param_dict=defaultdict(lambda:None)
    for key in flame_para.files:
        flame_param_dict[key]=torch.tensor(flame_para[key])
    if add_teeth_flag:
        vert,verts_cano=flame_model_with_teeth(
            flame_param_dict["shape"].unsqueeze(0).expand(bs,-1),
            flame_param_dict["expr"],
            flame_param_dict["rotation"],
            flame_param_dict["neck_pose"],
            flame_param_dict["jaw_pose"],
            flame_param_dict["eyes_pose"],
            flame_param_dict["translation"],
            zero_centered_at_root_node=False,
            return_landmarks=False,
            return_verts_cano=True,
            static_offset=flame_param_dict['static_offset'],
            dynamic_offset=flame_param_dict['dynamic_offset'],
        )
    else:
        vert,verts_cano=flame_model_with_teeth(
            flame_param_dict["shape"].unsqueeze(0).expand(bs,-1),
            flame_param_dict["expr"],
            flame_param_dict["rotation"],
            flame_param_dict["neck_pose"],
            flame_param_dict["jaw_pose"],
            flame_param_dict["eyes_pose"],
            flame_param_dict["translation"],
            zero_centered_at_root_node=False,
            return_landmarks=False,
            return_verts_cano=True,
            static_offset=flame_param_dict['static_offset'],
            dynamic_offset=flame_param_dict['dynamic_offset'],
        )        
    # 2.保存
    if vert_save_path is not None:
        torch.save(vert,vert_save_path)
    print(vert.shape)
    if obj_save_path is not None: # add_teeth
        save_obj(
            obj_save_path,
            vert.cpu().numpy()[0],
            flame_model_with_teeth.faces.cpu().numpy(),
        )
    # save_obj(
    #     "vert_redivide_code_l224_check.obj",
    #     torch.load("vert_redivide_code_l224_check.pt").cpu().numpy()[0],
    #     flame_model_with_teeth.faces.cpu().numpy(),
    # )


    return vert

def save_obj_pointcloud_from_ply_with_flame_path(
    ply_path      :str = "output/UNION10EMOEXP_306_eval_600k/point_cloud/iteration_600000/point_cloud.ply",
    flame_path    :str = "output/UNION10EMOEXP_306_eval_600k/point_cloud/iteration_600000/flame_param.npz",
    timestep      :int = 1000,
    save_pcd_path :str = None,
    save_mesh_path:str = None,
):
    # date:2025.03.20
    # note:
    # 从ply文件中(guassian的结果)读取点云并保存到obj文件中
    # 几何从flame中加载
    gaussians=FlameGaussianModel(
        sh_degree=3,n_shape=300,n_expr=100,
    )
    gaussians.load_ply(ply_path,flame_path=flame_path,has_target=False)
    gaussians.select_mesh_by_timestep(timestep)
    xyz=gaussians.get_xyz
    if save_pcd_path is not None:
        save_obj_pointcloud(
            save_pcd_path,
            xyz,
        )
    gaussians.save_ply("./test_ply_0417.ply")
    gaussians.save_deformed_ply("./test_deformed_ply_0417.ply")
    sys.exit()
    mesh_verts=gaussians.verts[0]
    mesh_faces=gaussians.flame_model.faces
    mesh_verts=mesh_verts.cpu().numpy()
    mesh_faces=mesh_faces.cpu().numpy()
    if save_mesh_path is not None:
        save_obj(
            save_mesh_path,
            mesh_verts,
            mesh_faces,
        )

    return


def save_obj_pointcloud_from_ply_with_vert_path(
    ply_path      :str = "output/UNION10EMOEXP_306_eval_600k/point_cloud/iteration_600000/point_cloud.ply",
    vert_path     :str = "output/UNION10EMOEXP_306_eval_600k/point_cloud/iteration_600000/flame_param.npz",
    timestep      :int = 1000,
    save_pcd_path :str = None,
    save_mesh_path:str = None,
):
    # date:2025.03.20
    # note:
    # 从ply文件中(guassian的结果)读取点云并保存到obj文件中
    # 几何从vert中加载
    gaussians=FlameGaussianModel(
        sh_degree=3,n_shape=300,n_expr=100,
    )
    gaussians.load_ply(ply_path,has_target=False)
    mesh_verts=torch.load(vert_path)
    gaussians.verts=mesh_verts
    gaussians.select_mesh_by_timestep(50)
    xyz=gaussians.get_xyz
    if save_pcd_path is not None:
        save_obj_pointcloud(
            save_pcd_path,
            xyz,
        )
    mesh_verts=gaussians.verts[0]
    mesh_faces=gaussians.flame_model.faces
    mesh_verts=mesh_verts.cpu().numpy()
    mesh_faces=mesh_faces.cpu().numpy()
    if save_mesh_path is not None:
        save_obj(
            save_mesh_path,
            mesh_verts,
            mesh_faces,
        )
    return


def check_audio_mesh():
    # date:2025.04.06
    # note:
    # audio-to-mesh的结果和gsa flame好像角度没对齐，检查一下
    # 1.1 加载 audio-to-mesh
    audio_vert_path="/home/zh/master_thesis_supplementary/UniTalker/test_data_for_gsa/D1_angry1_f_111.pt"
    audio_vert=torch.load(audio_vert_path)
    # 1.2 保存mesh查看
    faces=flame_model_pkg.FlameHead(
        shape_params=300,
        expr_params=100,
        add_teeth=True,
    ).faces
    # save_obj(
    #     "./audio_mesh_test.obj",
    #     audio_vert.cpu().numpy()[0],
    #     faces.cpu().numpy(),
    # )
    # 2.1 加载 gsa flame
    gsa_flame_path="data/306_EMO-2_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine/flame_param/00026.npz"
    gsa_flame_data = np.load(gsa_flame_path)
    # for key in gsa_flame_data.keys():
    #     print(key, gsa_flame_data[key].shape)
    # 2.2 将flame参数转为顶点
    
    flame_vert=flame_para_to_vert(
        flame_path=gsa_flame_path,
        add_teeth_flag=True,
        vert_save_path=None,
        obj_save_path=None,
    )
    print(flame_vert.shape)
    # 2.3 保存mesh查看
    save_obj(
        "./gsa_mesh_test.obj",
        flame_vert.cpu().numpy()[0],
        faces.cpu().numpy(),
    )
    # 3.1 检查修改后的audio mesh 先load pt数据
    audio_vert_v2_path="teeth_optimize/obj_demo/D1_angry1_f_111_0408.pt"
    audio_vert_v2=torch.load(audio_vert_v2_path)
    # 3.2 保存mesh查看
    save_obj(
        "./audio_mesh_test_0408.obj",
        audio_vert_v2.cpu().numpy()[0],
        faces.cpu().numpy(),
    )

    return



def main():
    
    for audio_name in all_audio_deformation_files:
        print(audio_name)
        # vert=ID_audio_deformation_convert_v2(
        #     audio_deformation_path=f"/home/zh/master_thesis_supplementary/UniTalker/audio_driven_D1_deformation/{audio_name}.npy",
        #     flame_path="/home/zh/master_thesis_supplementary/GaussianAvatars/output/UNION10EMOEXP_104_eval_600k/point_cloud/iteration_600000/flame_param.npz",
        #     add_teeth_flag=True,
        #     save_path=f"/home/zh/master_thesis_supplementary/UniTalker/test_data_for_gsa_104_v2/{audio_name}.pt",
        # )
        vert=ID_audio_deformation_convert_v1(
            audio_deformation_path=f"/home/zh/master_thesis_supplementary/UniTalker/audio_driven_D1_deformation/{audio_name}.npy",
            flame_path="/home/zh/master_thesis_supplementary/GaussianAvatars/output/UNION10EMOEXP_306_eval_600k/point_cloud/iteration_600000/flame_param.npz",
            add_teeth_flag=True,
            save_path=f"/home/zh/master_thesis_supplementary/UniTalker/test_data_for_gsa_306_v1/{audio_name}.pt",
        )        
    # vert=ID_audio_deformation_convert_v2(
    #     audio_deformation_path=f"/home/zh/master_thesis_supplementary/UniTalker/audio_driven_D1_deformation/D1_angry1_f_111.npy",
    #     flame_path="/home/zh/master_thesis_supplementary/GaussianAvatars/output/UNION10EMOEXP_306_eval_600k/point_cloud/iteration_600000/flame_param.npz",
    #     add_teeth_flag=True,
    #     save_path=f"teeth_optimize/obj_demo/D1_angry1_f_111_0408.pt",
    # )
    # check_audio_mesh()
    # print(vert.shape)
    # print(vert.shape)
    # flame_para_to_vert(
    #     flame_path="data/306_EMO-2_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine/flame_param/00026.npz",
    #     add_teeth_flag=True,
    #     vert_save_path="./check_dataset_redivide_error.pt",
    #     obj_save_path="./check_dataset_redivide_error.obj",
    # )
    # save_obj_pointcloud_from_ply_with_flame_path(
    #     ply_path       = "output/UNION10EMOEXP_306_eval_600k/point_cloud/iteration_600000/point_cloud.ply",
    #     flame_path     = "output/UNION10EMOEXP_306_eval_600k/point_cloud/iteration_600000/flame_param.npz",
    #     timestep       = 0,
    #     save_pcd_path  = "./test_pcd_1000.obj",
    #     save_mesh_path = "./test_mesh_1000.obj",
    # )
    # save_obj_pointcloud_from_ply_with_vert_path(
    #     ply_path       = "output/UNION10EMOEXP_306_eval_600k/point_cloud/iteration_600000/point_cloud.ply",
    #     vert_path      = "/home/zh/master_thesis_supplementary/UniTalker/test_data_for_gsa/D1_angry1_f_111.pt",
    #     timestep       = 1000,
    #     save_pcd_path  = "./test_pcd_1000_vert.obj",
    #     save_mesh_path = "./test_mesh_1000_vert.obj",
    # )
    return
if __name__ == '__main__':
    main()
