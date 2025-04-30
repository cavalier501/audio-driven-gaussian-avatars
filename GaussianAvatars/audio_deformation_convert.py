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

all_audio_deformation_files=['D1_ted1_f_643', 'D1_malaya_f_302', 
                             'D1_sad_f_127', 'D1_bazongname_f_672', 
                             'D1_disgust_f_104', 'D1_angry2_f_120', 
                             'D1_bazongnew_f_432', 'D1_ted2_f_1076', 
                             'D1_fearful_f_113', 'D1_can_you_feel_the_love_tonight_clip_f_866', 
                             'D1_happy_f_116', 'D1_angry1_f_111']


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
    # input                 : 
    # audio_deformation_path: 音频驱动形变量路径
    # flame_path            : 样本flame参数路径
    # add_teeth_flag        : 是否添加牙齿
    # save_path             : 保存路径
    
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
    

    # 5.保存
    if save_path is not None:
        if add_teeth_flag:
            torch.save(gsa_vert_id_5143,save_path)
        else:
            torch.save(gsa_vert_id_5023,save_path)

    return gsa_vert_id_5143 if add_teeth_flag else gsa_vert_id_5023


def main():
    subject = 306
    result_vert_path=f"../UniTalker/test_data_for_gsa_{subject}_v2"
    os.makedirs(result_vert_path,exist_ok=True)
    for audio_name in all_audio_deformation_files:
        print(audio_name)
        vert=ID_audio_deformation_convert_v2(
            audio_deformation_path=f"../UniTalker/audio_driven_D1_deformation/{audio_name}.npy",
            flame_path=f".//GaussianAvatars/output/UNION10EMOEXP_{subject}_eval_600k/point_cloud/iteration_600000/flame_param.npz",
            add_teeth_flag=True,
            save_path=f"../UniTalker/test_data_for_gsa_{subject}_v2/{audio_name}.pt",
        )


if __name__ == '__main__':
    main()

