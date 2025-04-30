#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch.utils.data import DataLoader
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
import concurrent.futures
import multiprocessing
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import sys
from copy import deepcopy 

from gaussian_renderer import render,render_face_parsing_func
from utils.general_utils import safe_state
from utils.obj_io import save_obj, read_obj
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, FlameGaussianModel
from mesh_renderer import NVDiffRenderer
import time
from utils.obj_io import save_obj_pointcloud
from face_parsing_check.face_parsing_info import face_parsing_info_class

"""
date:2025.04.10
note:
为了方便写subject=104的代码 单独开一个文件写
"""

mesh_renderer = NVDiffRenderer()

def select_faces_and_verts(verts, faces, target_faces_index):
    """
    Args:
        verts: (V, 3) tensor of vertex coordinates
        faces: (F, 3) tensor of face indices (into verts)
        target_faces_index: (K,) tensor of face indices to select
        
    Returns:
        verts_select: (M, 3) tensor of selected vertex coordinates
        faces_select: (K, 3) tensor of selected face indices (into verts_select)
    """
    # 1. Select the desired faces from original faces
    selected_faces = faces[target_faces_index]
    
    # 2. Get all unique vertex indices used in these faces
    unique_vert_indices = torch.unique(selected_faces.flatten())
    
    # 3. Create mapping from original vertex index to new vertex index
    # This creates a dictionary where keys are original indices and values are new indices
    # We'll represent this as a tensor where index is original vertex and value is new vertex
    vert_mapping = torch.full((verts.shape[0],), -1, dtype=torch.long).to(verts.device)
    vert_mapping[unique_vert_indices] = torch.arange(len(unique_vert_indices)).to(verts.device)
    
    # 4. Remap the face indices to the new vertex indices
    faces_select = vert_mapping[selected_faces]
    
    # 5. Select the corresponding vertices
    verts_select = verts[unique_vert_indices]
    
    return verts_select, faces_select




def write_data(path2data):
    for path, data in path2data.items():
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix in [".png", ".jpg"]:
            data = data.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            Image.fromarray(data).save(path)
        elif path.suffix in [".obj"]:
            with open(path, "w") as f:
                f.write(data)
        elif path.suffix in [".txt"]:
            with open(path, "w") as f:
                f.write(data)
        elif path.suffix in [".npz"]:
            np.savez(path, **data)
        else:
            raise NotImplementedError(f"Unknown file type: {path.suffix}")

def render_set(dataset : ModelParams, name, iteration, views, gaussians, pipeline, background, render_mesh,render_mesh_background:str="no_background"):
    
    if dataset.select_camera_id != -1:
        name = f"{name}_{dataset.select_camera_id}"
    # iter_path = Path(dataset.model_path) / name / f"ours_{iteration}"
    iter_path = Path(dataset.result_save_path) / f"{name}_iter_{iteration}"
    if os.path.exists(iter_path)==False:
        os.makedirs(iter_path)
    # TODO
    render_full_rgb_path = iter_path / "ours_v2"
    render_selected_on_full_parts_path = iter_path / "render_selected_part_on_full"
    gts_path = iter_path / "gt"
    if render_mesh:
        render_mesh_path = iter_path / "renders_mesh"

    if dataset.render_parsed_mesh:
        render_parsed_mesh_path = iter_path / "renders_parsed_mesh"

    makedirs(render_full_rgb_path, exist_ok=True)
    makedirs(render_selected_on_full_parts_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    if dataset.render_face_parsing:
        render_face_parsing_path = iter_path / "renders_mesh_face_parsing"
        makedirs(render_face_parsing_path, exist_ok=True)

    views_loader = DataLoader(views, batch_size=None, shuffle=False, num_workers=8)
    max_threads = multiprocessing.cpu_count()
    print('Max threads: ', max_threads)
    worker_args = []
    face_parsing_info_obj=face_parsing_info_class()
    for idx, view in enumerate(tqdm(views_loader, desc="Rendering progress")):
        # print("step",view.timestep)
        # time.sleep(0.1)
        # if exp_flag==False:
        #     view.timestep-=1003
        if gaussians.binding != None:
            gaussians.select_mesh_by_timestep(view.timestep)
        rendering_full = render(view, gaussians, pipeline, background)["render"]
        rendering_face_parsing = render_face_parsing_func(view, gaussians, pipeline, background,
                            render_face_parsing_list=dataset.render_face_parsing_list,face_parsing_info_obj=face_parsing_info_obj)["render"]
        # blend
        mask = (rendering_face_parsing != background.view(3, 1, 1)).any(dim=0, keepdim=True)
        mask = mask.expand_as(rendering_face_parsing)
        rendering_selected_on_full = torch.where(mask, rendering_face_parsing, 0.7*rendering_full+0.3*background.view(3, 1, 1))
        gt = view.original_image[0:3, :, :]
        if render_mesh:
            # render_mesh_background in ["gt","no_background","rendering_result"]
            out_dict = mesh_renderer.render_from_camera(gaussians.verts, gaussians.faces, view)
            rgba_mesh = out_dict['rgba'].squeeze(0).permute(2, 0, 1)  # (C, W, H)
            rgb_mesh = rgba_mesh[:3, :, :]
            alpha_mesh = rgba_mesh[3:, :, :]
            mesh_opacity = 0.5
            if render_mesh_background == "gt":
                rendering_mesh = rgb_mesh * alpha_mesh * mesh_opacity  + gt.to(rgb_mesh) * (alpha_mesh * (1 - mesh_opacity) + (1 - alpha_mesh))
            elif render_mesh_background == "no_background":
                # print(rgb_mesh.shape,alpha_mesh.shape,background.shape,gt.shape,rendering.shape)
                # input("sss")
                background_chw = background.unsqueeze(-1).unsqueeze(-1).expand_as(rgb_mesh)
                rendering_mesh = rgb_mesh * alpha_mesh * mesh_opacity  + background_chw.to(rgb_mesh) * (alpha_mesh * (1 - mesh_opacity) + (1 - alpha_mesh))
            elif render_mesh_background == "rendering_result":
                rendering_mesh = rgb_mesh * alpha_mesh * mesh_opacity  + rendering.to(rgb_mesh) * (alpha_mesh * (1 - mesh_opacity) + (1 - alpha_mesh))
        
        
        if dataset.render_parsed_mesh:
            target_faces_index=[]
            if "all" in dataset.render_face_parsing_list:
                target_faces_index=range(0, 10144)
            elif dataset.render_face_parsing_list[0]=="without_teeth" and len(dataset.render_face_parsing_list)==1:
                teeth_faces_index=face_parsing_info_obj.flame_faces_parsing_index_dict["teeth"].tolist()
                target_faces_index=[x for x in range(0, 10144) if x not in teeth_faces_index]
            else:
                for face_parsing_name in dataset.render_face_parsing_list:
                    # print(face_parsing_name)
                    target_faces_index.extend(face_parsing_info_obj.flame_faces_parsing_index_dict[face_parsing_name].tolist())
            target_faces_index=torch.tensor(target_faces_index).cuda()            
            # render_mesh_background in ["gt","no_background","rendering_result"]
            verts_select, faces_select = select_faces_and_verts(gaussians.verts[0], gaussians.faces, target_faces_index)
            out_dict = mesh_renderer.render_from_camera(verts_select.unsqueeze(0), faces_select, view)
            rgba_parsed_mesh = out_dict['rgba'].squeeze(0).permute(2, 0, 1)  # (C, W, H)
            rgb_parsed_mesh = rgba_parsed_mesh[:3, :, :]
            alpha_parsed_mesh = rgba_parsed_mesh[3:, :, :]
            parsed_mesh_opacity = 0.5
            if dataset.render_parsed_mesh_background == "gt":
                rendering_parsed_mesh = rgb_parsed_mesh * alpha_parsed_mesh * parsed_mesh_opacity  + gt.to(rgb_parsed_mesh) * (alpha_parsed_mesh * (1 - parsed_mesh_opacity) + (1 - alpha_parsed_mesh))
            elif dataset.render_parsed_mesh_background == "no_background":
                # print(rgb_mesh.shape,alpha_mesh.shape,background.shape,gt.shape,rendering.shape)
                # input("sss")
                background_chw = background.unsqueeze(-1).unsqueeze(-1).expand_as(rgb_parsed_mesh)
                rendering_parsed_mesh = rgb_parsed_mesh * alpha_parsed_mesh * parsed_mesh_opacity  + background_chw.to(rgb_parsed_mesh) * (alpha_parsed_mesh * (1 - parsed_mesh_opacity) + (1 - alpha_parsed_mesh))
            elif dataset.render_parsed_mesh_background == "rendering_result":
                rendering_parsed_mesh = rgb_parsed_mesh * alpha_parsed_mesh * parsed_mesh_opacity  + rendering.to(rgb_parsed_mesh) * (alpha_parsed_mesh * (1 - parsed_mesh_opacity) + (1 - alpha_parsed_mesh))
            elif dataset.render_parsed_mesh_background == "rendered_mesh":
                mask_mesh = (rgb_parsed_mesh != background.view(3, 1, 1)).any(dim=0, keepdim=True)
                mask_mesh = mask_mesh.expand_as(rgb_parsed_mesh)
                rendering_parsed_mesh = torch.where(mask_mesh, rgb_parsed_mesh, 0.7*rendering_mesh.to(rgb_parsed_mesh)+0.3*background.view(3, 1, 1))                
                
        if render_face_parsing_path:
            # render_face_parsing=rendering_face_parsing*0.4+rendering_mesh*0.6
            render_face_parsing=rendering_face_parsing
        
        path2data = {}
        path2data[Path(render_full_rgb_path) / f'{idx:05d}.png'] = rendering_full
        path2data[Path(gts_path) / f'{idx:05d}.png'] = gt
        path2data[Path(render_selected_on_full_parts_path) / f'{idx:05d}.png'] = rendering_selected_on_full
        if render_mesh:
            path2data[Path(render_mesh_path) / f'{idx:05d}.png'] = rendering_mesh
        if render_face_parsing_path:
            path2data[Path(render_face_parsing_path) / f'{idx:05d}.png'] = render_face_parsing
        if dataset.render_parsed_mesh:
            path2data[Path(render_parsed_mesh_path) / f'{idx:05d}.png'] = rendering_parsed_mesh
        worker_args.append([path2data])

        if len(worker_args) == max_threads or idx == len(views_loader)-1:
            with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
                futures = [executor.submit(write_data, *args) for args in worker_args]
                concurrent.futures.wait(futures)
            worker_args = []
    
    try:
        os.environ['PATH'] = "/usr/bin:" + os.environ['PATH']
        # NOTE
        # 25.4.10 将fps调整为30
        # TODO
        os.system(f"ffmpeg -y -framerate 30 -f image2 -pattern_type glob -i '{render_full_rgb_path}/*.png' -c:v libx264 -pix_fmt yuv420p {iter_path}/ours_v2.mp4")
        os.system(f"ffmpeg -y -framerate 30 -f image2 -pattern_type glob -i '{gts_path}/*.png' -c:v libx264 -pix_fmt yuv420p {iter_path}/gt.mp4")
        os.system(f"ffmpeg -y -framerate 30 -f image2 -pattern_type glob -i '{render_selected_on_full_parts_path}/*.png' -c:v libx264 -pix_fmt yuv420p {iter_path}/renders_selected_on_full.mp4")
        if render_mesh:
            os.system(f"ffmpeg -y -framerate 30 -f image2 -pattern_type glob -i '{render_mesh_path}/*.png' -c:v libx264 -pix_fmt yuv420p {iter_path}/renders_mesh.mp4")
        if render_face_parsing_path:
            os.system(f"ffmpeg -y -framerate 30 -f image2 -pattern_type glob -i '{render_face_parsing_path}/*.png' -c:v libx264 -pix_fmt yuv420p {iter_path}/renders_mesh_face_parsing.mp4")
        if dataset.render_parsed_mesh:
            os.system(f"ffmpeg -y -framerate 30 -f image2 -pattern_type glob -i '{render_parsed_mesh_path}/*.png' -c:v libx264 -pix_fmt yuv420p {iter_path}/renders_parsed_mesh.mp4")
    except Exception as e:
        print(e)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_val : bool, skip_test : bool, render_mesh: bool,render_mesh_background:str="no_background"):
    with torch.no_grad():
        if dataset.bind_to_mesh:
            # gaussians = FlameGaussianModel(dataset.sh_degree, dataset.disable_flame_static_offset)
            gaussians = FlameGaussianModel(dataset.sh_degree)
        else:
            gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        # 解决加载audio vert时，帧数大于json中存储的帧数的问题
        # 实际上json中的内容，只是涉及了camera、gt_image
        if exp_flag==False:
            all_cameras = scene.getTrainCameras() + scene.getValCameras() + scene.getTestCameras()
            selected_camera = None
            for cam in all_cameras:
                if cam.timestep == 0:
                    selected_camera = cam
                    break
            custom_views = []
            for frame_id in range(num_of_frames):
                cam_copy = deepcopy(selected_camera)
                cam_copy.timestep = frame_id
                custom_views.append(cam_copy)



        if dataset.target_path != "":
             name = os.path.basename(os.path.normpath(dataset.target_path))
             # when loading from a target path, test cameras are merged into the train cameras
             render_set(dataset, f'{name}', scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, render_mesh,render_mesh_background)
        else:
            if not skip_train:
                render_set(dataset, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, render_mesh,render_mesh_background)
            
            if not skip_val:
                render_set(dataset, "val", scene.loaded_iter, scene.getValCameras(), gaussians, pipeline, background, render_mesh,render_mesh_background)

            if not skip_test:
                if exp_flag==True:
                    render_set(dataset, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, render_mesh,render_mesh_background)
                else:
                    render_set(dataset, "test", scene.loaded_iter, custom_views, gaussians, pipeline, background, render_mesh,render_mesh_background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_val", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_mesh", action="store_true")
    # 需要在这里指定model_path 因为get_combined_args有一个加载cfg_args文件的过程
    parser.set_defaults(model_path="./output/UNION10EMOEXP_104_eval_600k")
    args = get_combined_args(parser)
    # NOTE 路径设置
    args.source_path      = (
        "data/UNION10_104_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine"
    ) # 相关参数加载
    base_name="D1_ted1_f_643" # ["D1_angry1_f_111","D1_angry2_f_120","D1_bazongnew_f_432","D1_ted1_f_643"]
    args.result_save_path=f"output_audio2video_v2/104_{base_name}"
    global exp_flag # 渲染的是表情序列还是语音序列，当渲染语音序列时，只设置skip_test为False，即传入数据集名称为test
    global num_of_frames # 渲染的帧数 当从vert_mesh_path加载时，num_of_frames为vert_meshes的帧数 不从json中加载
    exp_flag=False
    args.load_vert_meshes_flag=True # 为True时，加载flame mesh中的vertices顶点位置，否则加载flame参数
    # ↓基于vert_meshes时，顶点位置文件
    # args.vert_meshes_path=f"/home/zh/master_thesis_supplementary/UniTalker/test_data_for_gsa/{base_name}.pt" 
    args.vert_meshes_path=f"/home/zh/master_thesis_supplementary/UniTalker/test_data_for_gsa_104_v2/{base_name}.pt"
    num_of_frames=torch.load(args.vert_meshes_path).shape[0]
    # print(torch.load(args.vert_meshes_path).shape)
    # sys.exit()
    # ↓基于flame参数时，flame参数文件
    args.flame_path="./output/UNION10EMOEXP_306_dataset_redivide/point_cloud/iteration_600000/flame_param.npz"
    # ↓高斯属性文件（PLY文件）
    args.ply_path="./output/UNION10EMOEXP_104_eval_600k/point_cloud/iteration_600000/point_cloud.ply"
    
    # NOTE 渲染设置
    args.skip_train  = True
    args.skip_test   = False
    args.skip_val    = True
    args.render_mesh = True
    # 渲染mesh时的背景设置：gt/无背景/渲染RGB结果
    args.render_mesh_background="no_background" # ["gt","no_background","rendering_result"]
    # 25.4.2 查看人脸分割区域单独的效果，支持传入多个part
    # ["left_check","lip","lower_chin","lower_nose","right_check","teeth","upper_chin","all","without_teeth"]
    # args.render_face_parsing_list=["left_check","lip","lower_chin","lower_nose","right_check","teeth","upper_chin",]
    args.render_face_parsing_list=["teeth",]
    args.render_face_parsing=True
    args.render_parsed_mesh=True
    args.render_parsed_mesh_background="no_background" # ["gt","no_background","rendering_result","rendered_mesh"] # 其中rendering_result表示分割单独渲染rgb结果与整体渲染rgb结果的叠加
    # for camer_index in range(16):
    #     if camer_index==8: 
    #         continue
    args.select_camera_id = 9 # 9是正面"
    print("Rendering " + args.model_path)
    # 保存结果设置

    # Initialize system state (RNG)
    safe_state(args.quiet)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_val, args.skip_test, args.render_mesh,args.render_mesh_background)