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

from gaussian_renderer import render
from utils.general_utils import safe_state
from utils.obj_io import save_obj, read_obj
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, FlameGaussianModel
from mesh_renderer import NVDiffRenderer
import time
from utils.obj_io import save_obj_pointcloud


mesh_renderer = NVDiffRenderer()

import torch



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
    render_path = iter_path / "renders"
    gts_path = iter_path / "gt"
    if render_mesh:
        render_mesh_path = iter_path / "renders_mesh"


    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    views_loader = DataLoader(views, batch_size=None, shuffle=False, num_workers=8)
    max_threads = multiprocessing.cpu_count()
    print('Max threads: ', max_threads)
    worker_args = []
    for idx, view in enumerate(tqdm(views_loader, desc="Rendering progress")):
        # print("step",view.timestep)
        # time.sleep(0.1)
        if exp_flag==False:
            view.timestep-=1003
        if gaussians.binding != None:
            gaussians.select_mesh_by_timestep(view.timestep)
        rendering = render(view, gaussians, pipeline, background)["render"]
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
        

        path2data = {}
        path2data[Path(render_path) / f'{idx:05d}.png'] = rendering
        path2data[Path(gts_path) / f'{idx:05d}.png'] = gt
        if render_mesh:
            path2data[Path(render_mesh_path) / f'{idx:05d}.png'] = rendering_mesh
        worker_args.append([path2data])

        if len(worker_args) == max_threads or idx == len(views_loader)-1:
            with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
                futures = [executor.submit(write_data, *args) for args in worker_args]
                concurrent.futures.wait(futures)
            worker_args = []
    
    try:
        os.environ['PATH'] = "/usr/bin:" + os.environ['PATH']
        # os.system(f"ffmpeg -y -framerate 25 -f image2 -pattern_type glob -i '{render_path}/*.png' -c:v libx264 -pix_fmt yuv420p {iter_path}/renders.mp4")
        # TODO
        os.system(f"ffmpeg -y -framerate 25 -f image2 -pattern_type glob -i '{render_path}/*.png' -c:v libx264 -pix_fmt yuv420p {iter_path}/renders_rgb_result.mp4")
        os.system(f"ffmpeg -y -framerate 25 -f image2 -pattern_type glob -i '{gts_path}/*.png' -c:v libx264 -pix_fmt yuv420p {iter_path}/gt.mp4")
        if render_mesh:
            os.system(f"ffmpeg -y -framerate 25 -f image2 -pattern_type glob -i '{render_mesh_path}/*.png' -c:v libx264 -pix_fmt yuv420p {iter_path}/renders_mesh.mp4")
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
                render_set(dataset, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, render_mesh,render_mesh_background)

if __name__ == "__main__":
    """
    对原始的render.py进行了修改，
    通过load_vert_meshes_flag参数，支持传入FLAME mesh逐个顶点坐标来完成人脸驱动
    通过exp_flag参数，表征渲染的是表情序列还是音频序列，二者仅在序列编号方面有差异
    在本项目中，
    设置load_vert_meshes_flag=True; 设置exp_flag=False
    相关路径说明：
    1，source_path：高斯样本相关路径
    2，vert_meshes_path：（通过逐顶点坐标驱动模式下）FLAME mesh逐顶点坐标文件路径
    3，flame_path：（通过FLAME参数驱动模式下）FLAME参数文件路径
    4，ply_path：样本高斯属性文件路径
    5，result_save_path：渲染结果保存路径
    """
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
    parser.set_defaults(model_path=".//GaussianAvatars/output/UNION10EMOEXP_306_eval_600k")
    args = get_combined_args(parser)
    # NOTE 路径设置
    args.source_path      = (
        ".//GaussianAvatars/data/UNION10_306_EMO1234EXP234589_v16_DS2"
        "-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine" 
    ) # 相关参数加载
    # base_name="D1_can_you_feel_the_love_tonight_clip_f_866"
    base_name="D1_angry1_f_111"
    args.result_save_path=f"output/audio_check_demo"
    global exp_flag # 渲染的是表情序列还是语音序列
    exp_flag=False
    args.load_vert_meshes_flag=True # 为True时，加载flame mesh中的vertices顶点位置，否则加载flame参数
    # ↓基于vert_meshes时，顶点位置文件
    args.vert_meshes_path=f"/home/zh/master_thesis_supplementary/UniTalker/test_data_for_gsa_306_v2/{base_name}.pt" 
    # ↓基于flame参数时，flame参数文件
    args.flame_path=".//GaussianAvatars/output/UNION10EMOEXP_306_eval_600k/point_cloud/iteration_600000/flame_param.npz"
    # ↓高斯属性文件（PLY文件）
    args.ply_path=".//GaussianAvatars/output/UNION10EMOEXP_306_eval_600k/point_cloud/iteration_600000/point_cloud.ply"
    
    # NOTE 渲染设置
    args.skip_train  = True
    args.skip_test   = False
    args.skip_val    = True
    args.render_mesh = True
    # 渲染mesh时的背景设置：gt/无背景/渲染RGB结果
    args.render_mesh_background="no_background" # ["gt","no_background","rendering_result"]
    args.select_camera_id = 8 # 8是正面
    print("Rendering " + args.model_path)
    # 保存结果设置
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_val, args.skip_test, args.render_mesh,args.render_mesh_background)