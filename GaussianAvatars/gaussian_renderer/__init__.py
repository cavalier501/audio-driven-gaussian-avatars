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
import math
from typing import Union
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene import GaussianModel, FlameGaussianModel
from utils.sh_utils import eval_sh
from utils.obj_io import save_obj_pointcloud
from face_parsing_check.face_parsing_info import face_parsing_info_class

def render(viewpoint_camera, pc : Union[GaussianModel, FlameGaussianModel], pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}


def render_face_parsing_func(viewpoint_camera, pc : Union[GaussianModel, FlameGaussianModel], pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, 
                override_color = None,render_face_parsing_list=["all"],
                face_parsing_info_obj=face_parsing_info_class()):
    """
    25.4.2
    增加只渲染部分face_parsing的功能
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    
    def select_gaussian_attribute_from_target_face_index(full_gaussian_attribute, binding, target_face_index):
        # date:2025.04.02
        # note:
        # 输入：
        # 1 每个高斯点云的属性tensor（如点云位置）[N,...]
        # 2 每个高斯点云绑定到了哪个面片的binding信息 [N,]
        # 3 目标面片的索引 [m,]
        # 输出：
        # 绑定到目标面片的高斯点云属性tensor，注意点云中点的数量需要计算确定 [M,...]
        mask = torch.isin(binding, target_face_index)
        target_gaussian_attribute = full_gaussian_attribute[mask]
        return torch.clone(target_gaussian_attribute)


    target_faces_index=[]
    if "all" in render_face_parsing_list:
        target_faces_index=range(0, 10144)
    elif render_face_parsing_list[0]=="without_teeth" and len(render_face_parsing_list)==1:
        teeth_faces_index=face_parsing_info_obj.flame_faces_parsing_index_dict["teeth"].tolist()
        target_faces_index=[x for x in range(0, 10144) if x not in teeth_faces_index]
    else:
        for face_parsing_name in render_face_parsing_list:
            # print(face_parsing_name)
            target_faces_index.extend(face_parsing_info_obj.flame_faces_parsing_index_dict[face_parsing_name].tolist())
    target_faces_index=torch.tensor(target_faces_index).cuda()
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    binding=pc.binding
    selected_means3D   = select_gaussian_attribute_from_target_face_index(means3D, binding, target_faces_index)
    selected_means2D   = select_gaussian_attribute_from_target_face_index(means2D, binding, target_faces_index)
    selected_shs       = select_gaussian_attribute_from_target_face_index(shs, binding, target_faces_index)
    selected_opacity   = select_gaussian_attribute_from_target_face_index(opacity, binding, target_faces_index)
    selected_scales    = select_gaussian_attribute_from_target_face_index(scales, binding, target_faces_index)
    selected_rotations = select_gaussian_attribute_from_target_face_index(rotations, binding, target_faces_index)
    rendered_image, radii = rasterizer(
        means3D = selected_means3D,
        means2D = selected_means2D,
        shs = selected_shs,
        colors_precomp = colors_precomp,
        opacities = selected_opacity,
        scales = selected_scales,
        rotations = selected_rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}
