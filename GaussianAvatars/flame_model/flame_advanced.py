from .lbs import lbs, vertices2landmarks, blend_shapes, vertices2joints

import torch
import torch.nn as nn
import numpy as np
import pickle
from collections import defaultdict
try:
    from pytorch3d.io import load_obj
except ImportError:
    from utils.pytorch3d_load_obj import load_obj
import sys
import pdb
from .flame import *
from teeth_optimize.teeth_utils import *
from .lbs import batch_rodrigues

FLAME_MESH_PATH = "flame_model/assets/flame/head_template_mesh.obj"
FLAME_LMK_PATH = "flame_model/assets/flame/landmark_embedding_with_eyes.npy"

# to be downloaded from https://flame.is.tue.mpg.de/download.php
# FLAME_MODEL_PATH = "flame_model/assets/flame/generic_model.pkl"  # FLAME 2020
FLAME_MODEL_PATH = "flame_model/assets/flame/flame2023.pkl"  # FLAME 2023 (versions w/ jaw rotation)
FLAME_PARTS_PATH = "flame_model/assets/flame/FLAME_masks.pkl" # FLAME Vertex Masks

"""
date:2025.04.05
note:
对原flame模型进行改进，增加fitting等功能
python -m flame_model.flame_advanced 测试方法
"""





def to_tensor(array, dtype=torch.float32):
    if "torch.tensor" not in str(type(array)):
        return torch.tensor(array, dtype=dtype)


def to_np(array, dtype=np.float32):
    if "scipy.sparse" in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

def face_vertices(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """
    assert vertices.ndimension() == 3
    assert faces.ndimension() == 3
    assert vertices.shape[0] == faces.shape[0]
    assert vertices.shape[2] == 3
    assert faces.shape[2] == 3

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]


class FlameHead_advanced(FlameHead):
    def fitting_from_5023_vert(self, target_vertices, num_iters=20000, shape=None):
        """
        从目标顶点拟合 FLAME 参数，支持传入 shape 系数（可为[B,300]或[300]）以固定身份形状。
        恢复嘴部形状，num_iters建议大于5000
        Args:
            target_vertices: Tensor, shape (B, 5023, 3)
            num_iters: int
            shape: None | (300,) | (B, 300)
        
        Returns:
            dict of optimized FLAME parameters
        """
        batch_size = target_vertices.shape[0]
        target_vertices=target_vertices[:, :5023]  # 只取前5023个顶点
        device = target_vertices.device

        # 处理 shape 参数（是否固定）
        if shape is None:
            shape = torch.nn.Parameter(torch.zeros(batch_size, 300, device=device))
            optimize_shape = True
        else:
            shape = shape.to(device)
            if shape.ndim == 1 and shape.shape[0] == 300:
                shape = shape.unsqueeze(0).expand(batch_size, -1)  # 变成 [B, 300]
            elif shape.shape != (batch_size, 300):
                raise ValueError(f"shape 参数尺寸错误，应为 (300,) 或 ({batch_size}, 300)，但得到 {shape.shape}")
            optimize_shape = False

        expr = torch.nn.Parameter(torch.zeros(batch_size, 100, device=device))
        rotation = torch.nn.Parameter(torch.zeros(batch_size, 3, device=device))
        neck = torch.nn.Parameter(torch.zeros(batch_size, 3, device=device))
        jaw = torch.nn.Parameter(torch.zeros(batch_size, 3, device=device))
        eyes = torch.nn.Parameter(torch.zeros(batch_size, 6, device=device))
        trans = torch.nn.Parameter(torch.zeros(batch_size, 3, device=device))

        optim_params = [expr, rotation, neck, jaw, eyes, trans]
        if optimize_shape:
            optim_params.insert(0, shape)

        optimizer = torch.optim.Adam(optim_params, lr=1e-3)

        for _ in tqdm(range(num_iters),desc="Fitting FLAME parameters"):
            optimizer.zero_grad()
            current_verts = super().forward(
                shape=shape,
                expr=expr,
                rotation=rotation,
                neck=neck,
                jaw=jaw,
                eyes=eyes,
                translation=trans
            )[0]

            loss = (current_verts[:, :5023] - target_vertices).pow(2).mean()
            loss.backward()
            optimizer.step()

        return {
            'shape': shape.detach() if optimize_shape else shape,
            'expr': expr.detach(),
            'rotation': rotation.detach(),
            'neck': neck.detach(),
            'jaw': jaw.detach(),
            'eyes': eyes.detach(),
            'translation': trans.detach()
        }
    def add_teeth_to_mesh(self,vert_meshes:torch.Tensor,shape=None):
        """
        :param verts_no_teeth: [batch size, 5023, 3]
        :return: verts_with_teeth: [batch size, 5143, 3]
        与flame_model/flame.py中的add_teeth方法不同
        先估计flame参数，再添加牙齿，这里会执行add_teeth操作
        """
        # 5023顶点添加牙齿
        if self.add_teeth==False:
            self.add_teeth()
        vert_meshes=vert_meshes[:, :5023]  # 只取前5023个顶点
        flame_params = self.fitting_from_5023_vert(target_vertices=vert_meshes,shape=shape)
        vert_meshes_with_teeth=self.forward(
            shape=flame_params['shape'],
            expr=flame_params['expr'],
            rotation=flame_params['rotation'],
            neck=flame_params['neck'],
            jaw=flame_params['jaw'],
            eyes=flame_params['eyes'],
            translation=flame_params['translation']
        )[0]
        teeth_vert=vert_meshes_with_teeth[:,5023:5143]
        vert_meshes_with_teeth=torch.cat([vert_meshes, teeth_vert], dim=1)
        return vert_meshes_with_teeth

    # def apply_flame_global_transform(self,mesh, rotation, translation):
    #     """
    #     将 FLAME 标准空间下的 mesh 应用 rotation + translation 变换，转换到世界坐标系。
        
    #     参数：
    #         mesh: (B, N, 3) 处于 FLAME 标准空间的顶点
    #         rotation: (B, 3) axis-angle，表示 FLAME → 世界空间的刚体旋转
    #         translation: (B, 3) 平移向量
            
    #     返回：
    #         mesh_world: (B, N, 3)，处于世界坐标系的 mesh
    #     """
    #     rotmat = batch_rodrigues(rotation)  # (B, 3, 3)
    #     mesh_world = mesh @ rotmat.transpose(1, 2) + translation.unsqueeze(1)  # (B, N, 3)
    #     return mesh_world

    def apply_flame_global_transform(
        self,
        mesh,
        rotation,
        translation,
        jaw_pose=None,
        neck_pose=None,
        eyes_pose=None,
    ):
        """
        将 FLAME 标准空间下的 mesh 应用骨骼驱动 + rotation + translation，转换到世界坐标系。

        参数：
            mesh: (B, N, 3) - 标准空间下的 mesh（= template + ID + expr/audio）
            rotation: (B, 3) - axis-angle，FLAME → 世界空间的整体旋转
            translation: (B, 3) - 平移向量
            jaw_pose: (B, 3) - 下颌旋转（axis-angle），默认零
            neck_pose: (B, 3) - 脖子旋转（axis-angle），默认零
            eyes_pose: (B, 6) - 眼睛姿态，默认零

        返回：
            mesh_world: (B, N, 3)，最终在世界坐标系下的 mesh
        """
        batch_size = mesh.shape[0]
        device = mesh.device

        # 默认值补齐
        if jaw_pose is None:
            jaw_pose = torch.zeros(batch_size, 3, device=device)
        if neck_pose is None:
            neck_pose = torch.zeros(batch_size, 3, device=device)
        if eyes_pose is None:
            eyes_pose = torch.zeros(batch_size, 6, device=device)

        # 构造完整姿态参数（注意 root rotation 是0，这里不重复用 rotation）
        pose_root = torch.zeros(batch_size, 3, device=device)
        full_pose = torch.cat([pose_root, neck_pose, jaw_pose, eyes_pose], dim=1)  # (B, (1+4)*3)

        # 应用 LBS：将标准空间 mesh 转换为 pose 空间（包括骨骼驱动 jaw/neck 等）
        v_lbs, *_ = lbs(
            pose=full_pose,
            v_shaped=mesh,
            posedirs=self.posedirs,
            J_regressor=self.J_regressor,
            parents=self.parents,
            lbs_weights=self.lbs_weights,
            dtype=mesh.dtype,
        )

        # 加上整体刚体旋转和平移（rotation/translation 是 global 头部姿态）
        rotmat = batch_rodrigues(rotation)  # (B, 3, 3)
        mesh_world = v_lbs @ rotmat.transpose(1, 2) + translation.unsqueeze(1)  # (B, N, 3)

        return mesh_world

def flame_advanced_test():
    model = FlameHead_advanced(shape_params=300, expr_params=100,add_teeth=False)    
    target_verts=torch.load("teeth_optimize/obj_demo/D1_angry1_f_111.pt")[[50]].to(model.v_template.device)
    print(target_verts.shape)
    # 5023顶点恢复FLAME参数
    num_iters_list=[100,200,500,1000,2000,5000,10000,20000,50000]
    for num_iters in num_iters_list:
        params = model.fitting_from_5023_vert(target_verts, num_iters=num_iters)
        # 从FLAME参数生成5023顶点
        target_reconstructed = model.forward(**params)[0]  # 取第一个元素
        print(target_reconstructed.shape)
        # save_vert_no_teeth_to_path(
        #     target_verts[0],
        #     "flame_model/data_test/flame_5023_ori.obj"
        # )
        save_vert_no_teeth_to_path(
            target_reconstructed[0], 
            f"flame_model/data_test/flame_5023_reconstructed_iter_{num_iters}.obj"
        )

    return

def load_flame_param():
    flame_path="output/UNION10EMOEXP_306_dataset_redivide/point_cloud/best_ckpt/flame_param.npz"
    # 加载FLAME参数
    data = np.load(flame_path)
    for key in data.keys():
        print(key, data[key].shape)
    return

def add_teeth_test():
    model = FlameHead_advanced(shape_params=300, expr_params=100,add_teeth=False)
    target_verts_no_teeth=torch.load("teeth_optimize/obj_demo/D1_angry1_f_111.pt")[[50]].to(model.v_template.device)
    target_verts_with_teeth=model.add_teeth_to_mesh(target_verts_no_teeth)
    print(target_verts_with_teeth.shape)
    return


def add_teeth_on_frames():
    vert_path="/home/zh/master_thesis_supplementary/UniTalker/test_data_for_gsa/D1_angry1_f_111.pt"
    vert_old=torch.load(vert_path)
    print(vert_old.shape)
    # load flame shape
    flame_para_306_path="/home/zh/master_thesis_supplementary/GaussianAvatars/output/UNION10EMOEXP_306_dataset_redivide/point_cloud/best_ckpt/flame_param.npz"
    flame_para_306_data = np.load(flame_para_306_path)
    shape_306=flame_para_306_data['shape']
    shape_306=torch.tensor(shape_306).to(vert_old.device)
    # 5023顶点添加牙齿
    model = FlameHead_advanced(shape_params=300, expr_params=100,add_teeth=True)
    vert_meshes_with_teeth=model.add_teeth_to_mesh(vert_old,shape_306)
    print(vert_meshes_with_teeth.shape)
    torch.save(vert_meshes_with_teeth,"flame_model/D1_angry1_f_111_add_teeth_0405.pt")
    return

def main():
    # flame_advanced_test()
    # load_flame_param()
    # add_teeth_test()
    add_teeth_on_frames()
    return
if __name__ == '__main__':
    main()
