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
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
import pylab as pl
import flame_model.flame as flame_model_pkg
from flame_model.flame import FlameHead
from collections import defaultdict
from utils.obj_io import *
from scene.flame_gaussian_model import FlameGaussianModel
from face_parsing_check.face_parsing_info import face_parsing_info_class


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




def load_flame_model():
    # 0 相关加载
    flame_model_no_teeth   = FlameHead(shape_params=300,expr_params=100,add_teeth=False)
    flame_model_with_teeth = FlameHead(shape_params=300,expr_params=100,add_teeth=True)

    return

def verts_add_teeth(verts_no_teeth,):
    """
    date          : 2025.04.03
    input         : 
    verts_no_teeth: [bs,5023,3]
    output        : 
    teeth_verts   : [bs,5143,3]
    description   : 
    """
    flame_model_no_teeth   = FlameHead(shape_params=300,expr_params=100,add_teeth=False)
    flame_model_with_teeth = FlameHead(shape_params=300,expr_params=100,add_teeth=True)
    teeth_verts = flame_model_with_teeth.add_teeth_to_mesh(verts_no_teeth)
    return teeth_verts


def save_vert_no_teeth_to_path(verts_no_teeth, save_path):
    # verts_no_teeth: [5023,3]
    flame_model_no_teeth   = FlameHead(shape_params=300,expr_params=100,add_teeth=False)
    faces=flame_model_no_teeth.faces
    save_obj(
        save_path,
        verts_no_teeth.cpu().numpy(),
        faces.cpu().numpy(),
        None,
        None
    )
    return

def save_vert_with_teeth_to_path(verts_with_teeth, save_path):
    # verts_with_teeth: [5143,3]
    flame_model_with_teeth = FlameHead(shape_params=300,expr_params=100,add_teeth=True)
    faces=flame_model_with_teeth.faces
    save_obj(
        save_path,
        verts_with_teeth.cpu().numpy(),
        faces.cpu().numpy(),
        None,
        None
    )
    return


def save_teeth_to_path(verts_with_teeth, save_path):
    """
    date:2025.04.03
    input:
    verts_with_teeth: [5143,3]
    save_path:str
    output:
    description:
    从verts_with_teeth中提取出牙齿的顶点坐标，
    并根据先验的面片信息，存储只有牙齿的mesh
    """
    flame_model_with_teeth = FlameHead(shape_params=300,expr_params=100,add_teeth=True)
    faces_with_teeth=flame_model_with_teeth.faces
    face_parsing_info_obj=face_parsing_info_class()
    teeth_faces_index=face_parsing_info_obj.flame_faces_parsing_index_dict["teeth"].tolist()
    target_faces_index=torch.tensor(teeth_faces_index).to(verts_with_teeth.device)
    
    verts_select, faces_select = select_faces_and_verts(verts_with_teeth, faces_with_teeth, target_faces_index)
    # print(verts_select.shape)
    # print(faces_select.shape)
    save_obj(
        save_path,
        verts_select.cpu().numpy(),
        faces_select.cpu().numpy(),
        None,
        None
    )
    return verts_select, faces_select



def main():
    # load_flame_model()
    tmp=torch.load("teeth_optimize/obj_demo/D1_angry1_f_111.pt")[0]
    # save_vert_no_teeth_to_path(
    #     tmp,save_path="teeth_optimize/obj_demo/frame_0_no_teeth.obj"
    # )
    vert_with_teeth=verts_add_teeth(tmp.unsqueeze(0))[0]
    print(vert_with_teeth.shape)
    # save_vert_with_teeth_to_path(
    #     vert_with_teeth,save_path="teeth_optimize/obj_demo/frame_0_with_teeth.obj"
    # )
    save_teeth_to_path(
        vert_with_teeth,save_path="teeth_optimize/obj_demo/frame_0_teeth_only.obj"
    )
    return
if __name__ == '__main__':
    main()
