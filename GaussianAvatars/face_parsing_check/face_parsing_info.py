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
from collections import defaultdict

class face_parsing_info_class():
    def __init__(self):
        self.load_flame_faces_parsing_index()
        pass

    def load_flame_faces_parsing_index(self,):
        flame_faces_parsing_index_dict=defaultdict(torch.Tensor)
        lip_faces_index         = np.loadtxt("face_parsing_check/face_seg/lip.txt",dtype=int)
        left_check_faces_index  = np.loadtxt("face_parsing_check/face_seg/left_check.txt",dtype=int)
        right_check_faces_index = np.loadtxt("face_parsing_check/face_seg/right_check.txt",dtype=int)
        upper_chin_faces_index  = np.loadtxt("face_parsing_check/face_seg/upper_chin.txt",dtype=int)
        lower_chin_faces_index  = np.loadtxt("face_parsing_check/face_seg/lower_chin.txt",dtype=int)
        lower_nose_faces_index  = np.loadtxt("face_parsing_check/face_seg/lower_nose.txt",dtype=int)
        teeth_faces_index       = np.loadtxt("face_parsing_check/face_seg/teeth.txt",dtype=int)
        flame_faces_parsing_index_dict["lip"]=torch.from_numpy(lip_faces_index)
        flame_faces_parsing_index_dict["left_check"]=torch.from_numpy(left_check_faces_index)
        flame_faces_parsing_index_dict["right_check"]=torch.from_numpy(right_check_faces_index)
        flame_faces_parsing_index_dict["upper_chin"]=torch.from_numpy(upper_chin_faces_index)
        flame_faces_parsing_index_dict["lower_chin"]=torch.from_numpy(lower_chin_faces_index)
        flame_faces_parsing_index_dict["lower_nose"]=torch.from_numpy(lower_nose_faces_index)
        flame_faces_parsing_index_dict["teeth"]=torch.from_numpy(teeth_faces_index)
        # return flame_faces_parsing_index_dict
        self.flame_faces_parsing_index_dict=flame_faces_parsing_index_dict
        return


def face_parsing_info():

    return
def main():
    face_parsing_info()
    return
if __name__ == '__main__':
    main()
