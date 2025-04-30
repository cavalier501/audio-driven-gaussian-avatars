import numpy as np

def save_obj(filename, vertices, faces, vert_colors=None, vert_texcoords=None,
             mtl:str=None):
    """
    faces : np [N_face,3] start from 1, automatic check
    """
    import os
    if np.min(faces)==0:
        faces=faces+1
    with open(filename, 'w') as f:
        f.write('# %s\n' % os.path.basename(filename))
        f.write('#\n')
        f.write('\n')
        if vert_colors is not None:
            vert_colors=np.concatenate((vertices,vert_colors),axis=1)
        else:
            pass
        for vertex in vertices:
            if vert_colors is None:
                f.write('v %.8f %.8f %.8f\n' % (vertex[0], vertex[1], vertex[2]))
            else:
                f.write('v %.8f %.8f %.8f %d %d %d\n' % (vertex[0], vertex[1], vertex[2], vertex[3], vertex[4],vertex[5]))
        f.write('\n');
        if vert_texcoords is not None:
            for vert_texcoord in vert_texcoords:
                f.write('vt %.8f %.8f\n' % (vert_texcoord[0], vert_texcoord[1]))
            f.write('\n');
        if mtl is not None:
            f.write("usemtl "+mtl+"\n")        
        for face in faces:
            f.write('f %d %d %d\n' % (face[0], face[1], face[2]));
        f.write('\n');

    return

def save_obj_with_face_colors(filename, vertices, faces, face_colors):
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

def read_obj(fpath):
    """
    input
    obj_path     : str
    output
    vertice      : np [N,3]
    vn           : np [N,3]
    face         : np [N_face,3] start from 0
    vertice color: np [N,3]
    """
    mesh_dict={}
    vertices=[]
    faces = []
    vns = []
    vertice_colors = []
    for line in open(fpath,"r"):
        if line.startswith("#"):
            continue
        values = line.split()
        if not values:
            continue
        if values[0]=='v':
            v=[float(x) for x in values[1:4]]
            vertices.append(v)
            if len(values)>4:
                vc=[float(x) for x in values[4:7]]
                vertice_colors.append(vc)
        if values[0]=='vn':
            vn=[float(x) for x in values[1:4]]
            vns.append(vn)
        elif values[0]=='f':
            face = []
            for v in values[1:]:
                # w = v.split('//')
                w = v.split('/')
                face.append(int(w[0]))
            faces.append(face)
    mesh_dict['v'] = np.array(vertices)
    mesh_dict['vn'] = np.array(vns)
    mesh_dict['f'] = np.array(faces)
    mesh_dict['vc'] = np.array(vertice_colors)
    faces=np.array(faces)
    if np.min(faces)==1:
        faces=faces-1
    return np.array(vertices),np.array(vns),faces,np.array(vertice_colors)

def save_obj_pointcloud(filename, vertices, vert_colors=None):
    """
    存储点云到 .obj 文件，仅包含顶点（v）和可选的颜色信息。
    
    参数：
    - filename: str, 保存的文件名
    - vertices: np.ndarray [N,3]，点云顶点坐标
    - vert_colors: np.ndarray [N,3]，顶点颜色 (RGB, 0-255)，可选
    """
    import os
    with open(filename, 'w') as f:
        f.write('# %s\n' % os.path.basename(filename))
        f.write('# Point Cloud OBJ file\n')
        f.write('\n')

        if vert_colors is not None:
            assert vertices.shape[0] == vert_colors.shape[0], "顶点和颜色数量不匹配"
            data = np.hstack((vertices, vert_colors))  # 合并顶点和颜色
            for v in data:
                f.write('v %.6f %.6f %.6f %d %d %d\n' % (v[0], v[1], v[2], v[3], v[4], v[5]))
        else:
            for v in vertices:
                f.write('v %.6f %.6f %.6f\n' % (v[0], v[1], v[2]))
        f.write('\n')

    return

def read_obj_pointcloud(fpath):
    """
    读取 .obj 点云文件（仅包含 `v`）。
    
    参数：
    - fpath: str, .obj 文件路径

    返回：
    - vertices: np.ndarray [N, 3]，点云顶点
    - vert_colors: np.ndarray [N, 3]，顶点颜色（如果存在），否则返回 None
    """
    vertices = []
    vert_colors = []

    with open(fpath, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            values = line.split()
            if values[0] == 'v':
                v = [float(x) for x in values[1:4]]  # 读取顶点坐标
                vertices.append(v)
                if len(values) > 4:  # 可能包含颜色信息
                    vc = [int(x) for x in values[4:7]]
                    vert_colors.append(vc)

    vertices = np.array(vertices)
    vert_colors = np.array(vert_colors) if vert_colors else None

    return vertices, vert_colors

def main():
    
    return
if __name__ == '__main__':
    main()