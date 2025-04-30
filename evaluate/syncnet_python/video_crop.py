import cv2
import os
import os
import subprocess

def extract_and_crop_frames(video_path, save_dir, crop_box, target_fps=None):
    x, y, w, h = crop_box
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件：", video_path)
        return

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(orig_fps / target_fps) if target_fps else 1

    frame_idx = 0
    saved_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:
            cropped = frame[y:y+h, x:x+w]
            save_path = os.path.join(save_dir, f"{saved_idx:05d}.jpg")
            success, encoded_img = cv2.imencode('.jpg', cropped)
            if success:
                with open(save_path, mode='wb') as f:
                    f.write(encoded_img.tobytes())
            saved_idx += 1

        frame_idx += 1

    cap.release()
    print(f"完成：保存 {saved_idx} 帧（每秒 {target_fps or orig_fps:.2f} 帧）到 {save_dir}")

def extract_crop_with_ffmpeg(video_path, save_dir, crop_box, target_fps=None):
    """
    使用ffmpeg提取视频帧并裁剪

    参数:
        video_path (str): 视频路径
        save_dir (str): 保存裁剪帧的路径
        crop_box (tuple): (x, y, w, h) 裁剪框
        target_fps (int or None): 目标帧率（可选）
    """
    os.makedirs(save_dir, exist_ok=True)

    x, y, w, h = crop_box
    crop_str = f"crop={w}:{h}:{x}:{y}"

    cmd = [
        "ffmpeg",
        "-i", video_path,
    ]

    if target_fps:
        cmd += ["-r", str(target_fps)]  # 设置输出帧率

    cmd += [
        "-vf", crop_str,
        os.path.join(save_dir, "%05d.jpg")
    ]

    print("运行命令：", " ".join(cmd))
    subprocess.run(cmd)


import cv2

def draw_crop_box_on_image(image_path, crop_box, save_path=None, color=(0, 0, 0), thickness=2):
    """
    在图片上画出 crop_box 框，默认画黑框，支持保存或直接显示。

    参数:
        image_path (str): 输入图像路径
        crop_box (tuple): (x, y, w, h)
        save_path (str or None): 若指定则保存，否则直接显示
        color (tuple): 线的颜色，默认黑色 (0, 0, 0)
        thickness (int): 线的粗细
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return

    x, y, w, h = crop_box
    pt1 = (x, y)
    pt2 = (x + w, y + h)

    cv2.rectangle(img, pt1, pt2, color, thickness)

    if save_path:
        cv2.imwrite(save_path, img)
        print(f"保存成功：{save_path}")
    else:
        cv2.imshow("Crop Box", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



# 示例调用
if __name__ == "__main__":
    base_name="ours_v2_renders_mesh"
    video_path = f"thesis_evaluate/104_ted1/{base_name}.mp4" # 输入视频路径
    save_dir = f"./video_crop/104_ted1_{base_name}" # 结果保存路径
    target_fps = 25 # 目标帧率
    crop_box = (149, 350, 207, 207) # 左上角(x, y)，宽w，高h

    extract_crop_with_ffmpeg(video_path, save_dir, crop_box,target_fps=target_fps)

