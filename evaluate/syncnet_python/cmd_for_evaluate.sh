#!/bin/bash
# 路径说明：
# video_path为待评测合成视频的路径
# audio_path为输入音频的路径
# output_path将输入音频添加至合成视频的路径，即输出待音频的合成视频
#    默认最后会删除该文件
# 运行后，会在终端输出评测结果
base_name="306_angry1"
evaluate_method="ours_no_teeth"
video_path="thesis_evaluate/${base_name}/${evaluate_method}.mp4"  
audio_path="thesis_evaluate/${base_name}/${base_name:4}.wav" 
output_path="thesis_evaluate/${base_name}/${evaluate_method}_with_audio.mp4"  


# 执行 FFmpeg 合并命令
ffmpeg \
  -i "$video_path" \
  -i "$audio_path" \
  -c:v copy \
  -c:a aac \
  -shortest \
  -y "$output_path"


python run_pipeline.py \
    --videofile "$output_path" \
    --reference myvideo \
    --data_dir theis_evaluate_tmp_dir/
    

python run_syncnet.py \
    --videofile "$output_path" \
    --reference myvideo \
    --data_dir theis_evaluate_tmp_dir

rm ${output_path}