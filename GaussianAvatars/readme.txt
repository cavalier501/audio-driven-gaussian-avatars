环境 gaussian-avatars-4
训练：
SUBJECT=306

python train.py \
-s data/UNION10_${SUBJECT}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine \
-m output/UNION10EMOEXP_${SUBJECT}_eval_600k \
--eval --bind_to_mesh --white_background --port 60000

render
python render.py -m <path to trained model>
python render.py -m output/UNION10EMOEXP_${SUBJECT}_eval_600k --skip_train --skip_test
⬆️生成路径在output/UNION10EMOEXP_${SUBJECT}_eval_600k/test_8/ours_24000

