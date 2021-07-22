#/bin/bash
python train/eval_seg.py --gpu 1 --model frustum_pointnets_seg_v2  --log_dir train/log_v2 --num_point 25000 --max_epoch 1 --batch_size 16 --decay_step 800000 --decay_rate 0.5  --restore_model_path /root/frustum-pointnets_RSC_RADAR_fil_PC_batch_para/train/log_v2/27-05-2020-12:47:27/ckpt/model_200.ckpt
