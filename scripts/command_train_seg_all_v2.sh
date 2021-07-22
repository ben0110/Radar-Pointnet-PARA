#/bin/bash
python train/train_seg_all.py --gpu 1 --model frustum_pointnets_seg_v2  --log_dir train/log_v2 --num_point 35000 --max_epoch 11 --batch_size 16 --decay_step 800000 --decay_rate 0.5  --restore_model_path /root/frustum-pointnets_RSC_RADAR_fil_PC_batch_para/train/log_v2/25-06-2020-20:05:48/ckpt/model_190.ckpt
