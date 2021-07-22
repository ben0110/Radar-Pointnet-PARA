#/bin/bash
python train/train_seg.py --gpu 1 --model frustum_pointnets_seg_v2  --log_dir train/log_v2 --num_point 3500 --max_epoch 201 --batch_size 32 --decay_step 800000 --decay_rate 0.5  #--restore_model_path /root/frustum-pointnets_RSC_RADAR_fil_PC_batch_para/train/log_v2/04-08-2020-12:15:01/ckpt/model_130.ckpt