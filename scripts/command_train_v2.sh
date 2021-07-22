#/bin/bash
python train/train_allbatches.py --gpu 1 --model frustum_pointnets_v2  --log_dir train/log_v2 --num_point 3500 --max_epoch 201 --batch_size 32 --decay_step 800000 --decay_rate 0.5  #--restore_model_path /root/frustum-pointnets_RSC_RADAR_fil_PC_batch/train/log_v2/21-04-2020-21:11:28/ckpt/model_30.ckpt
