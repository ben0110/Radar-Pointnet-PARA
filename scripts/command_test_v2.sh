#/bin/bash
python train/test_para_cls.py --gpu 1 --model frustum_pointnets_v2   --num_point 25000  --batch_size 1   --model_path '/root/frustum-pointnets_RSC_RADAR_fil_PC_batch_kitti/train/log_v2/10-05-2020-13:59:22/ckpt/model_200.ckpt'
