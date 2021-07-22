from __future__ import print_function

#import cPickle as pickle
#import pcl
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from dataset import KittiDataset
from collections import Counter
import kitti_utils
import csv
import pandas
from pypcd import pypcd
try:
    raw_input  # Python 2
except NameError:
    raw_input = input  # Python 3







if __name__ == '__main__':

    dataset_kitti = KittiDataset("pc_radar_f3_vox", root_dir='/home/amben/frustum-pointnets_RSC/dataset/', mode='TRAIN', split="trainval")
    id_list = dataset_kitti.sample_id_list
    det_obj=[]
    present_obj = []
    radar_pts = []
    for i in range(len(id_list)):
        pc_radar =dataset_kitti.get_radar(id_list[i])
        gt_obj_list = dataset_kitti.filtrate_objects(dataset_kitti.get_label(id_list[i]))
        gt_boxes3d = kitti_utils.objs_to_boxes3d(gt_obj_list)

        gt_boxes3d[:,3]=gt_boxes3d[:,3]+2
        gt_boxes3d[:,4] = gt_boxes3d[:,4] + 2
        gt_boxes3d[:,5] = gt_boxes3d[:,5] + 2
        # gt_boxes3d = gt_boxes3d[self.box_present[index] - 1].reshape(-1, 7)
        gt_corners = kitti_utils.boxes3d_to_corners3d(gt_boxes3d, transform=True)
        for k in range(gt_boxes3d.shape[0]):
            box_corners = gt_corners[k]
            fg_pt_flag = kitti_utils.in_hull(pc_radar[:, 0:3], box_corners)
            count_radar = np.count_nonzero(fg_pt_flag==True)
            radar_rest_idx = np.argwhere(fg_pt_flag == False)
            pc_radar = pc_radar[radar_rest_idx.reshape(-1)]
            radar_pos_idx = np.argwhere(fg_pt_flag == True)
            #pc_radar_obj = pc_radar(radar_pos_idx)

            if count_radar>0:

              det_obj.append(count_radar)
        present_obj.append(len(gt_obj_list))
        radar_pts.append(len(pc_radar))
    print("average point cloud per box", np.mean(det_obj))
    print("average detected object per radar",len(det_obj)/np.sum(present_obj))
    #print(np.sum(present_obj))
    print("average radar per frame",np.mean(radar_pts))






