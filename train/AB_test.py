from __future__ import print_function
import time
import os
import sys
import argparse
import importlib
import numpy as np
from datetime import datetime
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
import provider_seg as provider
from train_util import get_batch_seg
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import mayavi
import mayavi.mlab as mlab
from viz_util import draw_lidar, draw_gt_boxes3d
TEST_DATASET = provider.RadarDataset_seg('pc_radar_2',database='KITTI',npoints=25000, split='train',rotate_to_center=False, one_hot=True,all_batches = True, translate_radar_center=False, store_data=True, proposals_3 =False ,no_color=True)
def corneers_from_minmax(min,max):
    corners = np.zeros((8,3))
    corners[0,]=[min[0],max[1],min[2]]
    corners[1,] = [min[0], max[1], max[2]]
    corners[2,] = [max[0], max[1], max[2]]
    corners[3,] = [max[0], max[1], min[2]]
    corners[4,] = [min[0], min[1], min[2]]
    corners[5,] = [min[0], min[1], max[2]]
    corners[6,] = [max[0], min[1], max[2]]
    corners[7,] = [max[0], min[1], min[2]]
    return corners

def local_min_method(bin_pc,centers,size,radar_angle,trans):
    print(len(bin_pc),len(centers))
    bin_y_max = []
    for i in range(len(bin_pc)):
        if(bin_pc[i].size==0):
            bin_y_max.append(centers[i][1] + size[0]/ 2)
        else:
            bin_y_max.append(np.min(bin_pc[i][:, 1]))

    minimum = []
    if (bin_y_max[0] < bin_y_max[1]):
        minimum.append(1)
    else:
        minimum.append(-1)
    for m in range(1, len(bin_y_max) - 1):
        if (bin_y_max[m] < bin_y_max[m - 1] and bin_y_max[m] < bin_y_max[m + 1]):
            minimum.append(1)
        elif (bin_y_max[m] > bin_y_max[m - 1] and bin_y_max[m] > bin_y_max[m + 1]):
            minimum.append(-1)
        else:
            minimum.append(0)
    if (bin_y_max[len(bin_y_max) - 1] < bin_y_max[len(bin_y_max) - 1]):
        minimum.append(1)
    else:
        minimum.append(-1)
    print(minimum)
    local_min_indices = np.argwhere(np.array(minimum) == -1)
    pc_AB_list = []
    corners_AB = []
    for n in range(len(local_min_indices)):
        pc_AB = np.empty([0, 3])
        for m in range(n + 1, len(local_min_indices)):
            for o in range(local_min_indices[n][0], local_min_indices[m][0]):
                if (bin_pc[o].size != 0):
                    pc_AB = np.concatenate((pc_AB, bin_pc[o]))
            print("pc_AB_list:", len(pc_AB_list))
            if (len(pc_AB) > 0):
                min = np.array([np.min(pc_AB[:, 0]), np.min(pc_AB[:, 1]), np.min(pc_AB[:, 2])])
                max = np.array([np.max(pc_AB[:, 0]), np.max(pc_AB[:, 1]), np.max(pc_AB[:, 2])])
                corners = corneers_from_minmax(min, max)
                center = (min + max) / 2.0

                corners = provider.inverse_rotate_pc_along_y(corners, radar_angle)
                corners = corners + trans
                pc_AB_list.append(pc_AB)
                corners_AB.append(corners)
    return pc_AB_list,corners_AB

def divide_in_n_AB(bin_pc,n):
    pc_AB_list=[]
    for i in range(0,len(bin_pc)-n,1):
        pc_AB = np.empty([0, 3])
        #print(len(bin_pc))
        #print(i,i+n)
        for j in range(i,i+n):
            #print(j)
            #print(bin_pc[j].size)
            if bin_pc[j].size != 0:
                pc_AB=np.concatenate((pc_AB,bin_pc[j]))
        pc_AB_list.append(pc_AB)
    return pc_AB_list

def iterative_method(bin_pc,centers,size,radar_angle,trans):
    pc_AB_list=[]
    corners_AB=[]
    for i in range(3,6):
        pc_AB_=divide_in_n_AB(bin_pc,i)
        for pc_ in pc_AB_:
            if (len(pc_) > 0):
                min = np.array([np.min(pc_[:, 0]), np.min(pc_[:, 1]), np.min(pc_[:, 2])])
                max = np.array([np.max(pc_[:, 0]), np.max(pc_[:, 1]), np.max(pc_[:, 2])])
                corners = corneers_from_minmax(min, max)
                center = (min + max) / 2.0

                corners = provider.inverse_rotate_pc_along_y(corners, radar_angle)
                corners = corners + trans
                corners_AB.append(corners)
                pc_AB_list.append(pc_)
    return pc_AB_list,corners_AB

def get_max_iou(gt_corners,corners_AB,pc_orig):
    iou_list=[]
    for n in range(len(gt_corners)):
        max_iou = 0.0
        corner_id = 0
        gt_id = 0

        for o in range(len(corners_AB)):
            iou_3d, iou_2d = provider.box3d_iou(corners_AB[o], gt_corners[n])
            if (iou_3d > max_iou):
                max_iou = iou_3d
                corner_id = n
                gt_id = o
        print("yohoo", max_iou, "pred box: ", corner_id, " gt box: ", gt_id)
        """if (max_iou == 0):
            fig = mlab.figure(figure=None, bgcolor=(0.4, 0.4, 0.4), fgcolor=None, engine=None,
                              size=(1000, 500))
            mlab.points3d(pc_orig[:, 0], pc_orig[:, 1], pc_orig[:, 2], mode='point', colormap='gnuplot', scale_factor=1,
                          figure=fig)
            draw_gt_boxes3d(gt_corners, fig, color=(0, 0, 1))
            for s in range(len(corners_AB)):
                draw_gt_boxes3d([corners_AB[s]], fig, color=(1, 0, 0))
            mlab.orientation_axes()
            provider.raw_input()"""
        if (max_iou < 1.0):
            iou_list.append(max_iou)
    return iou_list

if __name__ == "__main__":
    BATCH_SIZE=4
    num_batches = len(TEST_DATASET) / BATCH_SIZE
    train_idxs = np.arange(0, len(TEST_DATASET))
    AB_total = []
    time_total = []
    average_iou =np.empty([0])
    len_pc_radar= []
    for batch_idx in range(int(num_batches)):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE
        batch_data, batch_label, batch_one_hot_vec,batch_radar_mask_list,radar_rois,ids = \
            get_batch_seg(TEST_DATASET, train_idxs, start_idx, end_idx,
                          25000, 3)

        for i in range(len(batch_radar_mask_list)):
            time1 = time.time()
            AB_pc_list= []
            AB_corners_list= []
            len_pc_radar.append(len(batch_radar_mask_list[i]))
            for j in range(len(batch_radar_mask_list[i])):

                labels_per_roi = batch_radar_mask_list[i][j]*batch_label[i]
                if(np.count_nonzero(labels_per_roi==1)>50):
                    pos_indices = np.where(labels_per_roi== 1)[0]
                    point_set = batch_data[i][pos_indices, :]
                    pc_orig=point_set
                    trans = np.array([radar_rois[i][j][0],radar_rois[i][j][1],radar_rois[i][j][2]])
                    pc = point_set - trans
                    pc = provider.rotate_pc_along_y(pc,radar_rois[i][j][6])
                    min=np.array([np.min(pc[:,0]),np.min(pc[:,1]),np.min(pc[:,2])])
                    max=np.array([np.max(pc[:,0]),np.max(pc[:,1]),np.max(pc[:,2])])
                    radar_rot= provider.rotate_pc_along_y(np.array([[radar_rois[i][j][0], radar_rois[i][j][1], radar_rois[i][j][2],radar_rois[i][j][6]]]),radar_rois[i][j][6])
                    corners = corneers_from_minmax(min, max)
                    center = (min+max)/2.0
                    print("frame id: ",ids[i])
                    gt_obj_list = TEST_DATASET.dataset_kitti.filtrate_objects(
                        TEST_DATASET.dataset_kitti.get_label(ids[i]))
                    gt_boxes3d = provider.kitti_utils.objs_to_boxes3d(gt_obj_list)
                    gt_corners = provider.kitti_utils.boxes3d_to_corners3d(gt_boxes3d, transform=False)
                    centers = []
                    bin_pc = []
                    #for f in range(len(gt_corners)):
                    #    gt_corners[f] = provider.rotate_pc_along_y(gt_corners[f], radar_rois[i][j][6])
                    #draw_gt_boxes3d(gt_corners, fig, color=(1, 1, 0))
                    l=abs(max[2]-min[2])
                    h=abs(max[1]-min[1])
                    center_1=center
                    center_2=center
                    w=1/8
                    ds=0
                    boxes_1 = provider.get_3d_box((h, w, l), 0.0, center)
                    fg_pt_flag_1 = provider.kitti_utils.in_hull(pc[:, 0:3], boxes_1)

                    if (np.count_nonzero(fg_pt_flag_1 == 1) > 50):
                        pc_1 = pc[fg_pt_flag_1, :]
                        bin_pc.append(pc_1)
                        centers.append(center)
                    else:
                        bin_pc.append(np.array([]))
                        centers.append(center)
                    size=[h,w,l]
                    while center_2[0]<max[0]:
                        center_1= [center_1[0]-1/8,center_1[1],center_1[2]]
                        center_2 = [center_2[0] + 1 / 8, center_2[1], center_2[2]]
                        boxes_1 = provider.get_3d_box((h, w, l), 0.0, center_1)
                        boxes_2 = provider.get_3d_box((h, w, l), 0.0, center_2)
                        time1_1=time.time()
                        fg_pt_flag_1 = provider.kitti_utils.in_hull(pc[:, 0:3], boxes_1)
                        fg_pt_flag_2 = provider.kitti_utils.in_hull(pc[:, 0:3], boxes_2)
                        if np.count_nonzero(fg_pt_flag_1 == 1)>50:
                            pc_1=pc[fg_pt_flag_1,:]
                            bin_pc.append(pc_1)
                            centers.append(center_1)
                        else:
                            bin_pc.append(np.array([]))
                            centers.append(center_1)

                        if np.count_nonzero(fg_pt_flag_2 == 1)>50:
                            pc_2=pc[fg_pt_flag_2, :]
                            bin_pc.insert(0,pc_2)
                            centers.insert(0,center_2)
                        else:
                            bin_pc.insert(0,np.array([]))
                            centers.insert(0,center_2)
                        fg_pt_flag=np.logical_or(fg_pt_flag_1, fg_pt_flag_2)
                        pc=pc[~fg_pt_flag,:]
                    AB_pc,AB_corners = local_min_method(bin_pc,centers,size,radar_rois[i][j][6],trans)
                    #AB_pc,AB_corners = iterative_method(bin_pc,centers,size,radar_rois[i][j][6],trans)
                    print(len(AB_corners),len(AB_pc))
                    for q in range(len(AB_pc)):
                        AB_pc_list.append(AB_pc[q])
                        AB_corners_list.append(AB_corners[q])
            if(len(AB_pc_list)>0):
                iou_list = get_max_iou(gt_corners, AB_corners_list, pc_orig)
                average_iou = np.concatenate((average_iou, iou_list))

            print("number of AB in frame: ",len(AB_pc_list))
            AB_total.append(AB_pc_list)

            time2= time.time()
            time_total.append(time2-time1)
    print("pc radar:",np.max(len_pc_radar))

    print(" max:", len_pc_radar)
    print("average iOu:",np.mean(np.asarray(average_iou)))
    print("average time: ",np.mean(np.asarray(time_total)))
    max_AB=[]
    max=0
    for s in range(len(AB_total)):
        if max<len(AB_total[s]):
            max_AB.append(len(AB_total[s]))
            max=len(AB_total[s])
    print("max AB",max)
    total_anchors=0
    for i in range(len(AB_total)):
        total_anchors=total_anchors+len(AB_total[i])
    print("average AB: ", total_anchors/len(AB_total))
    plt.plot(len_pc_radar)
    plt.ylabel('len_pc_radar')
    plt.show()
    plt.plot(max_AB)
    plt.ylabel('max')
    plt.show()


