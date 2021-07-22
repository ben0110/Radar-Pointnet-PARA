''' Provider class and helper functions for Frustum PointNets.

Author: Charles R. Qi
Date: September 2017
'''
from __future__ import print_function

# import cPickle as pickle
# import pcl
import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import math
import random
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
# import mayavi
# import mayavi.mlab as mlab

sys.path.append(os.path.join(ROOT_DIR, 'mayavi'))
# from viz_util import draw_lidar, draw_gt_boxes3d

from box_util import box3d_iou
from model_util import g_type2class, g_class2type, g_type2onehotclass
from model_util import g_type_mean_size
from model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER

from dataset import KittiDataset
from collections import Counter
import kitti_utils
import pickle
import csv
import pandas
from pypcd import pypcd
import math

try:
    raw_input  # Python 2
except NameError:
    raw_input = input  # Python 3

class RADAR_dataset_seg_to_bbox(object):
    def __init__(self,AB_pc,AB_corners,batch_idx,TEST_DATASET):

        gt_obj_list=TEST_DATASET.dataset_kitti.get_label(batch_idx)
        gt_boxes3d = kitti_utils.objs_to_boxes3d(gt_obj_list)
        gt_corners = kitti_utils.boxes3d_to_corners3d(gt_boxes3d, transform=False)
        self.one_hot=TEST_DATASET.one_hot

        self.dataset=TEST_DATASET

        self.AB=[]
        self.type_list=[]
        self.box3d_list=[]
        self.AB_list=[]
        self.size_list=[]
        self.heading_list=[]
        self.batch_list=[]
        self.indice_box=[]

        for k in range(len(AB_corners)):
            for m in range(len(gt_corners)):
                # print("corners AB", AB_corners[k])
                # print("gt_corners[m]", gt_corners[m])
                if len(np.unique(AB_corners[k][:, 0])) == 1:
                    continue
                iou_3d, iou_2d = box3d_iou(AB_corners[k], gt_corners[m])
                print(iou_3d)
                if iou_3d > 0.0:
                    self.AB.append(AB_pc[k])
                    self.type_list.append("Pedestrian")
                    self.box3d_list.append(gt_corners[m])
                    self.AB_list.append(AB_corners[k])
                    self.size_list.append([gt_boxes3d[m][3], gt_boxes3d[m][4], gt_boxes3d[m][5]])
                    self.heading_list.append(gt_boxes3d[m][6])
                    self.batch_list.append(batch_idx)
                    self.indice_box.append(m)
                elif iou_3d == 0.0 :
                    self.AB.append(AB_pc[k])
                    box3d_center = np.random.rand(3) * (-10.0)
                    size = np.ones((3))
                    box3d = np.array(
                        [[box3d_center[0], box3d_center[1], box3d_center[2], size[0], size[1],
                          size[2], 0.0]])
                    corners_empty = kitti_utils.boxes3d_to_corners3d(box3d, transform=False)

                    self.type_list.append("Pedestrian")
                    self.box3d_list.append(corners_empty[0])
                    self.AB_list.append(AB_corners[k])

                    self.size_list.append(size)
                    self.heading_list.append(0.0)
                    self.batch_list.append(batch_idx)
                    self.indice_box.append(10)
        self.id_list = self.batch_list
    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        ''' Get index-th element from the picked file dataset. '''
        # ------------------------------ INPUTS ----------------------------

        # Compute one hot vector
        # label_mask = self.batch_train[index]
        # rot_angle = self.get_center_view_rot_angle(index)

        # Compute one hot vector
        if self.dataset.one_hot:
            cls_type = self.type_list[index]
            assert (cls_type in ['Car', 'Pedestrian', 'Cyclist'])
            one_hot_vec = np.zeros((3))
            one_hot_vec[g_type2onehotclass[cls_type]] = 1

        # Get point cloud
        if self.dataset.rotate_to_center:
            point_set = self.get_center_view_point_set(index)
        else:
            point_set = self.AB[index]
        # Resample
        choice = np.random.choice(point_set.shape[0], 512, replace=True)
        point_set = point_set[choice, :]

        # ------------------------------ LABELS ----------------------------

        if (self.dataset.no_color):
            ret_pts_features = np.ones((len(point_set), 1))
            point_set = np.concatenate((point_set[:, 0:3], ret_pts_features), axis=1)
        point_set = point_set[:, 0:3]
        pc_orig = point_set
        # Get center point of 3D box
        if self.dataset.rotate_to_center:
            box3d_center = self.get_center_view_box3d_center(index)
            proposal_center = self.get_center_view_proposal(index)

        else:
            box3d_center = self.get_box3d_center(index)
            # proposal_center = self.radar_point_list[index]

        if self.dataset.translate_to_radar_center:
            box3d_center = box3d_center - proposal_center
            point_set[:, 0] = point_set[:, 0] - proposal_center[0]
            point_set[:, 1] = point_set[:, 1] - proposal_center[1]
            point_set[:, 2] = point_set[:, 2] - proposal_center[2]
        # Heading
        if self.dataset.rotate_to_center:
            heading_angle = self.heading_list[index] - rot_angle
        else:
            heading_angle = self.heading_list[index]

        # Size
        size_class, size_residual = size2class(self.size_list[index],
                                               self.type_list[index])
        # translate point cloud to mean center
        center_mean = [np.mean(point_set[:, 0]), np.mean(point_set[:, 1]), np.mean(point_set[:, 2])]
        point_set[:, 0] = point_set[:, 0] - center_mean[0]
        point_set[:, 1] = point_set[:, 1] - center_mean[1]
        point_set[:, 2] = point_set[:, 2] - center_mean[2]
        # translate GT box to mean center
        # box3d_center[0]=box3d_center[0]-center_mean[0]
        # box3d_center[1] = box3d_center[1] - center_mean[1]
        # box3d_center[2] = box3d_center[2] - center_mean[2]
        # Data Augmentation
        if self.dataset.random_flip:
            # note: rot_angle won't be correct if we have random_flip
            # so do not use it in case of random flipping.
            if np.random.random() > 0.5:  # 50% chance flipping
                point_set[:, 0] *= -1
                box3d_center[0] *= -1
                heading_angle = np.pi - heading_angle
        if self.dataset.random_shift:
            dist = np.sqrt(np.sum(box3d_center[0] ** 2 + box3d_center[1] ** 2))
            shift = np.clip(np.random.randn() * dist * 0.05, dist * 0.8, dist * 1.2)
            point_set[:, 2] += shift
            box3d_center[2] += shift

        angle_class, angle_residual = angle2class(heading_angle,
                                                  NUM_HEADING_BIN)
        # print("10 points",point_set[0:10])
        if self.dataset.one_hot:
            return point_set, center_mean, one_hot_vec, box3d_center, angle_class, angle_residual, \
                   size_class, size_residual, self.AB_list[index]
        else:
            return point_set, center_mean, box3d_center, angle_class, angle_residual, \
                   size_class, size_residual, self.AB_list[index]
    def get_center_view_rot_angle(self, index):
        ''' Get the frustum rotation angle, it isshifted by pi/2 so that it
        can be directly used to adjust GT heading angle '''
        return np.pi / 2.0 + self.frustum_angle_list[index]
        # return self.frustum_angle_list[index]

    def get_box3d_center(self, index):
        ''' Get the center (XYZ) of 3D bounding box. '''
        box3d_center = (self.box3d_list[index][0, :] + \
                        self.box3d_list[index][6, :]) / 2.0
        return box3d_center

    def get_center_view_box3d_center(self, index):
        ''' Frustum rotation of 3D bounding box center. '''
        box3d_center = (self.box3d_list[index][0, :] + \
                        self.box3d_list[index][6, :]) / 2.0
        return rotate_pc_along_y(np.expand_dims(box3d_center, 0), \
                                 self.get_center_view_rot_angle(index)).squeeze()

    def get_center_view_proposal(self, index):
        return rotate_pc_along_y(np.expand_dims(self.radar_point_list[index], 0), \
                                 self.get_center_view_rot_angle(index)).squeeze()

    def get_center_view_box3d(self, index):
        ''' Frustum rotation of 3D bounding box corners. '''
        box3d = self.box3d_list[index]
        box3d_center_view = np.copy(box3d)
        return rotate_pc_along_y(box3d_center_view, \
                                 self.get_center_view_rot_angle(index))

    def get_center_view_point_set(self, index):
        ''' Frustum rotation of point clouds.
        NxC points with first 3 channels as XYZ
        z is facing forward, x is left ward, y is downward
        '''
        # Use np.copy to avoid corrupting original data
        point_set = np.copy(self.input_list[index])
        return rotate_pc_along_y(point_set, \
                                 self.get_center_view_rot_angle(index))


def rotate_pc_along_y(pc, rot_angle):
    '''
    Input:
        pc: numpy array (N,C), first 3 channels are XYZ
            z is facing forward, x is left ward, y is downward
        rot_angle: rad scalar
    Output:
        pc: updated pc with XYZ rotated
    '''
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, -sinval], [sinval, cosval]])
    pc[:, [0, 2]] = np.dot(pc[:, [0, 2]], np.transpose(rotmat))
    return pc


def inverse_rotate_pc_along_y(pc, rot_angle):
    '''
    Input:
        pc: numpy array (N,C), first 3 channels are XYZ
            z is facing forward, x is left ward, y is downward
        rot_angle: rad scalar
    Output:
        pc: updated pc with XYZ rotated
    '''
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, sinval], [-sinval, cosval]])
    pc[:, [0, 2]] = np.dot(pc[:, [0, 2]], np.transpose(rotmat))
    return pc


def angle2class(angle, num_class):
    ''' Convert continuous angle to discrete class and residual.

    Input:
        angle: rad scalar, from 0-2pi (or -pi~pi), class center at
            0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
        num_class: int scalar, number of classes N
    Output:
        class_id, int, among 0,1,...,N-1
        residual_angle: float, a number such that
            class*(2pi/N) + residual_angle = angle
    '''
    angle = angle % (2 * np.pi)
    assert (angle >= 0 and angle <= 2 * np.pi)
    angle_per_class = 2 * np.pi / float(num_class)
    shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
    class_id = int(shifted_angle / angle_per_class)
    residual_angle = shifted_angle - \
                     (class_id * angle_per_class + angle_per_class / 2)
    return class_id, residual_angle


def class2angle(pred_cls, residual, num_class, to_label_format=True):
    ''' Inverse function to angle2class.
    If to_label_format, adjust angle to the range as in labels.
    '''
    angle_per_class = 2 * np.pi / float(num_class)
    angle_center = pred_cls * angle_per_class
    angle = angle_center + residual
    if to_label_format and angle > np.pi:
        angle = angle - 2 * np.pi
    return angle


def size2class(size, type_name):
    ''' Convert 3D bounding box size to template class and residuals.
    todo (rqi): support multiple size clusters per type.

    Input:
        size: numpy array of shape (3,) for (l,w,h)
        type_name: string
    Output:
        size_class: int scalar
        size_residual: numpy array of shape (3,)
    '''
    size_class = g_type2class[type_name]
    size_residual = size - g_type_mean_size[type_name]
    return size_class, size_residual


def class2size(pred_cls, residual):
    ''' Inverse function to size2class. '''
    mean_size = g_type_mean_size[g_class2type[pred_cls]]
    return mean_size + residual


"""
def get_closest_radar_point(center,input_radar):
    cloud = pcl.PointCloud()
    cloud.from_array(input_radar[:,0:3])
    center_pc = pcl.PoinCloud()
    center_pc.from_array(center)
    kdtree = cloud
    [ind,sqdist] = kdtree.nearst_k_search_for_cloud(center_pc,0)
    closest_radar_point=np.array([cloud[ind[0][0]][0],cloud[ind[0][0]][1],cloud[ind[0][0]][2]])
"""


def get_box3d_center(box3d_list):
    ''' Get the center (XYZ) of 3D bounding box. '''
    box3d_center = (box3d_list[0, :] + \
                    box3d_list[6, :]) / 2.0
    return box3d_center


def get_radar_mask(input, input_radar):
    radar_mask = np.zeros((input.shape[0]), dtype=np.float32)
    gt_boxes3d = np.zeros((len(input_radar), 7), dtype=np.float32)
    for k in range(len(input_radar)):
        gt_boxes3d[k, 0] = input_radar[k, 0]
        gt_boxes3d[k, 1] = input_radar[k, 1]
        gt_boxes3d[k, 2] = input_radar[k, 2] + 1.0
        gt_boxes3d[k, 3] = 5.0
        gt_boxes3d[k, 4] = (np.tan((15.0) * np.pi / 180.0) * 2) * math.sqrt(
            math.pow(input_radar[k, 2], 2) + math.pow(input_radar[k, 0], 2)) + 2.0
        gt_boxes3d[k, 5] = 4.0
        gt_boxes3d[k, 6] = np.arctan2(input_radar[k, 0], input_radar[k, 2] + 1.0)
        print(gt_boxes3d)
    gt_corners = kitti_utils.boxes3d_to_corners3d(gt_boxes3d, transform=False)
    for k in range(len(gt_corners)):
        box_corners = gt_corners[k]
        fg_pt_flag = kitti_utils.in_hull(input[:, 0:3], box_corners)
        radar_mask[fg_pt_flag] = 1.0
    radar_masks = []
    radar_masks.append(radar_mask)

    radar_mask_center = []
    radar_mask_center.append(get_box3d_center(gt_corners[0]))
    """ seg_idx = np.argwhere(radar_mask == 1.0)
    pc_test = input[seg_idx.reshape(-1)]
    fig = mlab.figure(figure=None, bgcolor=(0.4, 0.4, 0.4), fgcolor=None, engine=None,
                      size=(1000, 500))
    mlab.points3d(input[:, 0], input[:, 1], input[:, 2], radar_mask, mode='point', colormap='gnuplot',
                  scale_factor=1, figure=fig)
    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2, figure=fig)
    draw_gt_boxes3d([gt_corners[0]], fig, color=(1, 0, 0))
    mlab.orientation_axes()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pc_test[:, 0], pc_test[:, 1], pc_test[:, 2], c=pc_test[:, 3:6], s=1)
    plt.show()"""
    return radar_masks, radar_mask_center


def get_radar_masks(input, input_radar):
    center_y = []
    center_y.append(input_radar[0, 0] - ((np.tan((7.5) * np.pi / 180) * 2) * math.sqrt(
        math.pow(input_radar[0, 2], 2) + math.pow(input_radar[0, 0], 2)) + 2.0) / 4)
    center_y.append(input_radar[0, 0])
    center_y.append(input_radar[0, 0] + ((np.tan((7.5) * np.pi / 180) * 2) * math.sqrt(
        math.pow(input_radar[0, 2], 2) + math.pow(input_radar[0, 0], 2)) + 2.0) / 4)
    radar_masks = []
    radar_center_masks = []
    corners3d = []
    for i in range(3):
        gt_boxes3d = np.zeros((len(input_radar), 7), dtype=np.float32)
        for k in range(len(input_radar)):
            gt_boxes3d[k, 0] = center_y[i]
            gt_boxes3d[k, 1] = input_radar[k, 1]
            gt_boxes3d[k, 2] = input_radar[k, 2] + 1.0

            gt_boxes3d[k, 3] = 5.0
            gt_boxes3d[k, 4] = ((np.tan((7.5) * np.pi / 180) * 2) * math.sqrt(
                math.pow(input_radar[k, 2], 2) + math.pow(input_radar[k, 0], 2)) + 2.0) / 2.0
            gt_boxes3d[k, 5] = 4.0

            gt_boxes3d[k, 6] = np.arctan2(input_radar[k, 0], input_radar[k, 2] + 1.0)
        gt_corners = kitti_utils.boxes3d_to_corners3d(gt_boxes3d, transform=False)
        radar_center_masks.append(get_box3d_center(gt_corners[0]))
        radar_mask = np.zeros((input.shape[0]), dtype=np.float32)
        for j in range(len(gt_corners)):
            box_corners = gt_corners[j]
            corners3d.append(box_corners)
            fg_pt_flag = kitti_utils.in_hull(input[:, 0:3], box_corners)
            radar_mask[fg_pt_flag] = 1.0
        radar_masks.append(radar_mask)
    """fig = mlab.figure(figure=None, bgcolor=(0.4, 0.4, 0.4), fgcolor=None, engine=None,
                      size=(1000, 500))
    mlab.points3d(input[:, 0], input[:, 1], input[:, 2], radar_masks[0], mode='point', colormap='gnuplot',
                  scale_factor=1, figure=fig)
    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2, figure=fig)
    draw_gt_boxes3d(corners3d, fig, color=(1, 0, 0))

    mlab.orientation_axes()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(input[:, 0], input[:, 1], input[:, 2], c=input[:, 3:6], s=1)
    plt.show()"""

    return radar_masks, radar_center_masks


def load_GT_eval(indice, database,split):
    data_val = KittiDataset("pc_radar_2", root_dir='/root/frustum-pointnets_RSC/dataset/',
                                          dataset=database,
                                          mode='TRAIN',
                                          split=split)
    id_list = data_val.sample_id_list
    obj_frame = []
    corners_frame = []
    size_class_frame = []
    size_residual_frame = []
    angle_class_frame = []
    angle_residual_frame = []
    center_frame = []
    id_list_new = []
    for i in range(len(id_list)):
        if (id_list[i] < indice + 1):
            gt_obj_list = data_val.filtrate_objects(
                data_val.get_label(id_list[i]))
            # print("GT objs per frame", id_list[i],len(gt_obj_list))
            gt_boxes3d = kitti_utils.objs_to_boxes3d(gt_obj_list)
            gt_corners = kitti_utils.boxes3d_to_corners3d(gt_boxes3d, transform=False)
            obj_frame.append(gt_obj_list)
            corners_frame.append(gt_corners)
            angle_class_list = []
            angle_residual_list = []
            size_class_list = []
            size_residual_list = []
            center_list = []
            for j in range(len(gt_obj_list)):
                angle_class, angle_residual = angle2class(gt_boxes3d[j][6],
                                                          NUM_HEADING_BIN)
                angle_class_list.append(angle_class)
                angle_residual_list.append(angle_residual)

                size_class, size_residual = size2class(np.array([gt_boxes3d[j][3], gt_boxes3d[j][4], gt_boxes3d[j][5]]),
                                                       "Pedestrian")
                size_class_list.append(size_class)
                size_residual_list.append(size_residual)

                center_list.append((gt_corners[j][0, :] + gt_corners[j][6, :]) / 2.0)
            size_class_frame.append(size_class_list)
            size_residual_frame.append(size_residual_list)
            angle_class_frame.append(angle_class_list)
            angle_residual_frame.append(angle_residual_list)
            center_frame.append(center_list)
            id_list_new.append(id_list[i])

    return corners_frame, id_list_new


def get_radar_pc_mask(input, input_radar):
    print("radar pc mask")
    radar_mask = np.zeros((input.shape[0]), dtype=np.float32)
    RoI_boxes3d = np.zeros((len(input_radar), 7), dtype=np.float32)
    for k in range(len(input_radar)):
        # print(pc_radar[j].reshape(-1, 3).shape[0])
        RoI_boxes3d[k, 0] = input_radar[k, 0]
        RoI_boxes3d[k, 1] = input_radar[k, 1]
        RoI_boxes3d[k, 2] = input_radar[k, 2] + 1.0
        RoI_boxes3d[k, 3] = 5.0
        RoI_boxes3d[k, 4] = (np.tan(15.0 * np.pi / 180.0) * 2) * math.sqrt(
            math.pow(input_radar[k, 2], 2) + math.pow(input_radar[k, 0], 2))
        RoI_boxes3d[k, 5] = 4.0
        RoI_boxes3d[k, 6] = np.arctan2(input_radar[k, 0], input_radar[k, 2] + 1.0)
        gt_corners = kitti_utils.boxes3d_to_corners3d(RoI_boxes3d, transform=False)
    radar_mask_list = []
    for k in range(len(gt_corners)):
        box_corners = gt_corners[k]
        fg_pt_flag = kitti_utils.in_hull(input[:, 0:3], box_corners)
        radar_mask_local = np.zeros((input.shape[0]), dtype=np.float32)
        radar_mask_local[fg_pt_flag] = 1.0
        radar_mask_list.append(radar_mask_local)
        radar_mask[fg_pt_flag] = 1.0

    return radar_mask, radar_mask_list, RoI_boxes3d


def corneers_from_minmax(min, max):
    corners = np.zeros((8, 3))
    corners[0,] = [min[0], max[1], min[2]]
    corners[1,] = [min[0], max[1], max[2]]
    corners[2,] = [max[0], max[1], max[2]]
    corners[3,] = [max[0], max[1], min[2]]
    corners[4,] = [min[0], min[1], min[2]]
    corners[5,] = [min[0], min[1], max[2]]
    corners[6,] = [max[0], min[1], max[2]]
    corners[7,] = [max[0], min[1], min[2]]
    return corners

class RadarDataset_seg_per_ROI(object):
    ''' Dataset class for Frustum PointNets training/evaluation.
    Load prepared KITTI data from pickled files, return individual data element
    [optional] along with its annotations.
    '''

    def __init__(self, radar_file, database, npoints, split,
                 random_flip=False, random_shift=False, rotate_to_center=False,
                 overwritten_data_path=None, from_rgb_detection=False, one_hot=False, all_batches=False,
                 translate_radar_center=False,
                 store_data=False, proposals_3=False, no_color=False):  # ,generate_database=False):
        '''
        Input:
            npoints: int scalar, number of points for frustum point cloud.
            split: string, train or val
            random_flip: bool, in 50% randomly flip the point cloud
                in left and right (after the frustum rotation if any)
            random_shift: bool, if True randomly shift the point cloud
                back and forth by a random distance
            rotate_to_center: bool, whether to do frustum rotation
            overwritten_data_path: string, specify pickled file path.
                if None, use default path (with the split)
            from_rgb_detection: bool, if True we assume we do not have
                groundtruth, just return data elements.
            one_hot: bool, if True, return one hot vector
        '''
        self.no_color = no_color
        self.translate_to_radar_center = translate_radar_center
        self.dataset_kitti = KittiDataset(radar_file, root_dir='/root/frustum-pointnets_RSC/dataset/',
                                          dataset=database,
                                          mode='TRAIN',
                                          split=split)
        self.all_batches = all_batches
        self.npoints = npoints
        self.random_flip = random_flip
        self.random_shift = random_shift
        self.rotate_to_center = rotate_to_center
        self.one_hot = one_hot
        self.pc_lidar_list = []
        if (proposals_3):
            box_number = 'threeboxes'
        else:
            box_number = 'one_boxes'
        if overwritten_data_path is None:
            overwritten_data_path = os.path.join('/root/frustum-pointnets_RSC_RADAR_fil_PC_batch/',
                                                 'dataset/RSC/radar_' + box_number + ('_%s.pickle' % (split)))
        output_filename = overwritten_data_path
        self.from_rgb_detection = from_rgb_detection

        # list = os.listdir("/root/3D_BoundingBox_Annotation_Tool_3D_BAT/input/NuScenes/ONE/pointclouds_Radar")
        self.id_list = self.dataset_kitti.sample_id_list
        self.idx_batch = self.id_list
        batch_list = []
        self.radar_OI = []
        self.batch_size = []
        batch_list = []
        self.input_list = []
        self.radar_masks = []
        self.label_list = []
        self.box3d_list = []
        self.radar_point_list = []
        self.indice_box = []
        self.type_list = []
        self.RoI_parameters = []
        self.GT_boxes = []
        accuracy_list = []
        frame_true_50 = []
        frame_true_70 = []
        if store_data:
            for i in range(len(self.id_list)):
                print(self.id_list[i])
                pc_radar = self.dataset_kitti.get_radar(self.id_list[i])
                pc_lidar = self.dataset_kitti.get_lidar(self.id_list[i])

                radar_mask, radar_mask_list, RoI_boxes_3d = get_radar_pc_mask(pc_lidar, pc_radar)
                for j in range(len(radar_mask_list)):
                    if (np.count_nonzero(radar_mask_list[j] == 1)) < 50:
                        print("no pc extracted", self.id_list[i])
                        accuracy_list.append(0)
                        frame_true_50.append(0)
                        frame_true_70.append(0)
                        continue
                    else:
                        radar_idx = np.argwhere(radar_mask_list[j] == 1)
                        pc_fil = pc_lidar[radar_idx.reshape(-1)]
                        #radar_masks_frame = []
                        #for k in range(len(radar_mask_list)):
                        #    radar_masks_frame.append(radar_mask_list[k][radar_idx.reshape(-1)])
                        cls_label = np.zeros((pc_fil.shape[0]), dtype=np.int32)
                        gt_obj_list = self.dataset_kitti.filtrate_objects(
                            self.dataset_kitti.get_label(self.id_list[i]))
                        gt_boxes3d = kitti_utils.objs_to_boxes3d(gt_obj_list)
                        gt_corners = kitti_utils.boxes3d_to_corners3d(gt_boxes3d, transform=False)
                        for k in range(gt_boxes3d.shape[0]):
                            box_corners = gt_corners[k]
                            fg_pt_flag = kitti_utils.in_hull(pc_fil[:, 0:3], box_corners)
                            cls_label[fg_pt_flag] = 1
                        cls_label_gt = np.zeros((pc_lidar.shape[0]), dtype=np.int32)
                        for k in range(gt_boxes3d.shape[0]):
                            box_corners = gt_corners[k]
                            fg_pt_flag = kitti_utils.in_hull(pc_lidar[:, 0:3], box_corners)
                            cls_label_gt[fg_pt_flag] = 1
                        gt_in_radar_masks = np.logical_and(cls_label_gt, radar_mask)
                        if (float(np.count_nonzero(cls_label_gt == 1)) != 0):
                            accuracy = np.count_nonzero(gt_in_radar_masks == 1) / float(np.count_nonzero(cls_label_gt == 1))
                            accuracy_list.append(accuracy)
                            print("GT accuracy in masks:", accuracy)
                            if (accuracy > 0.5):
                                frame_true_50.append(1)
                            else:
                                frame_true_50.append(0)
                            if (accuracy > 0.7):
                                frame_true_70.append(1)
                            else:
                                frame_true_70.append(0)
                        else:
                            continue
                        self.input_list.append(pc_fil)
                        batch_list.append(self.id_list[i])
                        self.label_list.append(cls_label)
                        self.type_list.append("Pedestrian")
                        self.radar_masks.append(radar_mask_list[j])
                        self.RoI_parameters.append(RoI_boxes_3d[j])
                        self.GT_boxes.append(gt_corners)
            self.id_list = batch_list

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        ''' Get index-th element from the picked file dataset. '''
        # ------------------------------ INPUTS ----------------------------

        # Compute one hot vector
        if self.one_hot:
            cls_type = self.type_list[index]
            assert (cls_type in ['Car', 'Pedestrian', 'Cyclist'])
            one_hot_vec = np.zeros((3))
            one_hot_vec[g_type2onehotclass[cls_type]] = 1

        # Get point cloud

        point_set = self.input_list[index]
        # Resample
        # print(point_set.shape[0])
        choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        point_set = point_set[choice, :]
        radar_mask_frame = []
        #for i in range(len(self.radar_masks[index])):
        #    radar_mask = np.asarray(self.radar_masks[index][i])
        #    radar_mask_frame.append(radar_mask[choice])
        radar_mask = np.asarray(self.radar_masks[index])
        radar_mask_frame=radar_mask[choice]
        RoI_para = self.RoI_parameters[index]
        # ------------------------------ LABELS ----------------------------
        seg = self.label_list[index]
        seg = seg[choice]

        gt_corners = self.GT_boxes[index]
        # print("shape",point_set.shape)
        if (self.no_color):
            ret_pts_features = np.ones((len(point_set), 1))
            point_set = np.concatenate((point_set[:, 0:3], ret_pts_features), axis=1)

        if self.one_hot:
            return point_set, seg, one_hot_vec, radar_mask_frame, RoI_para, self.id_list[index]
        else:
            return point_set, seg, radar_mask_frame, RoI_para, self.id_list[index]


class RadarDataset_seg(object):
    ''' Dataset class for Frustum PointNets training/evaluation.
    Load prepared KITTI data from pickled files, return individual data element
    [optional] along with its annotations.
    '''

    def __init__(self, radar_file, database, npoints, split,
                 random_flip=False, random_shift=False, rotate_to_center=False,
                 overwritten_data_path=None, from_rgb_detection=False, one_hot=False, all_batches=False,
                 translate_radar_center=False,
                 store_data=False, proposals_3=False, no_color=False):  # ,generate_database=False):
        '''
        Input:
            npoints: int scalar, number of points for frustum point cloud.
            split: string, train or val
            random_flip: bool, in 50% randomly flip the point cloud
                in left and right (after the frustum rotation if any)
            random_shift: bool, if True randomly shift the point cloud
                back and forth by a random distance
            rotate_to_center: bool, whether to do frustum rotation
            overwritten_data_path: string, specify pickled file path.
                if None, use default path (with the split)
            from_rgb_detection: bool, if True we assume we do not have
                groundtruth, just return data elements.
            one_hot: bool, if True, return one hot vector
        '''
        self.no_color = no_color
        self.translate_to_radar_center = translate_radar_center
        self.dataset_kitti = KittiDataset(radar_file, root_dir='/root/frustum-pointnets_RSC/dataset/',
                                          dataset=database,
                                          mode='TRAIN',
                                          split=split)
        self.all_batches = all_batches
        self.npoints = npoints
        self.random_flip = random_flip
        self.random_shift = random_shift
        self.rotate_to_center = rotate_to_center
        self.one_hot = one_hot
        self.pc_lidar_list = []
        if (proposals_3):
            box_number = 'threeboxes'
        else:
            box_number = 'one_boxes'
        if overwritten_data_path is None:
            overwritten_data_path = os.path.join('/root/frustum-pointnets_RSC_RADAR_fil_PC_batch/',
                                                 'dataset/RSC/radar_' + box_number + ('_%s.pickle' % (split)))
        output_filename = overwritten_data_path
        self.from_rgb_detection = from_rgb_detection

        # list = os.listdir("/root/3D_BoundingBox_Annotation_Tool_3D_BAT/input/NuScenes/ONE/pointclouds_Radar")
        self.id_list = self.dataset_kitti.sample_id_list
        self.idx_batch = self.id_list
        batch_list = []
        self.radar_OI = []
        self.batch_size = []
        batch_list = []
        self.input_list = []
        self.radar_masks = []
        self.label_list = []
        self.box3d_list = []
        self.radar_point_list = []
        self.indice_box = []
        self.type_list = []
        self.RoI_parameters = []
        self.GT_boxes = []
        accuracy_list = []
        frame_true_50 = []
        frame_true_70 = []
        if store_data:
            for i in range(len(self.id_list)):
                print(self.id_list[i])
                pc_radar = self.dataset_kitti.get_radar(self.id_list[i])
                pc_lidar = self.dataset_kitti.get_lidar(self.id_list[i])

                radar_mask, radar_mask_list, RoI_boxes_3d = get_radar_pc_mask(pc_lidar, pc_radar)
                if (np.count_nonzero(radar_mask == 1)) < 50:
                    print("no pc extracted", self.id_list[i])
                    accuracy_list.append(0)
                    frame_true_50.append(0)
                    frame_true_70.append(0)
                    continue
                else:
                    radar_idx = np.argwhere(radar_mask == 1)
                    pc_fil = pc_lidar[radar_idx.reshape(-1)]
                    radar_masks_frame = []
                    for k in range(len(radar_mask_list)):
                        radar_masks_frame.append(radar_mask_list[k][radar_idx.reshape(-1)])
                    cls_label = np.zeros((pc_fil.shape[0]), dtype=np.int32)
                    gt_obj_list = self.dataset_kitti.filtrate_objects(
                        self.dataset_kitti.get_label(self.id_list[i]))
                    gt_boxes3d = kitti_utils.objs_to_boxes3d(gt_obj_list)
                    gt_corners = kitti_utils.boxes3d_to_corners3d(gt_boxes3d, transform=False)
                    for k in range(gt_boxes3d.shape[0]):
                        box_corners = gt_corners[k]
                        fg_pt_flag = kitti_utils.in_hull(pc_fil[:, 0:3], box_corners)
                        cls_label[fg_pt_flag] = 1
                    cls_label_gt = np.zeros((pc_lidar.shape[0]), dtype=np.int32)
                    for k in range(gt_boxes3d.shape[0]):
                        box_corners = gt_corners[k]
                        fg_pt_flag = kitti_utils.in_hull(pc_lidar[:, 0:3], box_corners)
                        cls_label_gt[fg_pt_flag] = 1
                    gt_in_radar_masks = np.logical_and(cls_label_gt, radar_mask)
                    if (float(np.count_nonzero(cls_label_gt == 1)) != 0):
                        accuracy = np.count_nonzero(gt_in_radar_masks == 1) / float(np.count_nonzero(cls_label_gt == 1))
                        accuracy_list.append(accuracy)
                        print("GT accuracy in masks:", accuracy)
                        if (accuracy > 0.5):
                            frame_true_50.append(1)
                        else:
                            frame_true_50.append(0)
                        if (accuracy > 0.7):
                            frame_true_70.append(1)
                        else:
                            frame_true_70.append(0)
                    else:
                        continue
                    self.input_list.append(pc_fil)
                    batch_list.append(self.id_list[i])
                    self.label_list.append(cls_label)
                    self.type_list.append("Pedestrian")
                    self.radar_masks.append(radar_masks_frame)
                    self.RoI_parameters.append(RoI_boxes_3d)
                    self.GT_boxes.append(gt_corners)
            self.id_list = batch_list
            #print("id_list", len(self.id_list))
            #print("self.input_list", len(self.input_list))
            #print("mean accuracy:", np.mean(accuracy_list))
            #print("gt ratio 50:", np.mean(frame_true_50))
            #print("gt ratio 70:", np.mean(frame_true_70))
            # print("average nbr point cloud", np.sum(self.input_list[:,0])/len(self.id_list))

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        ''' Get index-th element from the picked file dataset. '''
        # ------------------------------ INPUTS ----------------------------

        # Compute one hot vector
        if self.one_hot:
            cls_type = self.type_list[index]
            assert (cls_type in ['Car', 'Pedestrian', 'Cyclist'])
            one_hot_vec = np.zeros((3))
            one_hot_vec[g_type2onehotclass[cls_type]] = 1

        # Get point cloud

        point_set = self.input_list[index]
        # Resample
        # print(point_set.shape[0])
        choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        point_set = point_set[choice, :]
        radar_mask_frame = []
        for i in range(len(self.radar_masks[index])):
            radar_mask = np.asarray(self.radar_masks[index][i])
            radar_mask_frame.append(radar_mask[choice])

        RoI_para = self.RoI_parameters[index]
        # ------------------------------ LABELS ----------------------------
        seg = self.label_list[index]
        seg = seg[choice]

        gt_corners = self.GT_boxes[index]
        # print("shape",point_set.shape)
        if (self.no_color):
            ret_pts_features = np.ones((len(point_set), 1))
            point_set = np.concatenate((point_set[:, 0:3], ret_pts_features), axis=1)

        if self.one_hot:
            return point_set, seg, one_hot_vec, radar_mask_frame, RoI_para, self.id_list[index]
        else:
            return point_set, seg, radar_mask_frame, RoI_para, self.id_list[index]


class RadarDataset_seg_cls(object):
    ''' Dataset class for Frustum PointNets training/evaluation.
    Load prepared KITTI data from pickled files, return individual data element
    [optional] along with its annotations.
    '''

    def __init__(self, radar_file, database, npoints, split,
                 random_flip=False, random_shift=False, rotate_to_center=False,
                 overwritten_data_path=None, from_rgb_detection=False, one_hot=False, all_batches=False,
                 translate_radar_center=False,
                 store_data=False, proposals_3=False, no_color=False):  # ,generate_database=False):
        '''
        Input:
            npoints: int scalar, number of points for frustum point cloud.
            split: string, train or val
            random_flip: bool, in 50% randomly flip the point cloud
                in left and right (after the frustum rotation if any)
            random_shift: bool, if True randomly shift the point cloud
                back and forth by a random distance
            rotate_to_center: bool, whether to do frustum rotation
            overwritten_data_path: string, specify pickled file path.
                if None, use default path (with the split)
            from_rgb_detection: bool, if True we assume we do not have
                groundtruth, just return data elements.
            one_hot: bool, if True, return one hot vector
        '''
        self.no_color = no_color
        self.translate_to_radar_center = translate_radar_center
        self.dataset_kitti = KittiDataset(radar_file, root_dir='/root/frustum-pointnets_RSC/dataset/', dataset=database,
                                          mode='TRAIN',
                                          split=split)
        self.all_batches = all_batches
        self.npoints = npoints
        self.random_flip = random_flip
        self.random_shift = random_shift
        self.rotate_to_center = rotate_to_center
        self.one_hot = one_hot
        self.pc_lidar_list = []
        if (proposals_3):
            box_number = 'threeboxes'
        else:
            box_number = 'one_boxes'
        if overwritten_data_path is None:
            overwritten_data_path = os.path.join('/root/frustum-pointnets_RSC_RADAR_fil_PC_batch/',
                                                 'dataset/RSC/radar_' + box_number + ('_%s_seg_cls.pickle' % (split)))
        output_filename = overwritten_data_path
        self.from_rgb_detection = from_rgb_detection

        # list = os.listdir("/root/3D_BoundingBox_Annotation_Tool_3D_BAT/input/NuScenes/ONE/pointclouds_Radar")
        self.id_list = self.dataset_kitti.sample_id_list
        self.idx_batch = self.id_list
        batch_list = []
        self.radar_OI = []
        self.batch_size = []
        batch_list = []
        self.input_list = []
        self.radar_masks = []
        self.label_list = []
        self.box3d_list = []
        self.radar_point_list = []
        self.indice_box = []
        self.type_list = []
        self.RoI_parameters = []
        self.GT_boxes = []
        accuracy_list = []
        frame_true_50 = []
        frame_true_70 = []
        if store_data:
            for i in range(len(self.id_list)):
                print(self.id_list[i])
                pc_radar = self.dataset_kitti.get_radar(self.id_list[i])
                pc_lidar = self.dataset_kitti.get_lidar(self.id_list[i])

                radar_mask, radar_mask_list, RoI_boxes_3d = get_radar_pc_mask(pc_lidar, pc_radar)
                if (np.count_nonzero(radar_mask == 1)) < 50:
                    print("no pc extracted", self.id_list[i])
                    accuracy_list.append(0)
                    frame_true_50.append(0)
                    frame_true_70.append(0)
                    continue
                else:
                    radar_idx = np.argwhere(radar_mask == 1)
                    pc_fil = pc_lidar[radar_idx.reshape(-1)]
                    radar_masks_frame = []
                    for k in range(len(radar_mask_list)):
                        radar_masks_frame.append(radar_mask_list[k][radar_idx.reshape(-1)])
                    cls_label = np.zeros((pc_fil.shape[0]), dtype=np.int32)
                    gt_obj_list = self.dataset_kitti.filtrate_objects(
                        self.dataset_kitti.get_label(self.id_list[i]))
                    gt_boxes3d = kitti_utils.objs_to_boxes3d(gt_obj_list)
                    gt_corners = kitti_utils.boxes3d_to_corners3d(gt_boxes3d, transform=False)

                    for k in range(gt_boxes3d.shape[0]):
                        box_corners = gt_corners[k]
                        fg_pt_flag = kitti_utils.in_hull(pc_fil[:, 0:3], box_corners)
                        cls_label[fg_pt_flag] = 1
                    if np.count_nonzero(cls_label == 1) > 50:
                        cls_label_gt = np.zeros((pc_lidar.shape[0]), dtype=np.int32)
                        for k in range(gt_boxes3d.shape[0]):
                            box_corners = gt_corners[k]
                            fg_pt_flag = kitti_utils.in_hull(pc_lidar[:, 0:3], box_corners)
                            cls_label_gt[fg_pt_flag] = 1
                        gt_in_radar_masks = np.logical_and(cls_label_gt, radar_mask)
                        if (float(np.count_nonzero(cls_label_gt == 1)) != 0):
                            accuracy = np.count_nonzero(gt_in_radar_masks == 1) / float(
                                np.count_nonzero(cls_label_gt == 1))
                            accuracy_list.append(accuracy)
                            print("GT accuracy in masks:", accuracy)
                            if (accuracy > 0.5):
                                frame_true_50.append(1)
                            else:
                                frame_true_50.append(0)
                            if (accuracy > 0.7):
                                frame_true_70.append(1)
                            else:
                                frame_true_70.append(0)
                        else:
                            continue
                        self.input_list.append(pc_fil)
                        batch_list.append(self.id_list[i])
                        self.label_list.append(cls_label)
                        self.type_list.append("Pedestrian")
                        self.radar_masks.append(radar_masks_frame)

                        self.RoI_parameters.append(RoI_boxes_3d)
                        gt_corners_stat = np.zeros((5, 8, 3))
                        for w in range(len(gt_corners)):
                            gt_corners_stat[w, :] = gt_corners[w, :]
                        self.GT_boxes.append(gt_corners_stat)

            self.id_list = batch_list
            print("id_list", len(self.id_list))
            print("self.id_list",self.id_list)
            print("self.input_list", len(self.input_list))
            print("mean accuracy:", np.mean(accuracy_list))
            print("gt ratio 50:", np.mean(frame_true_50))
            print("gt ratio 70:", np.mean(frame_true_70))
            # print("average nbr point cloud", np.sum(self.input_list[:,0])/len(self.id_list))

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        ''' Get index-th element from the picked file dataset. '''
        # ------------------------------ INPUTS ----------------------------
        print(index, len(self.id_list))
        # Compute one hot vector
        if self.one_hot:
            cls_type = self.type_list[index]
            assert (cls_type in ['Car', 'Pedestrian', 'Cyclist'])
            one_hot_vec = np.zeros((3))
            one_hot_vec[g_type2onehotclass[cls_type]] = 1

        # Get point cloud

        point_set = self.input_list[index]
        # Resample
        # print(point_set.shape[0])
        choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        point_set = point_set[choice, :]

        seg = self.label_list[index]
        seg = seg[choice]

        # RoI_para = np.zeros((16,7))

        radar_masks_ = np.asarray(self.radar_masks[index])
        radar_mask_frame = np.zeros((len(radar_masks_), self.npoints))
        RoI_para = np.asarray(self.RoI_parameters[index])
        for i in range(len(radar_masks_)):
            radar_mask_frame[i, :] = radar_masks_[i, choice]

        if len(radar_masks_) <= 16:
            choice = np.random.choice(len(radar_masks_),
                                      16 - len(radar_masks_), replace=True)
            choice = np.concatenate((np.arange(len(radar_masks_)), choice))


        elif len(radar_mask_frame) > 16:
            choice = np.random.choice(len(radar_mask_frame),
                                      16, replace=False)
        radar_mask_frame = radar_mask_frame[choice]
        RoI_para = RoI_para[choice]

        # ------------------------------ LABELS ----------------------------

        gt_corners = self.GT_boxes[index]
        # print("shape",point_set.shape)
        if (self.no_color):
            ret_pts_features = np.ones((len(point_set), 1))
            point_set = np.concatenate((point_set[:, 0:3], ret_pts_features), axis=1)

        if self.one_hot:
            return point_set, seg, one_hot_vec, radar_mask_frame, RoI_para, self.id_list[index], self.GT_boxes[index]
        else:
            return point_set, seg, radar_mask_frame, RoI_para, self.id_list[index], self.GT_boxes[index]


class Dataset_seg(object):
    ''' Dataset class for Frustum PointNets training/evaluation.
    Load prepared KITTI data from pickled files, return individual data element
    [optional] along with its annotations.
    '''

    def __init__(self, radar_file, database, npoints, split,
                 random_flip=False, random_shift=False, rotate_to_center=False,
                 overwritten_data_path=None, from_rgb_detection=False, one_hot=False, all_batches=False,
                 translate_radar_center=False,
                 store_data=False, proposals_3=False, no_color=False):  # ,generate_database=False):
        '''
        Input:
            npoints: int scalar, number of points for frustum point cloud.
            split: string, train or val
            random_flip: bool, in 50% randomly flip the point cloud
                in left and right (after the frustum rotation if any)
            random_shift: bool, if True randomly shift the point cloud
                back and forth by a random distance
            rotate_to_center: bool, whether to do frustum rotation
            overwritten_data_path: string, specify pickled file path.
                if None, use default path (with the split)
            from_rgb_detection: bool, if True we assume we do not have
                groundtruth, just return data elements.
            one_hot: bool, if True, return one hot vector
        '''
        self.no_color = no_color
        self.translate_to_radar_center = translate_radar_center
        self.dataset_kitti = KittiDataset(radar_file, root_dir='/root/frustum-pointnets_RSC/dataset/', dataset=database,
                                          mode='TRAIN',
                                          split=split)
        self.all_batches = all_batches
        self.npoints = npoints
        self.random_flip = random_flip
        self.random_shift = random_shift
        self.rotate_to_center = rotate_to_center
        self.one_hot = one_hot
        self.pc_lidar_list = []
        if (proposals_3):
            box_number = 'threeboxes'
        else:
            box_number = 'one_boxes'
        if overwritten_data_path is None:
            overwritten_data_path = os.path.join('/root/frustum-pointnets_RSC_RADAR_fil_PC_batch/',
                                                 'dataset/RSC/radar_' + box_number + ('_%s.pickle' % (split)))
        output_filename = overwritten_data_path
        self.from_rgb_detection = from_rgb_detection

        # list = os.listdir("/root/3D_BoundingBox_Annotation_Tool_3D_BAT/input/NuScenes/ONE/pointclouds_Radar")
        self.id_list = self.dataset_kitti.sample_id_list[0:42]
        self.idx_batch = self.id_list
        batch_list = []
        self.radar_OI = []
        self.batch_size = []
        batch_list = []
        self.input_list = []
        self.radar_masks = []
        self.label_list = []
        self.box3d_list = []
        self.radar_point_list = []
        self.indice_box = []
        self.type_list = []
        self.RoI_parameters = []
        self.GT_boxes = []
        accuracy_list = []
        frame_true_50 = []
        frame_true_70 = []
        if store_data:
            for i in range(len(self.id_list)):
                print(self.id_list[i])
                pc_radar = self.dataset_kitti.get_radar(self.id_list[i])
                pc_lidar = self.dataset_kitti.get_lidar(self.id_list[i])

                radar_mask, radar_mask_list, RoI_boxes_3d = get_radar_pc_mask(pc_lidar, pc_radar)

                radar_idx = np.argwhere(radar_mask == 1)
                pc_fil = pc_lidar[radar_idx.reshape(-1)]
                radar_masks_frame = []
                for k in range(len(radar_mask_list)):
                    radar_masks_frame.append(radar_mask_list[k][radar_idx.reshape(-1)])
                cls_label = np.zeros((pc_fil.shape[0]), dtype=np.int32)
                gt_obj_list = self.dataset_kitti.filtrate_objects(
                    self.dataset_kitti.get_label(self.id_list[i]))
                gt_boxes3d = kitti_utils.objs_to_boxes3d(gt_obj_list)
                gt_corners = kitti_utils.boxes3d_to_corners3d(gt_boxes3d, transform=False)
                cls_label_gt = np.zeros((pc_lidar.shape[0]), dtype=np.int32)
                for k in range(gt_boxes3d.shape[0]):
                    box_corners = gt_corners[k]
                    fg_pt_flag = kitti_utils.in_hull(pc_lidar[:, 0:3], box_corners)
                    cls_label_gt[fg_pt_flag] = 1
                # gt_in_radar_masks = np.logical_and(cls_label_gt,radar_mask)
                self.input_list.append(pc_lidar)
                batch_list.append(self.id_list[i])
                self.label_list.append(cls_label_gt)
                self.type_list.append("Pedestrian")
                # self.radar_masks.append(radar_masks_frame)
                # self.RoI_parameters.append(RoI_boxes_3d)
                self.GT_boxes.append(gt_corners)
            self.id_list = batch_list
            print("id_list", len(self.id_list))
            print("self.input_list", len(self.input_list))
            # print("mean accuracy:",np.mean(accuracy_list))
            # print("gt ratio 50:",np.mean(frame_true_50))
            # print("gt ratio 70:", np.mean(frame_true_70))
            # print("average nbr point cloud", np.sum(self.input_list[:,0])/len(self.id_list))

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        ''' Get index-th element from the picked file dataset. '''
        # ------------------------------ INPUTS ----------------------------

        # Compute one hot vector
        if self.one_hot:
            cls_type = self.type_list[index]
            assert (cls_type in ['Car', 'Pedestrian', 'Cyclist'])
            one_hot_vec = np.zeros((3))
            one_hot_vec[g_type2onehotclass[cls_type]] = 1

        # Get point cloud

        point_set = self.input_list[index]
        # Resample
        # print(point_set.shape[0])
        choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        point_set = point_set[choice, :]
        # radar_mask_frame = []
        # for i in range(len(self.radar_masks[index])):
        #    radar_mask = np.asarray(self.radar_masks[index][i])
        #    radar_mask_frame.append(radar_mask[choice])
        # RoI_para = self.RoI_parameters[index]
        # ------------------------------ LABELS ----------------------------
        seg = self.label_list[index]
        seg = seg[choice]

        gt_corners = self.GT_boxes[index]
        # print("shape",point_set.shape)
        if (self.no_color):
            ret_pts_features = np.ones((len(point_set), 1))
            point_set = np.concatenate((point_set[:, 0:3], ret_pts_features), axis=1)

        if self.one_hot:
            return point_set, seg, one_hot_vec, self.id_list[index]
        else:
            return point_set, seg, self.id_list[index]


def get_bins_in_RRoI(point_set, Radar_roi):
    centers = []
    bin_pc = []
    trans = np.array([Radar_roi[0], Radar_roi[1], Radar_roi[2]])
    pc = point_set - trans
    pc = rotate_pc_along_y(pc, Radar_roi[6])
    min = np.array([np.min(pc[:, 0]), np.min(pc[:, 1]), np.min(pc[:, 2])])
    max = np.array([np.max(pc[:, 0]), np.max(pc[:, 1]), np.max(pc[:, 2])])
    corners = corneers_from_minmax(min, max)
    center = (min + max) / 2.0
    l = abs(max[2] - min[2])
    h = abs(max[1] - min[1])
    center_1 = center
    center_2 = center
    w = 1.0 / 8.0
    ds = 0
    boxes_1 = get_3d_box((h, w, l), 0.0, center)
    fg_pt_flag_1 = kitti_utils.in_hull(pc[:, 0:3], boxes_1)
    if (np.count_nonzero(fg_pt_flag_1 == 1) > 0):
        pc_1 = pc[fg_pt_flag_1, :]
        bin_pc.append(pc_1)
        centers.append(center)
    else:
        bin_pc.append(np.array([]))
        centers.append(center)
    size = [h, w, l]
    while center_2[0] < max[0]:
        center_1 = [center_1[0] - 1.0 / 8.0, center_1[1], center_1[2]]
        center_2 = [center_2[0] + 1.0 / 8.0, center_2[1], center_2[2]]
        boxes_1 = get_3d_box((h, w, l), 0.0, center_1)
        boxes_2 = get_3d_box((h, w, l), 0.0, center_2)
        fg_pt_flag_1 = kitti_utils.in_hull(pc[:, 0:3], boxes_1)
        fg_pt_flag_2 = kitti_utils.in_hull(pc[:, 0:3], boxes_2)
        if np.count_nonzero(fg_pt_flag_1 == 1) > 0:
            pc_1 = pc[fg_pt_flag_1, :]
            bin_pc.append(pc_1)
            centers.append(center_1)
        else:
            bin_pc.append(np.array([]))
            centers.append(center_1)

        if np.count_nonzero(fg_pt_flag_2 == 1) > 0:
            pc_2 = pc[fg_pt_flag_2, :]
            bin_pc.insert(0, pc_2)
            centers.insert(0, center_2)
        else:
            bin_pc.insert(0, np.array([]))
            centers.insert(0, center_2)
        fg_pt_flag = np.logical_or(fg_pt_flag_1, fg_pt_flag_2)
        pc = pc[~fg_pt_flag, :]
    return bin_pc, centers, size, trans


def local_min_method(bin_pc, centers, size, radar_angle, trans):
    print(len(bin_pc), len(centers))
    bin_y_max = []
    for i in range(len(bin_pc)):
        if (bin_pc[i].size == 0):
            bin_y_max.append(centers[i][1] + size[0] / 2)
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

                corners = inverse_rotate_pc_along_y(corners, radar_angle)
                corners = corners + trans
                pc = inverse_rotate_pc_along_y(pc_AB, radar_angle)
                pc = pc + trans
                pc_AB_list.append(pc)
                corners_AB.append(corners)
    return pc_AB_list, corners_AB


def divide_in_n_AB(bin_pc, n):
    pc_AB_list = []
    for i in range(0, len(bin_pc) - n, 1):
        pc_AB = np.empty([0, 3])
        # print(len(bin_pc))
        # print(i,i+n)
        for j in range(i, i + n):
            # print(j)
            # print(bin_pc[j].size)
            if bin_pc[j].size != 0:
                pc_AB = np.concatenate((pc_AB, bin_pc[j]))
        pc_AB_list.append(pc_AB)
    return pc_AB_list


def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape) - 1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape) - 1, keepdims=True)
    return probs


def iterative_method(bin_pc, centers, size, radar_angle, trans):
    pc_AB_list = []
    corners_AB = []
    for i in range(3, 8):
        pc_AB_ = divide_in_n_AB(bin_pc, i)
        for pc_ in pc_AB_:
            if (len(pc_) > 0):
                min = np.array([np.min(pc_[:, 0]), np.min(pc_[:, 1]), np.min(pc_[:, 2])])
                max = np.array([np.max(pc_[:, 0]), np.max(pc_[:, 1]), np.max(pc_[:, 2])])
                corners = corneers_from_minmax(min, max)
                center = (min + max) / 2.0

                corners = inverse_rotate_pc_along_y(corners, radar_angle)
                corners = corners + trans
                corners_AB.append(corners)
                pc = inverse_rotate_pc_along_y(pc_, radar_angle)
                pc = pc + trans

                pc_AB_list.append(pc)
    return pc_AB_list, corners_AB


def NMS(pred_box_frame, IoU_frame, score_list_frame, gt_ids):
    bboxes_frame = []
    score_new_frame = []
    iou_new_frame = []
    id_new_frame = []
    indices_frame = []

    # estimate corners for all detections in box
    # estimate 3DIoU for a box with other boxes for a batch

    bboxes = []
    score_list = []
    id_list = []
    indice = []
    iou_prov = []
    gt_list = []

    ind_sort = np.argsort([x for x in score_list_frame[:, 1]])

    #print("ind_sort", ind_sort)
    for i in range(len(pred_box_frame)):
        bbox = pred_box_frame[ind_sort[i]]
        flag = 1
        for k in range(i + 1, len(pred_box_frame)):
            if (np.array_equal(bbox, pred_box_frame[ind_sort[k]])):
                flag = -1
                break
            # print("index ",ind_sort[i],score_list_frame[ind_sort[i]], "index _comp: ", ind_sort[k], score_list_frame[ind_sort[k]],"IoU: ",box3d_iou(bbox,pred_box_frame[ind_sort[k]]))
            if box3d_iou(bbox, pred_box_frame[ind_sort[k]])[1] > 0.65:
                flag = -1
                break
        if flag == 1:
            bboxes.append(bbox)
            indice.append(ind_sort[i])
            iou_prov.append(IoU_frame[ind_sort[i]])
            score_list.append(score_list_frame[ind_sort[i],:])
            gt_list.append(gt_ids[ind_sort[i]])

        # print("boxes size:", len(bboxes))

    return bboxes, score_list, iou_prov, indice, gt_list



def expand_cordinates(corners,width_plus,length_plus):
    min_point = np.amin(corners,axis=0)
    max_point = np.amax(corners,axis=0)
    center= (min_point + max_point)/2
    argmax_z = np.argmax(corners[:,2])
    argmin_x = np.argmin(corners[:,0])
    argmax_x = np.argmax(corners[:,0])
    width = np.sqrt(pow(corners[argmax_z,2]-corners[argmax_x,2],2)+ pow(corners[argmax_z,0]-corners[argmax_x,0],2))
    length= np.sqrt(pow(corners[argmax_z,2]-corners[argmin_x,2],2)+ pow(corners[argmax_z,0]-corners[argmin_x,0],2))
    height = abs(np.min(corners[:,1]) - np.max(corners[:,1]))
    diffs = [corners[argmax_z,2]-corners[argmin_x,2] ,corners[argmax_z,0]-corners[argmax_x,2]]
    rotation = math.atan(diffs[0]/diffs[1])
    #print([height,width,length])
    width = width + width_plus
    length = length + length_plus
    new_corners = get_3d_box([height,width,length],-rotation,center)
    #print(width_plus,length_plus)
    #print(center,rotation,[height,width,length])
    return new_corners
def corners3d_to_corners2d(corners3d):
    corners2d = np.zeros((8 * 2), dtype=np.float32).reshape((8, 2))
    for i in range(8):
        corners2d[i, 0], corners2d[i, 1] = pcd_to_img(corners3d[i])
    return corners2d


def pcd_to_img(pt):
    #point = np.array([-pt[1], -pt[2], pt[0]])  # 0=x 1=y z=2
    point=pt
    trans_matrix = np.array(
        [[700.8319702148438, 0.0, 623.9869995117188, 0.0], [0.0, 700.8319702148438, 362.1520080566406, 0.0],
         [0.0, 0.0, 1.0, 0.0]])
    px = point[0] * trans_matrix[0][0] / point[2] + trans_matrix[0][2]
    py = point[1] * trans_matrix[1][1] / point[2] + trans_matrix[1][2]
    # print "coordinate:", py,px
    return px-15, py
def draw_3dboxes(image,corners3d,color):

    for j in range(len(corners3d)):
        corner3d = corners3d[j]
        corner2d = corners3d_to_corners2d(corner3d)
        #s = np.argsort(corners3d[0, :, 2])
        #corners2d = corner2d[s]
        for i in range(4):
            point_1_ = corner2d[2 * i]
            point_2_ = corner2d[2 * i + 1]
            image = cv2.line(image, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), color, 2)
        #center = (
        #int((point_1_[0] + point_2_[0]) / 2), int((point_1_[1]+point_2_[1]) / 2))
        #font = cv2.FONT_HERSHEY_SIMPLEX
        #fontScale = 1
        #thickness = 2
        #img_k = cv2.putText(image, '%d' % i, center, font, fontScale, (0, 0, 255), thickness, cv2.LINE_AA)
        # for i in range(4):
        #    point_1_ = corner2d[2 * i + 1]
        #    point_2_ = corner2d[(2*i +2) % 8]
        #    print("2nd round", i,(i + 1) % 8)
        #    image = cv2.line(image, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), (255, 0, 0), 1)

        for i in range(4):
            point_1_ = corner2d[i]
            point_2_ = corner2d[i + 4]
            image = cv2.line(image, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), color, 1)

        for i in range(0, 5, 4):
            for k in range(2):
                point_1_ = corner2d[i + k]
                point_2_ = corner2d[i + 3 - k]
                image = cv2.line(image, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), color, 1)
    return image
class RadarDataset_bbox_CLS(object):
    ''' Dataset class for Frustum PointNets training/evaluation.
    Load prepared KITTI data from pickled files, return individual data element
    [optional] along with its annotations.
    '''

    def __init__(self, radar_file, database, npoints, split,
                 random_flip=False, random_shift=False, rotate_to_center=False,
                 overwritten_data_path=None, from_rgb_detection=False, one_hot=False, all_batches=False,
                 translate_radar_center=False,
                 store_data=False, proposals_3=False, no_color=False):  # ,generate_database=False):
        '''
        Input:
            npoints: int scalar, number of points for frustum point cloud.
            split: string, train or val
            random_flip: bool, in 50% randomly flip the point cloud
                in left and right (after the frustum rotation if any)
            random_shift: bool, if True randomly shift the point cloud
                back and forth by a random distance
            rotate_to_center: bool, whether to do frustum rotation
            overwritten_data_path: string, specify pickled file path.
                if None, use default path (with the split)
            from_rgb_detection: bool, if True we assume we do not have
                groundtruth, just return data elements.
            one_hot: bool, if True, return one hot vector
        '''
        self.no_color = no_color
        self.translate_to_radar_center = translate_radar_center
        self.all_batches = all_batches
        self.npoints = 512
        self.random_flip = random_flip
        self.random_shift = random_shift
        self.rotate_to_center = rotate_to_center
        self.one_hot = one_hot
        self.input_list = []
        self.type_list = []
        self.heading_list = []
        self.size_list = []
        self.box3d_list = []
        self.AB_list = []
        self.type_list = []
        self.heading_list = []
        self.size_list = []
        self.AB = []
        self.batch_list = []
        self.indice_box = []

        with open("/home/amben/frustum-pointnets_RSC_RADAR_fil_PC_batch_para/dataset/RSC/" + split + "_seg_cls_iter_method.pickle",
                'rb') as fp:
            u = pickle._Unpickler(fp)
            u.encoding = 'latin1'
            # logits_roi = pickle.load(fp)
            #ids = pickle.load(fp)
            #self.segp_list = pickle.load(fp)
            #ab_boxes = pickle.load(fp)
            #ab_cls_list = pickle.load(fp)
            #self.ab_ids_list = pickle.load(fp)
            ids = u.load()
            self.segp_list = u.load()
            ab_boxes = u.load()
            ab_cls_list = u.load()
            self.ab_ids_list = u.load()
            #print(self.ids)
        recall=[]
        recall_NMS=[]
        accuracy=[]
        self.dataset_kitti = KittiDataset('pc_radar_2', dataset=database,
                                          root_dir='/home/amben/frustum-pointnets_RSC/dataset/',
                                          mode='TRAIN',
                                          split=split)
        self.ab_cls_list = np.zeros((len(ab_cls_list), 2))
        self.ab_boxes = np.zeros((len(ab_cls_list), 8, 3))
        for i in range(len(ab_cls_list)):
            self.ab_cls_list[i, :] = ab_cls_list[i]
            self.ab_boxes[i, :] = ab_boxes[i]
        self.ids = []
        for i in range(len(ids)):
            self.ids.append(int(ids[i][0]))
        len_PC = []
        for i in range(len(self.segp_list)):
            len_PC.append(len(self.segp_list[i]))
        print(split,np.unique(np.asarray(self.ids)))
        batch_list = []

        for i in range(len(self.segp_list)):
            ab_arg_frame = []
            #self.ids = np.asarray(self.ids)
            self.ab_ids_list = np.asarray(self.ab_ids_list)
            # ab_arg_frame= np.argwhere(self.ids[i]==self.ab_ids_list)[0]
            for j in range(len(self.ab_ids_list)):
                if (self.ids[i] == self.ab_ids_list[j]):
                    ab_arg_frame.append(j)
            if (len(ab_arg_frame) > 0):

                cls_frame = self.ab_cls_list[ab_arg_frame]
                ab_frame = self.ab_boxes[ab_arg_frame]

                # pc_radar = self.dataset_kitti.get_radar(self.ids[i])
                pc_seg = self.segp_list[i]

                #print("number of point clouds: ", len(pc_seg))
                gt_obj_list = self.dataset_kitti.filtrate_objects(
                    self.dataset_kitti.get_label(self.ids[i]))
                gt_boxes3d = kitti_utils.objs_to_boxes3d(gt_obj_list)
                gt_corners = kitti_utils.boxes3d_to_corners3d(gt_boxes3d, transform=False)

                iou = []
                gt_ids = []
                for m in range(len(cls_frame)):
                    iou_max = 0
                    gt_id = 0
                    for k in range(len(gt_corners)):
                        iou3d, iou2d = box3d_iou(ab_frame[m], gt_corners[k])
                        if iou3d > iou_max:
                            iou_max = iou3d
                    iou.append(iou_max)
                    if iou_max == 0:
                        gt_ids.append(10)
                    else:
                        gt_ids.append(gt_id)
                for m in range(len(gt_corners)):
                    iou_max = 0
                    for k in range(len(cls_frame)):
                        iou3d, iou2d = box3d_iou(ab_frame[k], gt_corners[m])
                        if iou3d > iou_max:
                            iou_max = iou3d
                    iou.append(iou_max)
                    if iou_max>0.35:
                        recall.append(1)
                    else:
                        recall.append(0)
                staert_time=time.time()
                bboxes, score_list, iou_prov, indices, gt_list = NMS(ab_frame, iou, cls_frame, gt_ids)
                NMS_cls_frame = cls_frame[indices]
                self.time=time.time()-staert_time
                for m in range(len(gt_corners)):
                    iou_max = 0
                    for k in range(len(bboxes)):
                        iou3d, iou2d = box3d_iou(bboxes[k], gt_corners[m])
                        if iou3d > iou_max:
                            iou_max = iou3d
                    iou.append(iou_max)
                    if iou_max>0.35:
                        recall_NMS.append(1)
                    else:
                        recall_NMS.append(0)


                if (split == "train"):
                    for n in range(len(score_list)):

                        if (gt_list[n] != 10):
                            pc = self.segp_list[i][:, 0:3]

                            ab_bbox =bboxes[n]

                            #get_box3d_center(()
                            for l in range(5):
                                ab_bbox_= expand_cordinates(ab_bbox,random.random()/2,random.random()/2)
                                cls_label_gt = np.zeros(len(pc))
                                fg_pt_flag = kitti_utils.in_hull(pc[:, 0:3], ab_bbox_)

                                cls_label_gt[fg_pt_flag] = 1
                                indices = np.argwhere(cls_label_gt == 1)
                                indices_=indices.reshape(-1)
                                if (len(indices_) == 0):
                                    continue
                                AB_pc = pc[indices_]
                                self.AB.append(AB_pc)
                                self.type_list.append("Pedestrian")
                                self.box3d_list.append(gt_corners[gt_list[n]])
                                self.AB_list.append(bboxes[n])
                                self.size_list.append(
                                    [gt_boxes3d[gt_list[n]][3], gt_boxes3d[gt_list[n]][4], gt_boxes3d[gt_list[n]][5]])
                                self.heading_list.append(gt_boxes3d[gt_list[n]][6])
                                self.batch_list.append(self.ids[i])
                                self.indice_box.append(n)

                elif(split == "val" or split == "test"):
                    # print proposals on image in ordered way
                    """score_list=np.array(score_list)
                    print(score_list)
                    pos_det_cls = softmax(score_list)
                    pos_det_cls=np.argmax(pos_det_cls,1)
                    print(pos_det_cls)
                    pos_indices=np.argwhere(pos_det_cls==1)
                    pos_indices=pos_indices.reshape(-1)
                    if len(pos_indices)>0:
                        Image_file = "/media/xivt/DB/image_left_rect/" + "%06d.jpg" % self.ids[i]
                        image = cv2.imread(Image_file, cv2.IMREAD_COLOR)
                        image = cv2.blur(image, (5, 5))
                        image = draw_3dboxes(image, gt_corners, (255, 0, 0))
                        removed_AB = np.delete(ab_frame, indices, axis=0)
                        image = draw_3dboxes(image, removed_AB, (0, 255, 0))
                        print(pos_indices)

                        removed_AB= np.delete(bboxes,pos_indices,axis=0)
                        print(removed_AB.shape)
                        image = draw_3dboxes(image, removed_AB, (0, 255, 0))
                        bboxes = np.asarray(bboxes)
                        bboxes = bboxes[pos_indices]
                        #gt_list = gt_list[pos_indices]
                        image = draw_3dboxes(image, bboxes, (0, 0, 255))
                        cv2.imwrite("/media/xivt/proposals_cls/proposals_iter_softmax/" + "%06d.jpg" % self.ids[i], image)"""

                    #for n in range(len(gt_list)):


                    if len(NMS_cls_frame) < 10:
                        Image_file = "/media/xivt/DB/image_left_rect/" + "%06d.jpg" % self.ids[i]
                        image = cv2.imread(Image_file, cv2.IMREAD_COLOR)
                        image = cv2.blur(image, (5, 5))
                        image = draw_3dboxes(image, gt_corners, (255, 0, 0))
                        removed_AB = np.delete(ab_frame, indices, axis=0)
                        image = draw_3dboxes(image, removed_AB, (0, 255, 0))


                        image = draw_3dboxes(image, bboxes, (0, 0, 255))
                        cv2.imwrite("/media/xivt/proposals_cls/proposals_iter_10_iou_065/" + "%06d.jpg" % self.ids[i], image)
                        for n in range(len(gt_list)):
                            if gt_list[n] != 10:
                                pc = self.segp_list[i][:, 0:3]
                                cls_label_gt = np.zeros(len(pc))
                                fg_pt_flag = kitti_utils.in_hull(pc[:, 0:3], bboxes[n])
                                cls_label_gt[fg_pt_flag] = 1
                                indices = np.argwhere(cls_label_gt == 1)
                                AB_pc = pc[indices.reshape(-1)]

                                self.AB.append(AB_pc)
                                self.type_list.append("Pedestrian")
                                self.box3d_list.append(gt_corners[gt_list[n]])
                                self.AB_list.append(bboxes[n])
                                self.size_list.append([gt_boxes3d[gt_list[n]][3], gt_boxes3d[gt_list[n]][4],
                                                       gt_boxes3d[gt_list[n]][5]])
                                self.heading_list.append(gt_boxes3d[gt_list[n]][6])
                                self.batch_list.append(self.ids[i])
                                self.indice_box.append(n)
                                accuracy.append(1.0)
                            else:
                                pc = self.segp_list[i][:, 0:3]
                                cls_label_gt = np.zeros(len(pc))
                                fg_pt_flag = kitti_utils.in_hull(pc[:, 0:3], bboxes[n])
                                cls_label_gt[fg_pt_flag] = 1
                                indices = np.argwhere(cls_label_gt == 1)
                                AB_pc = pc[indices.reshape(-1)]
                                box3d_center = np.random.rand(3) * (-10.0)
                                size = np.ones((3))
                                box3d = np.array(
                                    [[box3d_center[0], box3d_center[1], box3d_center[2], size[0], size[1],
                                      size[2], 0.0]])
                                corners_empty = kitti_utils.boxes3d_to_corners3d(box3d, transform=False)
                                self.AB.append(AB_pc)
                                self.type_list.append("Pedestrian")
                                self.box3d_list.append(corners_empty[0])
                                self.AB_list.append(bboxes[n])

                                self.size_list.append(size)
                                self.heading_list.append(0.0)
                                self.batch_list.append(self.ids[i])
                                self.indice_box.append(n)
                                accuracy.append(0.0)

                    else:
                        Image_file = "/media/xivt/DB/image_left_rect/" + "%06d.jpg" % self.ids[i]
                        image = cv2.imread(Image_file, cv2.IMREAD_COLOR)
                        image = cv2.blur(image, (5, 5))
                        image = draw_3dboxes(image, gt_corners, (255, 0, 0))

                        removed_AB = np.delete(ab_frame, indices, axis=0)
                        image = draw_3dboxes(image, removed_AB, (0, 255, 0))
                        indices_=[]
                        for n in range(len(gt_list)-1, len(gt_list) - 11, -1):
                            indices_.append(n)
                        removed_AB = np.delete(bboxes, indices, axis=0)
                        image = draw_3dboxes(image, removed_AB, (0, 255, 0))
                        indices_=np.asarray(indices_)
                        print(len(gt_list))
                        print(indices_)
                        bboxes=np.asarray(bboxes)
                        image = draw_3dboxes(image, bboxes[indices_], (0, 0, 255))
                        cv2.imwrite("/media/xivt/proposals_cls/proposals_iter_10_iou_065/" + "%06d.jpg" % self.ids[i], image)
                        for n in range(len(gt_list)-1, len(gt_list) - 11, -1):
                            if (gt_list[n] != 10):
                                pc = self.segp_list[i][:, 0:3]
                                cls_label_gt = np.zeros(len(pc))
                                fg_pt_flag = kitti_utils.in_hull(pc[:, 0:3], bboxes[n])
                                cls_label_gt[fg_pt_flag] = 1
                                indices = np.argwhere(cls_label_gt == 1)

                                AB_pc = pc[indices.reshape(-1)]
                                self.AB.append(AB_pc)
                                self.type_list.append("Pedestrian")
                                self.box3d_list.append(gt_corners[gt_list[n]])
                                self.AB_list.append(bboxes[n])
                                self.size_list.append([gt_boxes3d[gt_list[n]][3], gt_boxes3d[gt_list[n]][4],
                                                       gt_boxes3d[gt_list[n]][5]])
                                self.heading_list.append(gt_boxes3d[gt_list[n]][6])
                                self.batch_list.append(self.ids[i])
                                self.indice_box.append(n)
                                accuracy.append(1.0)

                            else:
                                pc = self.segp_list[i][:, 0:3]
                                cls_label_gt = np.zeros(len(pc))
                                fg_pt_flag = kitti_utils.in_hull(pc[:, 0:3], bboxes[n])
                                cls_label_gt[fg_pt_flag] = 1
                                indices = np.argwhere(cls_label_gt == 1)
                                AB_pc = pc[indices.reshape(-1)]
                                box3d_center = np.random.rand(3) * (-10.0)
                                size = np.ones((3))
                                box3d = np.array(
                                    [[box3d_center[0], box3d_center[1], box3d_center[2], size[0], size[1],
                                      size[2], 0.0]])
                                corners_empty = kitti_utils.boxes3d_to_corners3d(box3d, transform=False)
                                self.AB.append(AB_pc)
                                self.type_list.append("Pedestrian")
                                self.box3d_list.append(corners_empty[0])
                                self.AB_list.append(bboxes[n])

                                self.size_list.append(size)
                                self.heading_list.append(0.0)
                                self.batch_list.append(self.ids[i])
                                self.indice_box.append(n)
                                accuracy.append(0.0)

        self.id_list = self.batch_list
        print("recall: ", np.mean(recall))
        print("recall NMS:",np.mean(recall_NMS))
        print("accuracy NMS:",np.mean(accuracy))
        time.sleep(10)
        print(np.unique(self.id_list))


    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        ''' Get index-th element from the picked file dataset. '''
        # ------------------------------ INPUTS ----------------------------

        # Compute one hot vector
        # label_mask = self.batch_train[index]
        # rot_angle = self.get_center_view_rot_angle(index)

        # Compute one hot vector
        if self.one_hot:
            cls_type = self.type_list[index]
            assert (cls_type in ['Car', 'Pedestrian', 'Cyclist'])
            one_hot_vec = np.zeros((3))
            one_hot_vec[g_type2onehotclass[cls_type]] = 1

        # Get point cloud
        if self.rotate_to_center:
            point_set = self.get_center_view_point_set(index)
        else:
            point_set = self.AB[index]
        # Resample
        choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        point_set = point_set[choice, :]

        # ------------------------------ LABELS ----------------------------

        if (self.no_color):
            ret_pts_features = np.ones((len(point_set), 1))
            point_set = np.concatenate((point_set[:, 0:3], ret_pts_features), axis=1)
        point_set = point_set[:, 0:3]
        pc_orig = point_set
        # Get center point of 3D box
        if self.rotate_to_center:
            box3d_center = self.get_center_view_box3d_center(index)
            proposal_center = self.get_center_view_proposal(index)

        else:
            box3d_center = self.get_box3d_center(index)
            # proposal_center = self.radar_point_list[index]

        if self.translate_to_radar_center:
            box3d_center = box3d_center - proposal_center
            point_set[:, 0] = point_set[:, 0] - proposal_center[0]
            point_set[:, 1] = point_set[:, 1] - proposal_center[1]
            point_set[:, 2] = point_set[:, 2] - proposal_center[2]
        # Heading
        if self.rotate_to_center:
            heading_angle = self.heading_list[index] - rot_angle
        else:
            heading_angle = self.heading_list[index]

        # Size
        size_class, size_residual = size2class(self.size_list[index],
                                               self.type_list[index])
        # translate point cloud to mean center
        center_mean = [np.mean(point_set[:, 0]), np.mean(point_set[:, 1]), np.mean(point_set[:, 2])]
        point_set[:, 0] = point_set[:, 0] - center_mean[0]
        point_set[:, 1] = point_set[:, 1] - center_mean[1]
        point_set[:, 2] = point_set[:, 2] - center_mean[2]
        # translate GT box to mean center
        # box3d_center[0]=box3d_center[0]-center_mean[0]
        # box3d_center[1] = box3d_center[1] - center_mean[1]
        # box3d_center[2] = box3d_center[2] - center_mean[2]
        # Data Augmentation
        if self.random_flip:
            # note: rot_angle won't be correct if we have random_flip
            # so do not use it in case of random flipping.
            if np.random.random() > 0.5:  # 50% chance flipping
                point_set[:, 0] *= -1
                box3d_center[0] *= -1
                heading_angle = np.pi - heading_angle
        if self.random_shift:
            dist = np.sqrt(np.sum(box3d_center[0] ** 2 + box3d_center[1] ** 2))
            shift = np.clip(np.random.randn() * dist * 0.05, dist * 0.8, dist * 1.2)
            point_set[:, 2] += shift
            box3d_center[2] += shift

        angle_class, angle_residual = angle2class(heading_angle,
                                                  NUM_HEADING_BIN)
        # print("10 points",point_set[0:10])
        if self.one_hot:
            return point_set, center_mean, one_hot_vec, box3d_center, angle_class, angle_residual, \
                   size_class, size_residual, self.AB_list[index]
        else:
            return point_set, center_mean, box3d_center, angle_class, angle_residual, \
                   size_class, size_residual, self.AB_list[index]

    def get_center_view_rot_angle(self, index):
        ''' Get the frustum rotation angle, it isshifted by pi/2 so that it
        can be directly used to adjust GT heading angle '''
        return np.pi / 2.0 + self.frustum_angle_list[index]
        # return self.frustum_angle_list[index]

    def get_box3d_center(self, index):
        ''' Get the center (XYZ) of 3D bounding box. '''
        box3d_center = (self.box3d_list[index][0, :] + \
                        self.box3d_list[index][6, :]) / 2.0
        return box3d_center

    def get_center_view_box3d_center(self, index):
        ''' Frustum rotation of 3D bounding box center. '''
        box3d_center = (self.box3d_list[index][0, :] + \
                        self.box3d_list[index][6, :]) / 2.0
        return rotate_pc_along_y(np.expand_dims(box3d_center, 0), \
                                 self.get_center_view_rot_angle(index)).squeeze()

    def get_center_view_proposal(self, index):
        return rotate_pc_along_y(np.expand_dims(self.radar_point_list[index], 0), \
                                 self.get_center_view_rot_angle(index)).squeeze()

    def get_center_view_box3d(self, index):
        ''' Frustum rotation of 3D bounding box corners. '''
        box3d = self.box3d_list[index]
        box3d_center_view = np.copy(box3d)
        return rotate_pc_along_y(box3d_center_view, \
                                 self.get_center_view_rot_angle(index))

    def get_center_view_point_set(self, index):
        ''' Frustum rotation of point clouds.
        NxC points with first 3 channels as XYZ
        z is facing forward, x is left ward, y is downward
        '''
        # Use np.copy to avoid corrupting original data
        point_set = np.copy(self.input_list[index])
        return rotate_pc_along_y(point_set, \
                                 self.get_center_view_rot_angle(index))


class RadarDataset_bbox(object):
    ''' Dataset class for Frustum PointNets training/evaluation.
    Load prepared KITTI data from pickled files, return individual data element
    [optional] along with its annotations.
    '''

    def __init__(self, radar_file, database, npoints, split,
                 random_flip=False, random_shift=False, rotate_to_center=False,
                 overwritten_data_path=None, from_rgb_detection=False, one_hot=False, all_batches=False,
                 translate_radar_center=False,
                 store_data=False, proposals_3=False, no_color=False):  # ,generate_database=False):
        '''
        Input:
            npoints: int scalar, number of points for frustum point cloud.
            split: string, train or val
            random_flip: bool, in 50% randomly flip the point cloud
                in left and right (after the frustum rotation if any)
            random_shift: bool, if True randomly shift the point cloud
                back and forth by a random distance
            rotate_to_center: bool, whether to do frustum rotation
            overwritten_data_path: string, specify pickled file path.
                if None, use default path (with the split)
            from_rgb_detection: bool, if True we assume we do not have
                groundtruth, just return data elements.
            one_hot: bool, if True, return one hot vector
        '''
        self.no_color = no_color
        self.translate_to_radar_center = translate_radar_center
        self.all_batches = all_batches
        self.npoints = 512
        self.random_flip = random_flip
        self.random_shift = random_shift
        self.rotate_to_center = rotate_to_center
        self.one_hot = one_hot
        self.input_list = []
        self.type_list = []
        self.heading_list = []
        self.size_list = []
        self.box3d_list = []
        self.AB_list = []
        self.type_list = []
        self.heading_list = []
        self.size_list = []
        self.AB = []
        self.batch_list = []
        self.indice_box = []
        with open(
                "/home/amben/frustum-pointnets_RSC_RADAR_fil_PC_batch_para/dataset/RSC/seg_rois_" + split + "_1.pickle",
                'rb') as fp:
             u = pickle._Unpickler(fp)
             u.encoding = 'latin1'
            # logits_roi = pickle.load(fp)
            #self.ids = pickle.load(fp)
            #self.segp_list = pickle.load(fp)
             self.ids = u.load()
             self.segp_list = u.load()
            #print(self.ids)
        print(self.ids)
        self.dataset_kitti = KittiDataset('pc_radar_2', dataset=database,
                                          root_dir='/home/amben/frustum-pointnets_RSC/dataset/',
                                          mode='TRAIN',
                                          split=split)
        print(len(self.dataset_kitti.sample_id_list))
        print(len(self.ids))
        print("segh", len(self.segp_list))
        len_PC = []
        for i in range(len(self.segp_list)):
            len_PC.append(len(self.segp_list[i]))
        # print(self.segp_list)
        # print(self.segp_list)
        # self.segp_list=self.segp_list[4:5]
        # self.ids= self.ids[4:5]
        #ids=[]
        #for i in range(len(self.ids)):
        #    ids.append(int(self.ids[i][0]))
        #self.ids=ids
        #self.ids=np.asarray(self.ids)

        recall=[]
        for i in range(len(self.segp_list)):

            print(self.ids[i])
            # print(self.segp_list[i])
            pc_radar = self.dataset_kitti.get_radar(self.ids[i])
            pc_seg = self.segp_list[i]

            print("number of point clouds: ", len(pc_seg))
            gt_obj_list = self.dataset_kitti.filtrate_objects(
                self.dataset_kitti.get_label(self.ids[i]))
            gt_boxes3d = kitti_utils.objs_to_boxes3d(gt_obj_list)
            gt_corners = kitti_utils.boxes3d_to_corners3d(gt_boxes3d, transform=False)
            cls_label = np.zeros((pc_seg.shape[0]), dtype=np.int32)
            for k in range(gt_boxes3d.shape[0]):
                box_corners = gt_corners[k]
                fg_pt_flag = kitti_utils.in_hull(pc_seg[:, 0:3], box_corners)
                cls_label[fg_pt_flag] = 1

            radar_mask, radar_mask_list, RoI_boxes_3d = get_radar_pc_mask(pc_seg, pc_radar)
            zs = 0
            print("len(radar_mask_list)",len(radar_mask_list))
            for j in range(len(radar_mask_list)):

                if np.count_nonzero(radar_mask_list[j] == 1) < 10:
                    print("no pc extracted", self.ids[i])
                    continue
                else:
                    #print("radar_mask_list", np.count_nonzero(radar_mask_list[j] == 1))
                    radar_idx = np.argwhere(radar_mask_list[j] == 1)
                    pc_fil = pc_seg[radar_idx.reshape(-1)]
                    #print(len(pc_seg))
                    pc_fil = pc_fil[:, 0:3]
                    print(len(pc_fil))
                    gt_obj_list = self.dataset_kitti.filtrate_objects(
                        self.dataset_kitti.get_label(self.ids[i]))
                    gt_boxes3d = kitti_utils.objs_to_boxes3d(gt_obj_list)
                    gt_corners = kitti_utils.boxes3d_to_corners3d(gt_boxes3d, transform=False)
                    cls_label = np.zeros((pc_fil.shape[0]), dtype=np.int32)
                    for k in range(gt_boxes3d.shape[0]):
                        box_corners = gt_corners[k]
                        fg_pt_flag = kitti_utils.in_hull(pc_fil[:, 0:3], box_corners)
                        cls_label[fg_pt_flag] = 1

                    if (float(np.count_nonzero(
                            cls_label == 1)) > 50 and split == "train") or split == "val" or split == "test":
                        start_t_box = time.time()
                        bin_pc, centers, size, trans = get_bins_in_RRoI(pc_fil, RoI_boxes_3d[j])
                        AB_pc, AB_corners = local_min_method(bin_pc, centers, size, RoI_boxes_3d[j][6], trans)
                        #AB_pc, AB_corners = iterative_method(bin_pc, centers, size, RoI_boxes_3d[j][6], trans)

                        end_t_box = time.time()
                        self.time=end_t_box-start_t_box
                        """fig = mlab.figure(figure=None, bgcolor=(0.4, 0.4, 0.4), fgcolor=None, engine=None,
                                          size=(1000, 500))
                        mlab.points3d(pc_fil[:, 0], pc_fil[:, 1], pc_fil[:, 2], cls_label, mode='point',
                                      colormap='gnuplot', scale_factor=1,
                                      figure=fig)
                        mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2, figure=fig)
                        draw_gt_boxes3d(gt_corners, fig, color=(1, 0, 0))
                        draw_gt_boxes3d(AB_corners, fig, color=(0, 0, 1))
                        mlab.orientation_axes()
                        raw_input()"""
                        Image_file = "/media/xivt/DB/image_left_rect/" + "%06d.jpg" % self.ids[i]
                        image = cv2.imread(Image_file, cv2.IMREAD_COLOR)
                        image = cv2.blur(image, (5, 5))
                        image = draw_3dboxes(image, gt_corners, (255, 0, 0))
                        image = draw_3dboxes(image, AB_corners, (0, 255, 0))
                        cv2.imwrite("/media/xivt/proposals_cls/proposals_iter_wo_cls/" + "%06d.jpg" % self.ids[i],
                                    image)
                        for m in range(len(gt_corners)):
                            iou_max = 0
                            for k in range(len(AB_corners)):
                                iou3d, iou2d = box3d_iou(AB_corners[k], gt_corners[m])
                                if iou3d > iou_max:
                                    iou_max = iou3d
                            #iou.append(iou_max)
                            if iou_max > 0.35:
                                recall.append(1)
                            else:
                                recall.append(0)
                        for k in range(len(AB_corners)):
                            for m in range(len(gt_corners)):
                                # print("corners AB", AB_corners[k])
                                # print("gt_corners[m]", gt_corners[m])
                                if len(np.unique(AB_corners[k][:, 0])) == 1:
                                    continue
                                iou_3d, iou_2d = box3d_iou(AB_corners[k], gt_corners[m])
                                #print(iou_3d)
                                if iou_3d > 0.0:
                                    """fig = mlab.figure(figure=None, bgcolor=(0.4, 0.4, 0.4), fgcolor=None, engine=None,
                                                      size=(1000, 500))
                                    mlab.points3d(AB_pc[k][:, 0], AB_pc[k][:, 1], AB_pc[k][:, 2], mode='point',
                                                  colormap='gnuplot', scale_factor=1,
                                                  figure=fig)
                                    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2, figure=fig)
                                    draw_gt_boxes3d([gt_corners[m]], fig, color=(1, 0, 0))
                                    draw_gt_boxes3d([AB_corners[k]], fig, color=(0, 0, 1))
                                    mlab.orientation_axes()
                                    raw_input()"""
                                    if split=="train":
                                        ab_bbox=AB_corners[k]
                                        for l in range(5):
                                            ab_bbox_ = expand_cordinates(ab_bbox, random.random() / 2,
                                                                         random.random() / 2)
                                            cls_label_gt = np.zeros(len(pc_fil))
                                            fg_pt_flag = kitti_utils.in_hull(pc_fil[:, 0:3], ab_bbox_)

                                            cls_label_gt[fg_pt_flag] = 1
                                            indices = np.argwhere(cls_label_gt == 1)
                                            indices_ = indices.reshape(-1)
                                            if (len(indices_) == 0):
                                                continue
                                            AB_pc = pc_fil[indices_]
                                            self.AB.append(AB_pc)
                                            self.type_list.append("Pedestrian")
                                            self.box3d_list.append(gt_corners[m])
                                            self.AB_list.append(ab_bbox_)
                                            self.size_list.append(
                                                [gt_boxes3d[m][3], gt_boxes3d[m][4],
                                                 gt_boxes3d[m][5]])
                                            self.heading_list.append(gt_boxes3d[m][6])

                                            self.batch_list.append(self.ids[i])
                                            self.indice_box.append(m)

                                    else:
                                        self.AB.append(AB_pc[k])
                                        self.type_list.append("Pedestrian")
                                        self.box3d_list.append(gt_corners[m])
                                        self.AB_list.append(AB_corners[k])
                                        self.size_list.append([gt_boxes3d[m][3], gt_boxes3d[m][4], gt_boxes3d[m][5]])
                                        self.heading_list.append(gt_boxes3d[m][6])
                                        self.batch_list.append(self.ids[i])
                                        self.indice_box.append(m)
                                elif iou_3d == 0.0 and (split == 'val' or split == 'test'):
                                    self.AB.append(AB_pc[k])
                                    box3d_center = np.random.rand(3) * (-10.0)
                                    size = np.ones((3))
                                    box3d = np.array(
                                        [[box3d_center[0], box3d_center[1], box3d_center[2], size[0], size[1],
                                          size[2], 0.0]])
                                    corners_empty = kitti_utils.boxes3d_to_corners3d(box3d, transform=False)

                                    self.type_list.append("Pedestrian")
                                    self.box3d_list.append(corners_empty[0])
                                    self.AB_list.append(AB_corners[k])

                                    self.size_list.append(size)
                                    self.heading_list.append(0.0)
                                    self.batch_list.append(self.ids[i])
                                    self.indice_box.append(10)

        self.id_list = self.batch_list
        print("recall",np.mean(recall) )


    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        ''' Get index-th element from the picked file dataset. '''
        # ------------------------------ INPUTS ----------------------------

        # Compute one hot vector
        # label_mask = self.batch_train[index]
        # rot_angle = self.get_center_view_rot_angle(index)

        # Compute one hot vector
        if self.one_hot:
            cls_type = self.type_list[index]
            assert (cls_type in ['Car', 'Pedestrian', 'Cyclist'])
            one_hot_vec = np.zeros((3))
            one_hot_vec[g_type2onehotclass[cls_type]] = 1

        # Get point cloud
        if self.rotate_to_center:
            point_set = self.get_center_view_point_set(index)
        else:
            point_set = self.AB[index]
        # Resample
        choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        point_set = point_set[choice, :]

        # ------------------------------ LABELS ----------------------------

        if (self.no_color):
            ret_pts_features = np.ones((len(point_set), 1))
            point_set = np.concatenate((point_set[:, 0:3], ret_pts_features), axis=1)
        point_set = point_set[:, 0:3]
        pc_orig = point_set
        # Get center point of 3D box
        if self.rotate_to_center:
            box3d_center = self.get_center_view_box3d_center(index)
            proposal_center = self.get_center_view_proposal(index)

        else:
            box3d_center = self.get_box3d_center(index)
            # proposal_center = self.radar_point_list[index]

        if self.translate_to_radar_center:
            box3d_center = box3d_center - proposal_center
            point_set[:, 0] = point_set[:, 0] - proposal_center[0]
            point_set[:, 1] = point_set[:, 1] - proposal_center[1]
            point_set[:, 2] = point_set[:, 2] - proposal_center[2]
        # Heading
        if self.rotate_to_center:
            heading_angle = self.heading_list[index] - rot_angle
        else:
            heading_angle = self.heading_list[index]

        # Size
        size_class, size_residual = size2class(self.size_list[index],
                                               self.type_list[index])
        # translate point cloud to mean center
        center_mean = [np.mean(point_set[:, 0]), np.mean(point_set[:, 1]), np.mean(point_set[:, 2])]
        point_set[:, 0] = point_set[:, 0] - center_mean[0]
        point_set[:, 1] = point_set[:, 1] - center_mean[1]
        point_set[:, 2] = point_set[:, 2] - center_mean[2]
        # translate GT box to mean center
        # box3d_center[0]=box3d_center[0]-center_mean[0]
        # box3d_center[1] = box3d_center[1] - center_mean[1]
        # box3d_center[2] = box3d_center[2] - center_mean[2]
        # Data Augmentation
        if self.random_flip:
            # note: rot_angle won't be correct if we have random_flip
            # so do not use it in case of random flipping.
            if np.random.random() > 0.5:  # 50% chance flipping
                point_set[:, 0] *= -1
                box3d_center[0] *= -1
                heading_angle = np.pi - heading_angle
        if self.random_shift:
            dist = np.sqrt(np.sum(box3d_center[0] ** 2 + box3d_center[1] ** 2))
            shift = np.clip(np.random.randn() * dist * 0.05, dist * 0.8, dist * 1.2)
            point_set[:, 2] += shift
            box3d_center[2] += shift

        angle_class, angle_residual = angle2class(heading_angle,
                                                  NUM_HEADING_BIN)
        # print("10 points",point_set[0:10])
        if self.one_hot:
            return point_set, center_mean, one_hot_vec, box3d_center, angle_class, angle_residual, \
                   size_class, size_residual, self.AB_list[index]
        else:
            return point_set, center_mean, box3d_center, angle_class, angle_residual, \
                   size_class, size_residual, self.AB_list[index]

    def get_center_view_rot_angle(self, index):
        ''' Get the frustum rotation angle, it isshifted by pi/2 so that it
        can be directly used to adjust GT heading angle '''
        return np.pi / 2.0 + self.frustum_angle_list[index]
        # return self.frustum_angle_list[index]

    def get_box3d_center(self, index):
        ''' Get the center (XYZ) of 3D bounding box. '''
        box3d_center = (self.box3d_list[index][0, :] + \
                        self.box3d_list[index][6, :]) / 2.0
        return box3d_center

    def get_center_view_box3d_center(self, index):
        ''' Frustum rotation of 3D bounding box center. '''
        box3d_center = (self.box3d_list[index][0, :] + \
                        self.box3d_list[index][6, :]) / 2.0
        return rotate_pc_along_y(np.expand_dims(box3d_center, 0), \
                                 self.get_center_view_rot_angle(index)).squeeze()

    def get_center_view_proposal(self, index):
        return rotate_pc_along_y(np.expand_dims(self.radar_point_list[index], 0), \
                                 self.get_center_view_rot_angle(index)).squeeze()

    def get_center_view_box3d(self, index):
        ''' Frustum rotation of 3D bounding box corners. '''
        box3d = self.box3d_list[index]
        box3d_center_view = np.copy(box3d)
        return rotate_pc_along_y(box3d_center_view, \
                                 self.get_center_view_rot_angle(index))

    def get_center_view_point_set(self, index):
        ''' Frustum rotation of point clouds.
        NxC points with first 3 channels as XYZ
        z is facing forward, x is left ward, y is downward
        '''
        # Use np.copy to avoid corrupting original data
        point_set = np.copy(self.input_list[index])
        return rotate_pc_along_y(point_set, \
                                 self.get_center_view_rot_angle(index))


# ----------------------------------
# Helper functions for evaluation
# ----------------------------------

def get_3d_box(box_size, heading_angle, center):
    ''' Calculate 3D bounding box corners from its parameterization.

    Input:
        box_size: tuple of (l,w,h)
        heading_angle: rad scalar, clockwise from pos x axis
        center: tuple of (x,y,z)
    Output:
        corners_3d: numpy array of shape (8,3) for 3D box cornders
    '''

    def roty(t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]])

    R = roty(heading_angle)

    h, w, l = box_size
    x_corners = [w / 2, w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2];
    y_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2];
    z_corners = [l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2];
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + center[0]
    corners_3d[1, :] = corners_3d[1, :] + center[1]
    corners_3d[2, :] = corners_3d[2, :] + center[2]
    corners_3d = np.transpose(corners_3d)
    return corners_3d


def get_3d_box_batch(center, angle_class, angle_res, \
                     size_class, size_res, rot_angle):
    ''' Calculate 3D bounding box corners from its parameterization.

    Input:
        box_size: tuple of (l,w,h)
        heading_angle: rad scalar, clockwise from pos x axis
        center: tuple of (x,y,z)
    Output:
        corners_3d: numpy array of shape (8,3) for 3D box cornders
    '''

    def roty(t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]])

    R = roty(class2angle(angle_class, angle_res, NUM_HEADING_BIN) + rot_angle)
    l, w, h = class2size(size_class, size_res)
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2];
    y_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2];
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2];
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + center[0];
    corners_3d[1, :] = corners_3d[1, :] + center[1];
    corners_3d[2, :] = corners_3d[2, :] + center[2];
    corners_3d = np.transpose(corners_3d)
    return corners_3d


def compute_box3d_iou_batch_test(center_pred,
                                 heading_class, heading_residual,
                                 size_class, size_residual,
                                 corners_3d_label):
    ''' Compute 3D bounding box IoU from network output and labels.
    All inputs are numpy arrays.

    Inputs:
        center_pred: (B,3)
        heading_logits: (B,NUM_HEADING_BIN)
        heading_residuals: (B,NUM_HEADING_BIN)
        size_logits: (B,NUM_SIZE_CLUSTER)
        size_residuals: (B,NUM_SIZE_CLUSTER,3)
        center_label: (B,3)
        heading_class_label: (B,)
        heading_residual_label: (B,)
        size_class_label: (B,)
        size_residual_label: (B,3)
    Output:
        iou2ds: (B,) birdeye view oriented 2d box ious
        iou3ds: (B,) 3d box ious
    '''
    batch_size = heading_class.shape[0]
    # heading_class = np.argmax(heading_logits, 1)  # B
    # heading_residual = np.array([heading_residuals[i, heading_class[i]] \
    #                             for i in range(batch_size)])  # B,
    # size_class = np.argmax(size_logits, 1)  # B
    # size_residual = np.vstack([size_residuals[i, size_class[i], :] \
    #                           for i in range(batch_size)])
    # print(heading_class_label, heading_residual_label)
    iou2d_list = []
    iou3d_list = []

    for i in range(batch_size):
        heading_angle = class2angle(heading_class[i],
                                    heading_residual[i], NUM_HEADING_BIN)
        box_size = class2size(size_class[i], size_residual[i])
        corners_3d = get_3d_box(box_size, heading_angle, center_pred[i])

        # heading_angle_label = class2angle(heading_class_label[i],
        #                                  heading_residual_label[i], NUM_HEADING_BIN)
        # box_size_label = class2size(size_class_label[i], size_residual_label[i])
        # corners_3d_label = get_3d_box(box_size_label,
        #                              heading_angle_label, center_label[i])
        for j in range(len(corners_3d_label)):
            iou_3d, iou_2d = box3d_iou(corners_3d, corners_3d_label[j])
            print(corners_3d, corners_3d_label[j])
            print("iou_3d:", iou_3d, "iou_2d", iou_2d)
            iou3d_list.append(iou_3d)
            iou2d_list.append(iou_2d)
    return np.array(iou2d_list, dtype=np.float32), \
           np.array(iou3d_list, dtype=np.float32)


def compute_box3d_iou_batch(logits, center_pred,
                            heading_logits, heading_residuals,
                            size_logits, size_residuals,
                            center_label,
                            heading_class_label, heading_residual_label,
                            size_class_label, size_residual_label):
    ''' Compute 3D bounding box IoU from network output and labels.
    All inputs are numpy arrays.

    Inputs:
        center_pred: (B,3)
        heading_logits: (B,NUM_HEADING_BIN)
        heading_residuals: (B,NUM_HEADING_BIN)
        size_logits: (B,NUM_SIZE_CLUSTER)
        size_residuals: (B,NUM_SIZE_CLUSTER,3)
        center_label: (B,3)
        heading_class_label: (B,)
        heading_residual_label: (B,)
        size_class_label: (B,)
        size_residual_label: (B,3)
    Output:
        iou2ds: (B,) birdeye view oriented 2d box ious
        iou3ds: (B,) 3d box ious
    '''
    pred_val = np.argmax(logits, 2)
    batch_size = heading_logits.shape[0]
    heading_class = np.argmax(heading_logits, 1)  # B
    heading_residual = np.array([heading_residuals[i, heading_class[i]] \
                                 for i in range(batch_size)])  # B,
    size_class = np.argmax(size_logits, 1)  # B
    size_residual = np.vstack([size_residuals[i, size_class[i], :] \
                               for i in range(batch_size)])

    iou2d_list = []
    iou3d_list = []
    box_pred_nbr = 0.0
    for i in range(batch_size):
        # if object has low seg mask break
        if (np.sum(pred_val[i]) < 50):
            continue
        else:
            heading_angle = class2angle(heading_class[i],
                                        heading_residual[i], NUM_HEADING_BIN)
            box_size = class2size(size_class[i], size_residual[i])
            corners_3d = get_3d_box(box_size, heading_angle, center_pred[i])

            heading_angle_label = class2angle(heading_class_label[i],
                                              heading_residual_label[i], NUM_HEADING_BIN)
            box_size_label = class2size(size_class_label[i], size_residual_label[i])
            if (center_label[i][2] < 0.0):
                iou3d_list.append(0.0)
                iou2d_list.append(0.0)
            else:

                corners_3d_label = get_3d_box(box_size_label,
                                              heading_angle_label, center_label[i])

                iou_3d, iou_2d = box3d_iou(corners_3d, corners_3d_label)
                iou3d_list.append(iou_3d)
                iou2d_list.append(iou_2d)
            box_pred_nbr = box_pred_nbr + 1.0

    return np.array(iou2d_list, dtype=np.float32), \
           np.array(iou3d_list, dtype=np.float32), np.array(box_pred_nbr, dtype=np.float32)


def compute_box3d_iou_bbox(center_pred,
                           heading_logits, heading_residuals,
                           size_logits, size_residuals,
                           center_label,
                           heading_class_label, heading_residual_label,
                           size_class_label, size_residual_label):
    ''' Compute 3D bounding box IoU from network output and labels.
    All inputs are numpy arrays.

    Inputs:
        center_pred: (B,3)
        heading_logits: (B,NUM_HEADING_BIN)
        heading_residuals: (B,NUM_HEADING_BIN)
        size_logits: (B,NUM_SIZE_CLUSTER)
        size_residuals: (B,NUM_SIZE_CLUSTER,3)
        center_label: (B,3)
        heading_class_label: (B,)
        heading_residual_label: (B,)
        size_class_label: (B,)
        size_residual_label: (B,3)
    Output:
        iou2ds: (B,) birdeye view oriented 2d box ious
        iou3ds: (B,) 3d box ious
    '''
    batch_size = heading_logits.shape[0]
    heading_class = np.argmax(heading_logits, 1)  # B
    heading_residual = np.array([heading_residuals[i, heading_class[i]] \
                                 for i in range(batch_size)])  # B,
    size_class = np.argmax(size_logits, 1)  # B
    size_residual = np.vstack([size_residuals[i, size_class[i], :] \
                               for i in range(batch_size)])

    iou2d_list = []
    iou3d_list = []
    box_pred_nbr = 0.0
    for i in range(batch_size):
        # if object has low seg mask break
        heading_angle = class2angle(heading_class[i],
                                    heading_residual[i], NUM_HEADING_BIN)
        box_size = class2size(size_class[i], size_residual[i])
        corners_3d = get_3d_box(box_size, heading_angle, center_pred[i])

        heading_angle_label = class2angle(heading_class_label[i],
                                          heading_residual_label[i], NUM_HEADING_BIN)
        box_size_label = class2size(size_class_label[i], size_residual_label[i])
        if (center_label[i][2] < 0.0):
            iou3d_list.append(0.0)
            iou2d_list.append(0.0)
        else:

            corners_3d_label = get_3d_box(box_size_label,
                                          heading_angle_label, center_label[i])

            iou_3d, iou_2d = box3d_iou(corners_3d, corners_3d_label)
            iou3d_list.append(iou_3d)
            iou2d_list.append(iou_2d)
        box_pred_nbr = box_pred_nbr + 1.0

    return np.array(iou2d_list, dtype=np.float32), \
           np.array(iou3d_list, dtype=np.float32), np.array(box_pred_nbr, dtype=np.float32)


def compute_box3d_iou_batch_test1(output, center_pred,
                                  heading_logits, heading_residuals,
                                  size_logits, size_residuals,
                                  center_label,
                                  heading_class_label, heading_residual_label,
                                  size_class_label, size_residual_label):
    ''' Compute 3D bounding box IoU from network output and labels.
    All inputs are numpy arrays.

    Inputs:
        center_pred: (B,3)
        heading_logits: (B,NUM_HEADING_BIN)
        heading_residuals: (B,NUM_HEADING_BIN)
        size_logits: (B,NUM_SIZE_CLUSTER)
        size_residuals: (B,NUM_SIZE_CLUSTER,3)
        center_label: (B,3)
        heading_class_label: (B,)
        heading_residual_label: (B,)
        size_class_label: (B,)
        size_residual_label: (B,3)
    Output:
        iou2ds: (B,) birdeye view oriented 2d box ious
        iou3ds: (B,) 3d box ious
    '''
    # pred_val = np.argmax(logits, 2)
    batch_size = heading_logits.shape[0]
    heading_class = heading_logits  # np.argmax(heading_logits, 1)  # B
    heading_residual = heading_residuals  # np.array([heading_residuals[i, heading_class[i]] \
    #         for i in range(batch_size)])  # B,
    size_class = size_logits  # np.argmax(size_logits, 1)  # B
    size_residual = size_residuals  # np.vstack([size_residuals[i, size_class[i], :] \
    # for i in range(batch_size)])

    iou2d_list = []
    iou3d_list = []
    box_pred_nbr = 0.0
    for i in range(batch_size):
        # if object has low seg mask continue
        if (np.sum(output[i]) < 50):
            continue
        else:
            heading_angle = class2angle(heading_class[i],
                                        heading_residual[i], NUM_HEADING_BIN)
            box_size = class2size(size_class[i], size_residual[i])
            corners_3d = get_3d_box(box_size, heading_angle, center_pred[i])

            heading_angle_label = class2angle(heading_class_label[i],
                                              heading_residual_label[i], NUM_HEADING_BIN)
            box_size_label = class2size(size_class_label[i], size_residual_label[i])
            if (center_label[i][2] < 0.0):
                iou3d_list.append(0.0)
                iou2d_list.append(0.0)
            else:

                corners_3d_label = get_3d_box(box_size_label,
                                              heading_angle_label, center_label[i])

                iou_3d, iou_2d = box3d_iou(corners_3d, corners_3d_label)
                iou3d_list.append(iou_3d)
                iou2d_list.append(iou_2d)
            box_pred_nbr = box_pred_nbr + 1.0

    return np.array(iou2d_list, dtype=np.float32), \
           np.array(iou3d_list, dtype=np.float32), np.array(box_pred_nbr, dtype=np.float32)


def compute_box3d_iou(center_pred,
                      heading_logits, heading_residuals,
                      size_logits, size_residuals,
                      center_label,
                      heading_class_label, heading_residual_label,
                      size_class_label, size_residual_label):
    ''' Compute 3D bounding box IoU from network output and labels.
    All inputs are numpy arrays.

    Inputs:
        center_pred: (B,3)
        heading_logits: (B,NUM_HEADING_BIN)
        heading_residuals: (B,NUM_HEADING_BIN)
        size_logits: (B,NUM_SIZE_CLUSTER)
        size_residuals: (B,NUM_SIZE_CLUSTER,3)
        center_label: (B,3)
        heading_class_label: (B,)
        heading_residual_label: (B,)
        size_class_label: (B,)
        size_residual_label: (B,3)
    Output:
        iou2ds: (B,) birdeye view oriented 2d box ious
        iou3ds: (B,) 3d box ious
    '''
    batch_size = heading_logits.shape[0]
    heading_class = np.argmax(heading_logits, 1)  # B
    heading_residual = np.array([heading_residuals[i, heading_class[i]] \
                                 for i in range(batch_size)])  # B,
    size_class = np.argmax(size_logits, 1)  # B
    size_residual = np.vstack([size_residuals[i, size_class[i], :] \
                               for i in range(batch_size)])

    iou2d_list = []
    iou3d_list = []
    for i in range(batch_size):
        # if object has low seg mask break
        heading_angle = class2angle(heading_class[i],
                                    heading_residual[i], NUM_HEADING_BIN)
        box_size = class2size(size_class[i], size_residual[i])
        corners_3d = get_3d_box(box_size, heading_angle, center_pred[i])

        heading_angle_label = class2angle(heading_class_label[i],
                                          heading_residual_label[i], NUM_HEADING_BIN)
        box_size_label = class2size(size_class_label[i], size_residual_label[i])
        corners_3d_label = get_3d_box(box_size_label,
                                      heading_angle_label, center_label[i])

        iou_3d, iou_2d = box3d_iou(corners_3d, corners_3d_label)
        iou3d_list.append(iou_3d)
        iou2d_list.append(iou_2d)
    return np.array(iou2d_list, dtype=np.float32), \
           np.array(iou3d_list, dtype=np.float32)


def from_prediction_to_label_format(center, angle_class, angle_res, \
                                    size_class, size_res, rot_angle):
    ''' Convert predicted box parameters to label format. '''
    l, w, h = class2size(size_class, size_res)
    ry = class2angle(angle_class, angle_res, NUM_HEADING_BIN) + rot_angle
    tx, ty, tz = rotate_pc_along_y(np.expand_dims(center, 0), -rot_angle).squeeze()
    # ty += h / 2.0
    return h, w, l, tx, ty, tz, ry


if __name__ == '__main__':
    import mayavi.mlab as mlab

    sys.path.append(os.path.join(ROOT_DIR, 'mayavi'))
    from viz_util import draw_lidar, draw_gt_boxes3d

    median_list = []
    """dataset = RadarDataset_seg('pc_radar_2','KITTI_2', npoints=25000, split='test',
                                rotate_to_center=False, one_hot=True, all_batches=False, translate_radar_center=False,
                                store_data=True, proposals_3=False, no_color=True)
    for i in range(len(dataset)):
        data = dataset[i]
        print(dataset.id_list[i])
        print(('Center: ', data[2], \
               'angle_class: ', data[3], 'angle_res:', data[4], \
               'size_class: ', data[5], 'size_residual:', data[6]))  # , \
        # 'real_size:', g_type_mean_size([g_class2type[data[4]]] + data[5]))
        # print("radar_point", dataset.radar_point_list[i])
        # print(('Frustum angle: ', dataset.frustum_angle_list[i]))
        # median_list.append(np.median(data[0][:, 0]))
        # print((data[2], dataset.box3d_list[i], median_list[-1]))
        #box3d_from_label = get_3d_box(class2size(data[5], data[6]), class2angle(data[3], data[4], 12), data[2])
        # print("angle: ",dataset.heading_list[i],class2angle(data[3], data[4], 12))
        ps = data[0]
        seg = data[1]
        print(np.count_nonzero(seg==1))
        fig = mlab.figure(figure=None, bgcolor=(0.4, 0.4, 0.4), fgcolor=None, engine=None, size=(1000, 500))
        mlab.points3d(ps[:, 0], ps[:, 1], ps[:, 2], seg, mode='point', colormap='gnuplot', scale_factor=1, figure=fig)
        mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2, figure=fig)
        draw_gt_boxes3d(data[2], fig, color=(1, 0, 0))
        #print([box3d_from_label])
        # print(data[7])
        #draw_gt_boxes3d([data[8]], fig, color=(0, 0, 1))
        mlab.orientation_axes()
        raw_input()"""

    dataset = RadarDataset_bbox('pc_radar_2', "KITTI", npoints=3500, split='val',
                                    rotate_to_center=False, one_hot=True, all_batches=False,
                                    translate_radar_center=False,
                                    store_data=True, proposals_3=False, no_color=True)
    """for i in range(len(dataset)):
        data = dataset[i]
        print(dataset.id_list[i])
        print(('Center: ', data[3], \
               'angle_class: ', data[4], 'angle_res:', data[5], \
               'size_class: ', data[6], 'size_residual:', data[7]))  # , \
        # 'real_size:', g_type_mean_size([g_class2type[data[4]]] + data[5]))
        # print("radar_point", dataset.radar_point_list[i])
        # print(('Frustum angle: ', dataset.frustum_angle_list[i]))
        # median_list.append(np.median(data[0][:, 0]))
        # print((data[2], dataset.box3d_list[i], median_list[-1]))
        box3d_from_label = get_3d_box(class2size(data[6], data[7]), class2angle(data[4], data[5], 12), data[3])
        # print("angle: ",dataset.heading_list[i],class2angle(data[3], data[4], 12))
        ps = data[0]
        # ps[:,0]= ps[:, 0]+data[1][0]
        # ps[:, 1]=ps[:, 1]+data[1][0]
        seg = np.zeros(len(ps))
        fig = mlab.figure(figure=None, bgcolor=(0.4, 0.4, 0.4), fgcolor=None, engine=None, size=(1000, 500))
        mlab.points3d(ps[:, 0] + data[1][0], ps[:, 1] + data[1][1], ps[:, 2] + data[1][2], mode='point',
                      colormap='gnuplot', scale_factor=1, figure=fig)
        # mlab.points3d(data[9][:,0], data[9][:, 1] , data[9][:, 2], mode='point',
        #                           colormap='gnuplot', scale_factor=1, figure=fig)
        mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2, figure=fig)
        mlab.points3d(data[1][0], data[1][1], data[1][2], color=(1, 0, 0), mode='sphere', scale_factor=0.2, figure=fig)
        draw_gt_boxes3d([box3d_from_label], fig, color=(1, 0, 0))
        print([box3d_from_label])
        # print(data[7])
        draw_gt_boxes3d([data[8]], fig, color=(0, 0, 1))
        mlab.orientation_axes()
        raw_input()"""
    # print(np.mean(np.abs(median_list)))
