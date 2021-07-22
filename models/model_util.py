import numpy as np
#import tensorflow as tf
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
#import tf_util

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'train'))
import kitti_utils
from box_util import box3d_iou

# -----------------
# Global Constants
# -----------------

NUM_HEADING_BIN = 12
NUM_SIZE_CLUSTER = 8  # one cluster for each type
NUM_OBJECT_POINT = 512

n_AB = 25
n_radar_set = 16
npoints_radar_mask = 3500
n_points_AB = 512
n_batches = 4

g_type2class = {'Car': 0, 'Van': 1, 'Truck': 2, 'Pedestrian': 3,
                'Person_sitting': 4, 'Cyclist': 5, 'Tram': 6, 'Misc': 7}
g_class2type = {g_type2class[t]: t for t in g_type2class}
g_type2onehotclass = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
g_type_mean_size = {'Car': np.array([3.88311640418, 1.62856739989, 1.52563191462]),
                    'Van': np.array([5.06763659, 1.9007158, 2.20532825]),
                    'Truck': np.array([10.13586957, 2.58549199, 3.2520595]),
                    'Pedestrian': np.array([1.822122, 1.4857313, 1.407308]),
                    # 'Pedestrian': np.array([0.84422524,0.66068622,1.76255119]),
                    'Person_sitting': np.array([0.80057803, 0.5983815, 1.27450867]),
                    'Cyclist': np.array([1.76282397, 0.59706367, 1.73698127]),
                    'Tram': np.array([16.17150617, 2.53246914, 3.53079012]),
                    'Misc': np.array([3.64300781, 1.54298177, 1.92320313])}
g_mean_size_arr = np.zeros((NUM_SIZE_CLUSTER, 3))  # size clustrs
for i in range(NUM_SIZE_CLUSTER):
    g_mean_size_arr[i, :] = g_type_mean_size[g_class2type[i]]


# -----------------
# TF Functions Helpers
# -----------------

def tf_gather_object_pc(point_cloud, mask, npoints=512):
    ''' Gather object point clouds according to predicted masks.
    Input:
        point_cloud: TF tensor in shape (B,N,C)
        mask: TF tensor in shape (B,N) of 0 (not pick) or 1 (pick)
        npoints: int scalar, maximum number of points to keep (default: 512)
    Output:
        object_pc: TF tensor in shape (B,npoint,C)
        indices: TF int tensor in shape (B,npoint,2)
    '''

    def mask_to_indices(mask):
        indices = np.zeros((mask.shape[0], npoints, 2), dtype=np.int32)
        for i in range(mask.shape[0]):
            pos_indices = np.where(mask[i, :] > 0.5)[0]
            # skip cases when pos_indices is empty
            if len(pos_indices) > 0:
                if len(pos_indices) > npoints:
                    choice = np.random.choice(len(pos_indices),
                                              npoints, replace=False)
                else:
                    choice = np.random.choice(len(pos_indices),
                                              npoints - len(pos_indices), replace=True)
                    choice = np.concatenate((np.arange(len(pos_indices)), choice))
                np.random.shuffle(choice)
                indices[i, :, 1] = pos_indices[choice]
            indices[i, :, 0] = i
        return indices

    indices = tf.py_func(mask_to_indices, [mask], tf.int32)
    #print("indices", indices)
    #print("indices", indices.shape)
    object_pc = tf.gather_nd(point_cloud, indices)
    return object_pc, indices


def tf_gather_object_pc_feats(point_cloud_, pc_features_, mask_, radar_set_, radar_rois_, idx_, npoints=512):
    ''' Gather object point clouds according to predicted masks.
    Input:
        point_cloud: TF tensor in shape (B,N,C)
        mask: TF tensor in shape (B,N) of 0 (not pick) or 1 (pick)
        npoints: int scalar, maximum number of points to keep (default: 512)
    Output:
        object_pc: TF tensor in shape (B,npoint,C)
        indices: TF int tensor in shape (B,npoint,2)
    '''

    def mask_to_indices(mask, radar_set, radar_rois, point_cloud, pc_features, idx):

        indices = np.zeros((mask.shape[0], n_radar_set, npoints_radar_mask), dtype=np.int32)
        object_pc_list = np.zeros((mask.shape[0] * n_radar_set, npoints_radar_mask, 4), dtype=np.float32)
        radar_rois_list = np.zeros((mask.shape[0] * n_radar_set, 7), dtype=np.float32)
        object_feats_list = np.zeros((mask.shape[0] * n_radar_set, npoints_radar_mask, 128), dtype=np.float32)
        idx_list = np.zeros((mask.shape[0] * n_radar_set, 1), dtype=np.int32)
        # indices[420,] =0
        #print('indices', indices.shape)
        #print('object_pc_list', object_pc_list.shape)
        #print('object_feats_list', object_feats_list.shape)
        #print('idx_list', idx_list.shape)
        for i in range(mask.shape[0]):
            for j in range(radar_set.shape[1]):
                pos_indices = np.where(mask[i, :] > 0.5)[0]
                pred_pos = np.zeros(mask.shape[1])
                pred_pos[pos_indices] = 1.0
                labels_per_roi = radar_set[i][j] * pred_pos
                pos_indices = np.where(labels_per_roi == 1)[0]

                # skip cases when pos_indices is empty
                if len(pos_indices) > 0:
                    if len(pos_indices) > npoints_radar_mask:
                        choice = np.random.choice(len(pos_indices),
                                                  npoints_radar_mask, replace=False)
                    else:
                        choice = np.random.choice(len(pos_indices),
                                                  npoints_radar_mask - len(pos_indices), replace=True)
                        choice = np.concatenate((np.arange(len(pos_indices)), choice))
                    np.random.shuffle(choice)
                    indices[i, j, :] = pos_indices[choice]
                    object_pc_list[i * j, :] = point_cloud[i, indices[i, j]]
                    object_feats_list[i * j, :] = pc_features[i, indices[i, j]]
                    radar_rois_list[i * j, :] = radar_rois[i, j]
                    idx_list[i * j, 0] = idx[i]
                # indices[i,j,:] = i
        # indices[420,] = 0
        # for i in range(indices.shape[0]):
        #    for j in range(indices.shape[1]):

        return object_pc_list, object_feats_list, idx_list, indices, radar_rois_list

    object_pc_list, object_feats_list, idx_list, indices, radar_rois_list = tf.py_func(mask_to_indices,
                                                                                       [mask_, radar_set_, radar_rois_,
                                                                                        point_cloud_, pc_features_,
                                                                                        idx_],
                                                                                       [tf.float32, tf.float32,
                                                                                        tf.int32, tf.int32, tf.float32])
    #print("object_pc_list", object_pc_list)
    #print("object_feats_list", object_feats_list)
    # object_pc_list=[]
    # object_feats_list=[]
    # for i in range(indices.shape[0]):
    #    object_pc = tf.gather_nd(point_cloud, indices[i])
    #    object_feats = tf.gather_nd(pc_features, indices)
    #    object_pc_list.append(object_pc)
    #    object_feats_list.append(object_feats)
    return object_pc_list, object_feats_list, idx_list, radar_rois_list, indices


def extract_proposals(point_cloud_, pc_features_, mask_, radar_set_, radar_rois_, idx_, npoints=512):
    def get_bins_in_RRoI(point_set, point_feats, Radar_roi):

        centers = []
        bin_pc = []
        bin_feat = []
        trans = np.array([Radar_roi[0], Radar_roi[1], Radar_roi[2]])
        pc_feat = point_feats
        pc = point_set[:, :3] - trans
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
        print("fg_pt_flag_1",np.count_nonzero(fg_pt_flag_1 == 1))
        if (np.count_nonzero(fg_pt_flag_1 == 1) > 2):
            pc_1 = pc[fg_pt_flag_1, :]
            bin_pc.append(pc_1)
            feat1 = pc_feat[fg_pt_flag_1, :]
            bin_feat.append(feat1)
            centers.append(center)
        else:
            bin_pc.append(np.array([]))
            bin_feat.append(np.array([]))
            centers.append(center)
        size = [h, w, l]

        while center_2[0] < max[0] and len(pc)>2 :

            center_1 = [center_1[0] - 1.0 / 8.0, center_1[1], center_1[2]]
            center_2 = [center_2[0] + 1.0 / 8.0, center_2[1], center_2[2]]
            boxes_1 = get_3d_box((h, w, l), 0.0, center_1)
            boxes_2 = get_3d_box((h, w, l), 0.0, center_2)
            fg_pt_flag_1 = kitti_utils.in_hull(pc[:, 0:3], boxes_1)
            print("fg_pt_flag_1", np.count_nonzero(fg_pt_flag_1 == 1))
            fg_pt_flag_2 = kitti_utils.in_hull(pc[:, 0:3], boxes_2)
            print("fg_pt_flag_2", np.count_nonzero(fg_pt_flag_2 == 1))
            if np.count_nonzero(fg_pt_flag_1 == 1) > 1:
                pc_1 = pc[fg_pt_flag_1, :]
                bin_pc.append(pc_1)
                feat1 = pc_feat[fg_pt_flag_1, :]
                bin_feat.append(feat1)
                centers.append(center_1)
            else:
                bin_pc.append(np.array([]))
                bin_feat.append(np.array([]))
                centers.append(center_1)

            if np.count_nonzero(fg_pt_flag_2 == 1) > 1:
                pc_2 = pc[fg_pt_flag_2, :]
                bin_pc.insert(0, pc_2)
                feat2 = pc_feat[fg_pt_flag_2, :]
                bin_feat.insert(0, feat2)
                centers.insert(0, center_2)
            else:
                bin_pc.insert(0, np.array([]))
                bin_feat.insert(0, np.array([]))
                centers.insert(0, center_2)
            fg_pt_flag = np.logical_or(fg_pt_flag_1, fg_pt_flag_2)
            pc = pc[~fg_pt_flag, :]
            pc_feat = pc_feat[~fg_pt_flag, :]
        print("bin_pc in bins",len(bin_pc))
        return bin_pc, bin_feat, centers, size, trans

    def local_min_method(bin_pc,bin_feat, centers, size, radar_angle, trans):
        if(len(bin_pc)>2):
            bin_y_max = []
            for i in range(len(bin_pc)):
                if (bin_pc[i].size == 0):
                    bin_y_max.append(centers[i][1] + size[0] / 2)
                else:
                    bin_y_max.append(np.min(bin_pc[i][:, 1]))

            minimum = []
            if bin_y_max[0] < bin_y_max[1]:
                minimum.append(1)
            else:
                minimum.append(-1)
            for m in range(1, len(bin_y_max) - 1):
                if bin_y_max[m] < bin_y_max[m - 1] and bin_y_max[m] < bin_y_max[m + 1]:
                    minimum.append(1)
                elif bin_y_max[m] > bin_y_max[m - 1] and bin_y_max[m] > bin_y_max[m + 1]:
                    minimum.append(-1)
                else:
                    minimum.append(0)
            if bin_y_max[len(bin_y_max) - 1] < bin_y_max[len(bin_y_max) - 1]:
                minimum.append(1)
            else:
                minimum.append(-1)
            print(minimum)
            local_min_indices = np.argwhere(np.array(minimum) == -1)
            pc_AB_list = np.empty([0, 512, 3])
            feat_AB_list = np.empty([0, 512, 128])
            corners_AB = np.empty([0, 8, 3])
            for n in range(len(local_min_indices)):
                pc_AB = np.empty([0, 3])
                feat_AB = np.empty([0, 128])
                for m in range(n + 1, len(local_min_indices)):
                    for o in range(local_min_indices[n][0], local_min_indices[m][0]):
                        if (bin_pc[o].size != 0):
                            pc_AB = np.concatenate((pc_AB, bin_pc[o]))
                            feat_AB = np.concatenate((feat_AB, bin_feat[o]))
                    print("pc_AB_list:", len(pc_AB_list))
                    if (len(pc_AB) > 50):
                        min = np.array([np.min(pc_AB[:, 0]), np.min(pc_AB[:, 1]), np.min(pc_AB[:, 2])])
                        max = np.array([np.max(pc_AB[:, 0]), np.max(pc_AB[:, 1]), np.max(pc_AB[:, 2])])
                        corners = corneers_from_minmax(min, max)
                        center = (min + max) / 2.0

                        corners = inverse_rotate_pc_along_y(corners, radar_angle)
                        corners = corners + trans
                        pc = inverse_rotate_pc_along_y(pc_AB, radar_angle)
                        pc_ = pc + trans
                        if len(pc_) > n_points_AB:
                            choice = np.random.choice(len(pc_),
                                                      n_points_AB, replace=False)
                        else:
                            choice = np.random.choice(len(pc_),
                                                      n_points_AB - len(pc_), replace=True)
                            choice = np.concatenate((np.arange(len(pc_)), choice))
                        #print("len iter pc",len(pc_))
                        #print("len iter feat", len(feat))
                        pc_ = pc_[choice]
                        feat = feat_AB[choice]
                        pc_ = np.expand_dims(pc_, 0)
                        feat = np.expand_dims(feat, 0)
                        corners = np.expand_dims(corners, 0)
                        corners_AB = np.concatenate((corners_AB, corners))
                        pc_AB_list = np.concatenate((pc_AB_list, pc_))
                        feat_AB_list = np.concatenate((feat_AB_list, feat))
        else:
            pc_AB_list = np.empty([0, 512, 3])
            feat_AB_list = np.empty([0, 512, 128])
            corners_AB = np.empty([0, 8, 3])

        return pc_AB_list,feat_AB_list, corners_AB


    def divide_in_n_AB(bin_pc, bin_feat, n):
        pc_AB_list = []
        feat_AB_list = []
        for i in range(0, len(bin_pc) - n, 1):
            pc_AB = np.empty([0, 3])
            feat_AB = np.empty([0, 128])
            # print(len(bin_pc))
            # print(i,i+n)
            for j in range(i, i + n):
                # print(j)
                # print(bin_pc[j].size)
                if bin_pc[j].size != 0:
                    pc_AB = np.concatenate((pc_AB, bin_pc[j]))
                    feat_AB = np.concatenate((feat_AB, bin_feat[j]))

            pc_AB_list.append(pc_AB)
            feat_AB_list.append(feat_AB)
        return pc_AB_list, feat_AB_list

    def iterative_method(bin_pc, bin_feat, centers, size, radar_angle, trans):
        pc_AB_list = np.empty([0,512,3])
        feat_AB_list = np.empty([0,512,128])
        corners_AB =np.empty([0,8,3])
        for i in range(3, 8):
            pc_AB_, feat_AB_ = divide_in_n_AB(bin_pc, bin_feat, i)
            #print("extracted AB:",len(pc_AB_))
            for j in range(len(pc_AB_)):
                pc_ = pc_AB_[j]
                feat= feat_AB_[j]
                if (len(pc_) > 50):
                    if len(pc_) > n_points_AB:
                        choice = np.random.choice(len(pc_),
                                                  n_points_AB, replace=False)
                    else:
                        choice = np.random.choice(len(pc_),
                                                  n_points_AB - len(pc_), replace=True)
                        choice = np.concatenate((np.arange(len(pc_)), choice))
                    #print("len iter pc",len(pc_))
                    #print("len iter feat", len(feat))
                    pc_ = pc_[choice]
                    feat = feat[choice]
                    #print("len iter pc coice", len(pc_))
                    #print("len iter feat choice", len(feat))
                    min = np.array([np.min(pc_[:, 0]), np.min(pc_[:, 1]), np.min(pc_[:, 2])])
                    max = np.array([np.max(pc_[:, 0]), np.max(pc_[:, 1]), np.max(pc_[:, 2])])
                    #print("min max",min,max)
                    corners = corneers_from_minmax(min, max)
                    center = (min + max) / 2.0

                    corners = inverse_rotate_pc_along_y(corners, radar_angle)
                    #print("corners",corners)
                    corners = corners + trans
                    #print("corners_trans", corners)
                    #corners_AB.append(corners)
                    pc = inverse_rotate_pc_along_y(pc_, radar_angle)
                    pc = pc + trans
                    #print("pcc_trans", pc)
                    pc=np.expand_dims(pc,0)
                    feat=np.expand_dims(feat,0)
                    #print(pc.shape)
                    corners=np.expand_dims(corners,0)
                    corners_AB= np.concatenate((corners_AB, corners))
                    pc_AB_list = np.concatenate((pc_AB_list, pc))
                    feat_AB_list = np.concatenate((feat_AB_list, feat))
                    #pc_AB_list.append(pc)
                    #feat_AB_list.append(feat)
        return pc_AB_list, feat_AB_list, corners_AB

    def mask_to_indices(mask, radar_set, radar_rois, point_cloud, pc_features, idx):
        indices = []  # np.zeros((mask.shape[0],n_radar_set, npoints_radar_mask), dtype=np.int32)
        object_pc_list = []  # np.zeros((mask.shape[0]*n_radar_set,npoints_radar_mask,4),dtype=np.float32)
        radar_rois_list = []  # np.zeros((mask.shape[0]*n_radar_set,7),dtype=np.float32)
        object_feats_list = []  # np.zeros((mask.shape[0]*n_radar_set,npoints_radar_mask,128),dtype=np.float32)
        idx_list = []  # np.zeros((mask.shape[0]*n_radar_set,1),dtype=np.int32)

        for i in range(mask.shape[0]):
            for j in range(radar_set.shape[1]):
                if (np.count_nonzero(radar_set[i, :] == 1) > 0):
                    pc = point_cloud[i]
                    pc_f = pc_features[i]
                    pos_indices = np.where(mask[i, :] > 0.5)[0]
                    pred_pos = np.zeros(mask.shape[1])
                    pred_pos[pos_indices] = 1.0
                    # print("pos_indices",np.count_nonzero(pred_pos==1))
                    # print("radar_set[i][j]",np.count_nonzero(radar_set[i][j]==1))
                    labels_per_roi = radar_set[i][j] * pred_pos
                    # print("labels_per_roi",np.count_nonzero(labels_per_roi==1))
                    # print("labels_per_roi",labels_per_roi)
                    pos_indices = np.where(labels_per_roi == 1)[0]
                    # print("indices",labels_per_roi[pos_indices[10]])

                    # skip cases when pos_indices is empty
                    if len(pos_indices) > 0:
                        indices.append(pos_indices)
                        object_pc_list.append(pc[pos_indices])
                        object_feats_list.append(pc_f[pos_indices])
                        radar_rois_list.append(radar_rois[i, j])
                        idx_list.append(idx[i])
        #print('object_pc_list', len(object_pc_list))
        #print('object_feats_list', len(object_feats_list))

        return object_pc_list, object_feats_list, idx_list, radar_rois_list, indices

    def proposals_batches(mask, radar_set, radar_rois, point_cloud, pc_features, idx):

        AB_pc_batches = np.zeros((n_AB * n_batches, n_points_AB, 3), dtype=np.float32)
        AB_corners_batches = np.zeros((n_AB * n_batches, 8, 3), dtype=np.float32)
        AB_feat_batches = np.zeros((n_AB * n_batches, n_points_AB, 128), dtype=np.float32)
        ids_prop = np.zeros((n_AB * n_batches, 1), dtype=np.int32)
        AB_pc_batches_ = np.empty([0,512,3])
        AB_corners_batches_ = np.empty([0,8,3])
        AB_feat_batches_ = np.empty([0,512,128])
        ids_prop_ = np.empty([0])

        object_pc_list, object_feats_list, idx_list, \
        radar_rois_list, indices = mask_to_indices(mask, radar_set, radar_rois, point_cloud, pc_features, idx)

        for i in range(len(object_pc_list)):

            bin_pc, bin_feat, centers, size, trans = get_bins_in_RRoI(object_pc_list[i], object_feats_list[i],
                                                                      radar_rois_list[i])

            AB_pc, AB_feat, AB_corners = iterative_method(bin_pc, bin_feat, centers, size, radar_rois_list[i][6], trans)
            #AB_pc, AB_feat, AB_corners =local_min_method(bin_pc, bin_feat, centers, size, radar_rois_list[i][6], trans)
            #AB_pc_batches[6000] = 0
            print("AB_pc",AB_pc.shape)
            print("AB_corners",AB_corners.shape)
            print("AB_feat",AB_feat.shape)
            AB_pc_batches_=np.concatenate((AB_pc_batches_,AB_pc))
            AB_corners_batches_=np.concatenate((AB_corners_batches_,AB_corners))
            AB_feat_batches_=np.concatenate((AB_feat_batches_,AB_feat))
            ids_=np.repeat(idx_list[i],AB_pc.shape[0])
            #ids_=np.expand_dims(ids_,0)
            ids_prop_=np.concatenate((ids_prop_,ids_))

        nb_AB = len(AB_pc_batches_)
        if nb_AB > 0:
            if nb_AB > n_AB * n_batches:
                choice = np.random.choice(nb_AB,
                                          n_AB * n_batches, replace=False)

            else:
                choice = np.random.choice(nb_AB,
                                          n_AB * n_batches - nb_AB, replace=True)
                choice = np.concatenate((np.arange(nb_AB), choice))
                np.random.shuffle(choice)
            print("choice",choice)
            AB_pc_batches = np.float32(AB_pc_batches_[choice,...])
            AB_corners_batches = np.float32(AB_corners_batches_[choice,...])
            AB_feat_batches = np.float32(AB_feat_batches_[choice,...])
            ids_prop = np.int32(ids_prop_[choice])
        print("AB_pc_batches",AB_pc_batches.shape)
        print("AB_corners_batches",AB_corners_batches.shape)

        return AB_pc_batches, AB_corners_batches, AB_feat_batches, ids_prop

    pc_AB, corners_AB, feat_AB, ids_AB = tf.py_func(proposals_batches,
                                                                 [mask_, radar_set_, radar_rois_, point_cloud_,
                                                                  pc_features_, idx_],
                                                                 [tf.float32, tf.float32, tf.float32, tf.int32])

    return pc_AB, feat_AB, corners_AB, ids_AB


def get_box3d_corners_helper(centers, headings, sizes):
    """ TF layer. Input: (N,3), (N,), (N,3), Output: (N,8,3) """
    # print '-----', centers
    N = centers.get_shape()[0].value
    h = tf.slice(sizes, [0, 0], [-1, 1])  # (N,1)
    w = tf.slice(sizes, [0, 1], [-1, 1])  # (N,1)
    l = tf.slice(sizes, [0, 2], [-1, 1])  # (N,1)
    # print l,w,h
    x_corners = tf.concat([w / 2, w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2], axis=1)  # (N,8)
    y_corners = tf.concat([h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2], axis=1)  # (N,8)
    z_corners = tf.concat([l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2], axis=1)  # (N,8)
    corners = tf.concat([tf.expand_dims(x_corners, 1), tf.expand_dims(y_corners, 1), tf.expand_dims(z_corners, 1)],
                        axis=1)  # (N,3,8)
    # print x_corners, y_corners, z_corners
    c = tf.cos(headings)
    s = tf.sin(headings)
    ones = tf.ones([N], dtype=tf.float32)
    zeros = tf.zeros([N], dtype=tf.float32)
    row1 = tf.stack([c, zeros, s], axis=1)  # (N,3)
    row2 = tf.stack([zeros, ones, zeros], axis=1)
    row3 = tf.stack([-s, zeros, c], axis=1)
    R = tf.concat([tf.expand_dims(row1, 1), tf.expand_dims(row2, 1), tf.expand_dims(row3, 1)], axis=1)  # (N,3,3)
    # print row1, row2, row3, R, N
    corners_3d = tf.matmul(R, corners)  # (N,3,8)
    corners_3d += tf.tile(tf.expand_dims(centers, 2), [1, 1, 8])  # (N,3,8)
    corners_3d = tf.transpose(corners_3d, perm=[0, 2, 1])  # (N,8,3)
    return corners_3d


def get_box3d_corners(center, heading_residuals, size_residuals):
    """ TF layer.
    Inputs:
        center: (B,3)
        heading_residuals: (B,NH)
        size_residuals: (B,NS,3)
    Outputs:
        box3d_corners: (B,NH,NS,8,3) tensor
    """
    batch_size = center.get_shape()[0].value
    heading_bin_centers = tf.constant(np.arange(0, 2 * np.pi, 2 * np.pi / NUM_HEADING_BIN), dtype=tf.float32)  # (NH,)
    headings = heading_residuals + tf.expand_dims(heading_bin_centers, 0)  # (B,NH)

    mean_sizes = tf.expand_dims(tf.constant(g_mean_size_arr, dtype=tf.float32), 0) + size_residuals  # (B,NS,1)
    sizes = mean_sizes + size_residuals  # (B,NS,3)
    sizes = tf.tile(tf.expand_dims(sizes, 1), [1, NUM_HEADING_BIN, 1, 1])  # (B,NH,NS,3)
    headings = tf.tile(tf.expand_dims(headings, -1), [1, 1, NUM_SIZE_CLUSTER])  # (B,NH,NS)
    centers = tf.tile(tf.expand_dims(tf.expand_dims(center, 1), 1),
                      [1, NUM_HEADING_BIN, NUM_SIZE_CLUSTER, 1])  # (B,NH,NS,3)

    N = batch_size * NUM_HEADING_BIN * NUM_SIZE_CLUSTER
    corners_3d = get_box3d_corners_helper(tf.reshape(centers, [N, 3]), tf.reshape(headings, [N]),
                                          tf.reshape(sizes, [N, 3]))

    return tf.reshape(corners_3d, [batch_size, NUM_HEADING_BIN, NUM_SIZE_CLUSTER, 8, 3])


def huber_loss(error, delta):
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = (abs_error - quadratic)
    losses = 0.5 * quadratic ** 2 + delta * linear
    return tf.reduce_mean(losses)


def parse_output_to_tensors(output, end_points):
    ''' Parse batch output to separate tensors (added to end_points)
    Input:
        output: TF tensor in shape (B,3+2*NUM_HEADING_BIN+4*NUM_SIZE_CLUSTER)
        end_points: dict
    Output:
        end_points: dict (updated)
    '''
    batch_size = output.get_shape()[0].value
    center = tf.slice(output, [0, 0], [-1, 3])
    end_points['center_boxnet'] = center

    heading_scores = tf.slice(output, [0, 3], [-1, NUM_HEADING_BIN])
    heading_residuals_normalized = tf.slice(output, [0, 3 + NUM_HEADING_BIN],
                                            [-1, NUM_HEADING_BIN])
    end_points['heading_scores'] = heading_scores  # BxNUM_HEADING_BIN
    end_points['heading_residuals_normalized'] = \
        heading_residuals_normalized  # BxNUM_HEADING_BIN (-1 to 1)
    end_points['heading_residuals'] = \
        heading_residuals_normalized * (np.pi / NUM_HEADING_BIN)  # BxNUM_HEADING_BIN

    size_scores = tf.slice(output, [0, 3 + NUM_HEADING_BIN * 2],
                           [-1, NUM_SIZE_CLUSTER])  # BxNUM_SIZE_CLUSTER
    size_residuals_normalized = tf.slice(output,
                                         [0, 3 + NUM_HEADING_BIN * 2 + NUM_SIZE_CLUSTER], [-1, NUM_SIZE_CLUSTER * 3])
    size_residuals_normalized = tf.reshape(size_residuals_normalized,
                                           [batch_size, NUM_SIZE_CLUSTER, 3])  # BxNUM_SIZE_CLUSTERx3
    end_points['size_scores'] = size_scores
    end_points['size_residuals_normalized'] = size_residuals_normalized
    end_points['size_residuals'] = size_residuals_normalized * \
                                   tf.expand_dims(tf.constant(g_mean_size_arr, dtype=tf.float32), 0)

    return end_points


# --------------------------------------
# Shared subgraphs for v1 and v2 models
# -------------------------------------
def placeholder_inputs(batch_size, num_point):
    ''' Get useful placeholder tensors.
    Input:
        batch_size: scalar int
        num_point: scalar int
    Output:
        TF placeholders for inputs and ground truths
    '''
    pointclouds_pl = tf.placeholder(tf.float32,
                                    shape=(batch_size, num_point, 4))
    one_hot_vec_pl = tf.placeholder(tf.float32, shape=(batch_size, 3))

    # labels_pl is for segmentation label
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    centers_pl = tf.placeholder(tf.float32, shape=(batch_size, 3))
    heading_class_label_pl = tf.placeholder(tf.int32, shape=(batch_size,))
    heading_residual_label_pl = tf.placeholder(tf.float32, shape=(batch_size,))
    size_class_label_pl = tf.placeholder(tf.int32, shape=(batch_size,))
    size_residual_label_pl = tf.placeholder(tf.float32, shape=(batch_size, 3))
    radar_mask_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point))

    return pointclouds_pl, one_hot_vec_pl, labels_pl, centers_pl, \
           heading_class_label_pl, heading_residual_label_pl, \
           size_class_label_pl, size_residual_label_pl


def placeholder_inputs_seg(batch_size, num_point):
    ''' Get useful placeholder tensors.
    Input:
        batch_size: scalar int
        num_point: scalar int
    Output:
        TF placeholders for inputs and ground truths
    '''
    pointclouds_pl = tf.placeholder(tf.float32,
                                    shape=(batch_size, 3500, 4))
    one_hot_vec_pl = tf.placeholder(tf.float32, shape=(batch_size, 3))

    # labels_pl is for segmentation label
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, 3500))

    return pointclouds_pl, one_hot_vec_pl, labels_pl


def placeholder_inputs_seg_cls(batch_size, num_point):
    ''' Get useful placeholder tensors.
    Input:
        batch_size: scalar int
        num_point: scalar int
    Output:
        TF placeholders for inputs and ground truths
    '''
    pointclouds_pl = tf.placeholder(tf.float32,
                                    shape=(batch_size, num_point, 4))

    one_hot_vec_pl = tf.placeholder(tf.float32, shape=(batch_size, 3))
    radar_set_pl = tf.placeholder(tf.int32, shape=(batch_size, 16, num_point))

    # labels_pl is for segmentation label
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    gt_corners_pl = tf.placeholder(tf.float32, shape=(batch_size, 5, 8, 3))

    ids_pl = tf.placeholder(tf.int32, shape=(batch_size, 1))
    radar_rois_pl = tf.placeholder(tf.float32, shape=(batch_size, 16, 7))

    return pointclouds_pl, one_hot_vec_pl, labels_pl, ids_pl, radar_rois_pl, radar_set_pl, gt_corners_pl


def placeholder_inputs_bbox(batch_size, num_point):
    ''' Get useful placeholder tensors.
    Input:
        batch_size: scalar int
        num_point: scalar int
    Output:
        TF placeholders for inputs and ground truths
    '''
    pointclouds_pl = tf.placeholder(tf.float32,
                                    shape=(batch_size, 512, 3))
    pointclouds_center_pl = tf.placeholder(tf.float32, shape=(batch_size, 3))
    one_hot_vec_pl = tf.placeholder(tf.float32, shape=(batch_size, 3))

    # labels_pl is for segmentation label
    centers_pl = tf.placeholder(tf.float32, shape=(batch_size, 3))
    heading_class_label_pl = tf.placeholder(tf.int32, shape=(batch_size,))
    heading_residual_label_pl = tf.placeholder(tf.float32, shape=(batch_size,))
    size_class_label_pl = tf.placeholder(tf.int32, shape=(batch_size,))
    size_residual_label_pl = tf.placeholder(tf.float32, shape=(batch_size, 3))

    return pointclouds_pl, pointclouds_center_pl, one_hot_vec_pl, centers_pl, \
           heading_class_label_pl, heading_residual_label_pl, \
           size_class_label_pl, size_residual_label_pl


def placeholder_inputs_batch(batch_size, num_point):
    ''' Get useful placeholder tensors.
    Input:
        batch_size: scalar int
        num_point: scalar int
    Output:
        TF placeholders for inputs and ground truths
    '''
    pointclouds_pl = tf.placeholder(tf.float32,
                                    shape=(batch_size, num_point, 4))
    one_hot_vec_pl = tf.placeholder(tf.float32, shape=(batch_size, 3))

    # labels_pl is for segmentation label
    # batch mask
    batch_mask = tf.placeholder(tf.float32, shape=(batch_size, 1))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    centers_pl = tf.placeholder(tf.float32, shape=(batch_size, 3))
    heading_class_label_pl = tf.placeholder(tf.int32, shape=(batch_size,))
    heading_residual_label_pl = tf.placeholder(tf.float32, shape=(batch_size,))
    size_class_label_pl = tf.placeholder(tf.int32, shape=(batch_size,))
    size_residual_label_pl = tf.placeholder(tf.float32, shape=(batch_size, 3))
    radar_mask_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point))

    return pointclouds_pl, one_hot_vec_pl, labels_pl, centers_pl, \
           heading_class_label_pl, heading_residual_label_pl, \
           size_class_label_pl, size_residual_label_pl, batch_mask


def placeholder_inputs_RGB(batch_size, num_point):
    ''' Get useful placeholder tensors.
    Input:
        batch_size: scalar int
        num_point: scalar int
    Output:
        TF placeholders for inputs and ground truths
    '''
    pointclouds_pl = tf.placeholder(tf.float32,
                                    shape=(batch_size, num_point, 6))
    one_hot_vec_pl = tf.placeholder(tf.float32, shape=(batch_size, 3))

    # labels_pl is for segmentation label
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    centers_pl = tf.placeholder(tf.float32, shape=(batch_size, 3))
    heading_class_label_pl = tf.placeholder(tf.int32, shape=(batch_size,))
    heading_residual_label_pl = tf.placeholder(tf.float32, shape=(batch_size,))
    size_class_label_pl = tf.placeholder(tf.int32, shape=(batch_size,))
    size_residual_label_pl = tf.placeholder(tf.float32, shape=(batch_size, 3))
    radar_mask_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point))

    return pointclouds_pl, one_hot_vec_pl, labels_pl, centers_pl, \
           heading_class_label_pl, heading_residual_label_pl, \
           size_class_label_pl, size_residual_label_pl


def placeholder_inputs_test_raw(batch_size, num_point):
    ''' Get useful placeholder tensors.
    Input:
        batch_size: scalar int
        num_point: scalar int
    Output:
        TF placeholders for inputs and ground truths
    '''
    pointclouds_pl = tf.placeholder(tf.float32,
                                    shape=(batch_size, num_point, 4))
    one_hot_vec_pl = tf.placeholder(tf.float32, shape=(batch_size, 3))

    # labels_pl is for segmentation label
    # labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    # centers_pl = tf.placeholder(tf.float32, shape=(batch_size, 3))
    # heading_class_label_pl = tf.placeholder(tf.int32, shape=(batch_size,))
    # heading_residual_label_pl = tf.placeholder(tf.float32, shape=(batch_size,))
    # size_class_label_pl = tf.placeholder(tf.int32, shape=(batch_size,))
    # size_residual_label_pl = tf.placeholder(tf.float32, shape=(batch_size,3))
    # radar_mask_pl= tf.placeholder(tf.float32,shape=(batch_size,num_point))

    return pointclouds_pl, one_hot_vec_pl  # , labels_pl, centers_pl, \
    # heading_class_label_pl, heading_residual_label_pl, \
    # size_class_label_pl, size_residual_label_pl


def point_cloud_masking(point_cloud, logits, end_points, xyz_only=True):
    ''' Select point cloud with predicted 3D mask,
    translate coordinates to the masked points centroid.
    
    Input:
        point_cloud: TF tensor in shape (B,N,C)
        logits: TF tensor in shape (B,N,2)
        end_points: dict
        xyz_only: boolean, if True only return XYZ channels
    Output:
        object_point_cloud: TF tensor in shape (B,M,3)
            for simplicity we only keep XYZ here
            M = NUM_OBJECT_POINT as a hyper-parameter
        mask_xyz_mean: TF tensor in shape (B,3)
    '''
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    mask = tf.slice(logits, [0, 0, 0], [-1, -1, 1]) < \
           tf.slice(logits, [0, 0, 1], [-1, -1, 1])
    mask = tf.to_float(mask)  # BxNx1
    mask_count = tf.tile(tf.reduce_sum(mask, axis=1, keep_dims=True),
                         [1, 1, 3])  # Bx1x3
    point_cloud_xyz = tf.slice(point_cloud, [0, 0, 0], [-1, -1, 3])  # BxNx3
    mask_xyz_mean = tf.reduce_sum(tf.tile(mask, [1, 1, 3]) * point_cloud_xyz,
                                  axis=1, keep_dims=True)  # Bx1x3
    mask = tf.squeeze(mask, axis=[2])  # BxN
    end_points['mask'] = mask
    mask_xyz_mean = mask_xyz_mean / tf.maximum(mask_count, 1)  # Bx1x3

    # Translate to masked points' centroid
    point_cloud_xyz_stage1 = point_cloud_xyz - \
                             tf.tile(mask_xyz_mean, [1, num_point, 1])

    if xyz_only:
        point_cloud_stage1 = point_cloud_xyz_stage1
    else:
        point_cloud_features = tf.slice(point_cloud, [0, 0, 3], [-1, -1, -1])
        point_cloud_stage1 = tf.concat( \
            [point_cloud_xyz_stage1, point_cloud_features], axis=-1)
    num_channels = point_cloud_stage1.get_shape()[2].value

    object_point_cloud, _ = tf_gather_object_pc(point_cloud_stage1,
                                                mask, NUM_OBJECT_POINT)
    object_point_cloud.set_shape([batch_size, NUM_OBJECT_POINT, num_channels])

    return object_point_cloud, tf.squeeze(mask_xyz_mean, axis=1), end_points


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


def get_center_regression_net(object_point_cloud, one_hot_vec,
                              is_training, bn_decay, end_points):
    ''' Regression network for center delta. a.k.a. T-Net.
    Input:
        object_point_cloud: TF tensor in shape (B,M,C)
            point clouds in 3D mask coordinate
        one_hot_vec: TF tensor in shape (B,3)
            length-3 vectors indicating predicted object type
    Output:
        predicted_center: TF tensor in shape (B,3)
    '''
    num_point = object_point_cloud.get_shape()[1].value
    net = tf.expand_dims(object_point_cloud, 2)
    net = tf_util.conv2d(net, 128, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv-reg1-stage1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv-reg2-stage1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 256, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv-reg3-stage1', bn_decay=bn_decay)
    net = tf_util.max_pool2d(net, [num_point, 1],
                             padding='VALID', scope='maxpool-stage1')
    net = tf.squeeze(net, axis=[1, 2])
    net = tf.concat([net, one_hot_vec], axis=1)
    net = tf_util.fully_connected(net, 256, scope='fc1-stage1', bn=True,
                                  is_training=is_training, bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 128, scope='fc2-stage1', bn=True,
                                  is_training=is_training, bn_decay=bn_decay)
    predicted_center = tf_util.fully_connected(net, 3, activation_fn=None,
                                               scope='fc3-stage1')
    return predicted_center, end_points


def get_loss_batch(mask_label, mask_batch, center_label, \
                   heading_class_label, heading_residual_label, \
                   size_class_label, size_residual_label, \
                   end_points, \
                   corner_loss_weight=10.0, \
                   box_loss_weight=1.0):
    ''' Loss functions for 3D object detection.
    Input:
        mask_label: TF int32 tensor in shape (B,N)

        center_label: TF tensor in shape (B,3)
        heading_class_label: TF int32 tensor in shape (B,)
        heading_residual_label: TF tensor in shape (B,)
        size_class_label: TF tensor int32 in shape (B,)
        size_residual_label: TF tensor tensor in shape (B,)
        end_points: dict, outputs from our model
        corner_loss_weight: float scalar
        box_loss_weight: float scalar
        mask_batch: TF int32 tensor in shape(B,)
    Output:
        total_loss: TF scalar tensor
            the total_loss is also added to the losses collection
    '''
    # 3D Segmentation loss
    mask_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits( \
        logits=end_points['mask_logits'], labels=mask_label))
    tf.summary.scalar('3d mask loss', mask_loss)

    # Center regression losses
    center_dist = tf.norm(center_label - (end_points['center']), axis=-1)
    center_loss = huber_loss(center_dist * mask_batch, delta=2.0)
    tf.summary.scalar('center loss', center_loss)
    stage1_center_dist = tf.norm(center_label - \
                                 end_points['stage1_center'], axis=-1)
    stage1_center_loss = huber_loss(stage1_center_dist * mask_batch, delta=1.0)
    tf.summary.scalar('stage1 center loss', stage1_center_loss)

    # Heading loss
    heading_class_loss = tf.reduce_sum( \
        tf.nn.sparse_softmax_cross_entropy_with_logits( \
            logits=end_points['heading_scores'], labels=heading_class_label) * mask_batch) / tf.maximum(
        tf.reduce_sum(mask_batch), 1.0)
    tf.summary.scalar('heading class loss', heading_class_loss)
    hcls_onehot = tf.one_hot(heading_class_label,
                             depth=NUM_HEADING_BIN,
                             on_value=1, off_value=0, axis=-1)  # BxNUM_HEADING_BIN

    heading_residual_normalized_label = \
        heading_residual_label / (np.pi / NUM_HEADING_BIN)

    heading_residual_normalized_loss = huber_loss((tf.reduce_sum( \
        end_points['heading_residuals_normalized'] * tf.to_float(hcls_onehot), axis=1) - \
                                                   heading_residual_normalized_label) * mask_batch, delta=1.0)
    tf.summary.scalar('heading residual normalized loss',
                      heading_residual_normalized_loss)

    # Size loss
    size_class_loss = tf.reduce_sum( \
        tf.nn.sparse_softmax_cross_entropy_with_logits( \
            logits=end_points['size_scores'], labels=size_class_label) * mask_batch) / tf.maximum(
        tf.reduce_sum(mask_batch), 1.0)
    tf.summary.scalar('size class loss', size_class_loss)

    scls_onehot = tf.one_hot(size_class_label,
                             depth=NUM_SIZE_CLUSTER,
                             on_value=1, off_value=0, axis=-1)  # BxNUM_SIZE_CLUSTER
    scls_onehot_tiled = tf.tile(tf.expand_dims( \
        tf.to_float(scls_onehot), -1), [1, 1, 3])  # BxNUM_SIZE_CLUSTERx3
    predicted_size_residual_normalized = tf.reduce_sum( \
        end_points['size_residuals_normalized'] * scls_onehot_tiled, axis=[1])  # Bx3
    # *tf.tile(tf.expand_dims(mask_batch,axis=-1),[1,8,3])
    mean_size_arr_expand = tf.expand_dims( \
        tf.constant(g_mean_size_arr, dtype=tf.float32), 0)  # 1xNUM_SIZE_CLUSTERx3
    mean_size_label = tf.reduce_sum( \
        scls_onehot_tiled * mean_size_arr_expand, axis=[1])  # Bx3
    size_residual_label_normalized = size_residual_label / mean_size_label
    size_normalized_dist = tf.norm( \
        size_residual_label_normalized - predicted_size_residual_normalized,
        axis=-1)
    size_residual_normalized_loss = huber_loss(size_normalized_dist * mask_batch, delta=1.0)
    tf.summary.scalar('size residual normalized loss',
                      size_residual_normalized_loss)

    # Corner loss
    # We select the predicted corners corresponding to the
    # GT heading bin and size cluster.
    corners_3d = get_box3d_corners(end_points['center'],
                                   end_points['heading_residuals'],
                                   end_points['size_residuals'])  # (B,NH,NS,8,3)
    gt_mask = tf.tile(tf.expand_dims(hcls_onehot, 2), [1, 1, NUM_SIZE_CLUSTER]) * \
              tf.tile(tf.expand_dims(scls_onehot, 1), [1, NUM_HEADING_BIN, 1])  # (B,NH,NS)
    corners_3d_pred = tf.reduce_sum( \
        tf.to_float(tf.expand_dims(tf.expand_dims(gt_mask, -1), -1)) * corners_3d,
        axis=[1, 2])  # (B,8,3)

    heading_bin_centers = tf.constant( \
        np.arange(0, 2 * np.pi, 2 * np.pi / NUM_HEADING_BIN), dtype=tf.float32)  # (NH,)
    heading_label = tf.expand_dims(heading_residual_label, 1) + \
                    tf.expand_dims(heading_bin_centers, 0)  # (B,NH)
    heading_label = tf.reduce_sum(tf.to_float(hcls_onehot) * heading_label, 1)
    mean_sizes = tf.expand_dims( \
        tf.constant(g_mean_size_arr, dtype=tf.float32), 0)  # (1,NS,3)
    size_label = mean_sizes + \
                 tf.expand_dims(size_residual_label, 1)  # (1,NS,3) + (B,1,3) = (B,NS,3)
    size_label = tf.reduce_sum( \
        tf.expand_dims(tf.to_float(scls_onehot), -1) * size_label, axis=[1])  # (B,3)
    corners_3d_gt = get_box3d_corners_helper( \
        center_label, heading_label, size_label)  # (B,8,3)
    corners_3d_gt_flip = get_box3d_corners_helper( \
        center_label, heading_label + np.pi, size_label)  # (B,8,3)
    corners_dist = tf.minimum(tf.norm(corners_3d_pred - corners_3d_gt, axis=-1),
                              tf.norm(corners_3d_pred - corners_3d_gt_flip, axis=-1))
    # *tf.tile(tf.expand_dims(mask_batch,axis=-1),[1,8,3])
    corners_loss = huber_loss(corners_dist * tf.tile(mask_batch, [1, 8]), delta=1.0)
    tf.summary.scalar('corners loss', corners_loss)

    # Weighted sum of all losses
    total_loss = mask_loss + box_loss_weight * (center_loss + \
                                                heading_class_loss + size_class_loss + \
                                                heading_residual_normalized_loss * 20 + \
                                                size_residual_normalized_loss * 20 + \
                                                stage1_center_loss + \
                                                corner_loss_weight * corners_loss)
    tf.add_to_collection('losses', total_loss)

    return total_loss, mask_loss, center_loss, heading_class_loss, size_class_loss, heading_residual_normalized_loss * 20, size_residual_normalized_loss * 20, stage1_center_loss, corner_loss_weight * corners_loss, \
           end_points['center']


def get_loss(mask_label, mask_batch, center_label, \
             heading_class_label, heading_residual_label, \
             size_class_label, size_residual_label, \
             end_points, \
             corner_loss_weight=10.0, \
             box_loss_weight=1.0):
    ''' Loss functions for 3D object detection.
    Input:
        mask_label: TF int32 tensor in shape (B,N)
        center_label: TF tensor in shape (B,3)
        heading_class_label: TF int32 tensor in shape (B,) 
        heading_residual_label: TF tensor in shape (B,) 
        size_class_label: TF tensor int32 in shape (B,)
        size_residual_label: TF tensor tensor in shape (B,)
        end_points: dict, outputs from our model
        corner_loss_weight: float scalar
        box_loss_weight: float scalar
    Output:
        total_loss: TF scalar tensor
            the total_loss is also added to the losses collection
    '''
    # 3D Segmentation loss
    mask_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits( \
        logits=end_points['mask_logits'], labels=mask_label))
    tf.summary.scalar('3d mask loss', mask_loss)

    # Center regression losses
    center_dist = tf.norm(center_label - end_points['center'], axis=-1)
    center_loss = huber_loss(center_dist, delta=2.0)
    tf.summary.scalar('center loss', center_loss)
    stage1_center_dist = tf.norm(center_label - \
                                 end_points['stage1_center'], axis=-1)
    stage1_center_loss = huber_loss(stage1_center_dist, delta=1.0)
    tf.summary.scalar('stage1 center loss', stage1_center_loss)

    # Heading loss
    heading_class_loss = tf.reduce_mean( \
        tf.nn.sparse_softmax_cross_entropy_with_logits( \
            logits=end_points['heading_scores'], labels=heading_class_label))
    tf.summary.scalar('heading class loss', heading_class_loss)

    hcls_onehot = tf.one_hot(heading_class_label,
                             depth=NUM_HEADING_BIN,
                             on_value=1, off_value=0, axis=-1)  # BxNUM_HEADING_BIN
    heading_residual_normalized_label = \
        heading_residual_label / (np.pi / NUM_HEADING_BIN)
    heading_residual_normalized_loss = huber_loss(tf.reduce_sum( \
        end_points['heading_residuals_normalized'] * tf.to_float(hcls_onehot), axis=1) - \
                                                  heading_residual_normalized_label, delta=1.0)
    tf.summary.scalar('heading residual normalized loss',
                      heading_residual_normalized_loss)

    # Size loss
    size_class_loss = tf.reduce_mean( \
        tf.nn.sparse_softmax_cross_entropy_with_logits( \
            logits=end_points['size_scores'], labels=size_class_label))
    tf.summary.scalar('size class loss', size_class_loss)

    scls_onehot = tf.one_hot(size_class_label,
                             depth=NUM_SIZE_CLUSTER,
                             on_value=1, off_value=0, axis=-1)  # BxNUM_SIZE_CLUSTER
    scls_onehot_tiled = tf.tile(tf.expand_dims( \
        tf.to_float(scls_onehot), -1), [1, 1, 3])  # BxNUM_SIZE_CLUSTERx3
    predicted_size_residual_normalized = tf.reduce_sum( \
        end_points['size_residuals_normalized'] * scls_onehot_tiled, axis=[1])  # Bx3

    mean_size_arr_expand = tf.expand_dims( \
        tf.constant(g_mean_size_arr, dtype=tf.float32), 0)  # 1xNUM_SIZE_CLUSTERx3
    mean_size_label = tf.reduce_sum( \
        scls_onehot_tiled * mean_size_arr_expand, axis=[1])  # Bx3
    size_residual_label_normalized = size_residual_label / mean_size_label
    size_normalized_dist = tf.norm( \
        size_residual_label_normalized - predicted_size_residual_normalized,
        axis=-1)
    size_residual_normalized_loss = huber_loss(size_normalized_dist, delta=1.0)
    tf.summary.scalar('size residual normalized loss',
                      size_residual_normalized_loss)

    # Corner loss
    # We select the predicted corners corresponding to the
    # GT heading bin and size cluster.
    corners_3d = get_box3d_corners(end_points['center'],
                                   end_points['heading_residuals'],
                                   end_points['size_residuals'])  # (B,NH,NS,8,3)
    gt_mask = tf.tile(tf.expand_dims(hcls_onehot, 2), [1, 1, NUM_SIZE_CLUSTER]) * \
              tf.tile(tf.expand_dims(scls_onehot, 1), [1, NUM_HEADING_BIN, 1])  # (B,NH,NS)
    corners_3d_pred = tf.reduce_sum( \
        tf.to_float(tf.expand_dims(tf.expand_dims(gt_mask, -1), -1)) * corners_3d,
        axis=[1, 2])  # (B,8,3)

    heading_bin_centers = tf.constant( \
        np.arange(0, 2 * np.pi, 2 * np.pi / NUM_HEADING_BIN), dtype=tf.float32)  # (NH,)
    heading_label = tf.expand_dims(heading_residual_label, 1) + \
                    tf.expand_dims(heading_bin_centers, 0)  # (B,NH)
    heading_label = tf.reduce_sum(tf.to_float(hcls_onehot) * heading_label, 1)
    mean_sizes = tf.expand_dims( \
        tf.constant(g_mean_size_arr, dtype=tf.float32), 0)  # (1,NS,3)
    size_label = mean_sizes + \
                 tf.expand_dims(size_residual_label, 1)  # (1,NS,3) + (B,1,3) = (B,NS,3)
    size_label = tf.reduce_sum(tf.expand_dims(tf.to_float(scls_onehot), -1) * size_label, axis=[1])  # (B,3)
    corners_3d_gt = get_box3d_corners_helper( \
        center_label, heading_label, size_label)  # (B,8,3)
    corners_3d_gt_flip = get_box3d_corners_helper( \
        center_label, heading_label + np.pi, size_label)  # (B,8,3)

    corners_dist = tf.minimum(tf.norm(corners_3d_pred - corners_3d_gt, axis=-1),
                              tf.norm(corners_3d_pred - corners_3d_gt_flip, axis=-1))
    corners_loss = huber_loss(corners_dist, delta=1.0)
    tf.summary.scalar('corners loss', corners_loss)

    # Weighted sum of all losses
    total_loss = mask_loss + box_loss_weight * (center_loss + \
                                                heading_class_loss + size_class_loss + \
                                                heading_residual_normalized_loss * 20 + \
                                                size_residual_normalized_loss * 20 + \
                                                stage1_center_loss + \
                                                corner_loss_weight * corners_loss)
    tf.add_to_collection('losses', total_loss)
    return total_loss, mask_loss, center_loss, heading_class_loss, size_class_loss, heading_residual_normalized_loss * 20, size_residual_normalized_loss * 20, stage1_center_loss, corner_loss_weight * corners_loss, \
           end_points['center']


def get_loss_seg(mask_label, end_points):
    ''' Loss functions for 3D object detection.
    Input:
        mask_label: TF int32 tensor in shape (B,N)
        center_label: TF tensor in shape (B,3)
        heading_class_label: TF int32 tensor in shape (B,)
        heading_residual_label: TF tensor in shape (B,)
        size_class_label: TF tensor int32 in shape (B,)
        size_residual_label: TF tensor tensor in shape (B,)
        end_points: dict, outputs from our model
        corner_loss_weight: float scalar
        box_loss_weight: float scalar
    Output:
        total_loss: TF scalar tensor
            the total_loss is also added to the losses collection
    '''
    # 3D Segmentation loss
    mask_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits( \
        logits=end_points['mask_logits'], labels=mask_label))
    tf.summary.scalar('3d mask loss', mask_loss)

    # Weighted sum of all losses
    total_loss = mask_loss
    tf.add_to_collection('losses', total_loss)
    return total_loss


def get_iou_AB(ab_corners, ab_ids, gt_ids, gt_corners):
    iou_AB = np.zeros((ab_corners.shape[0]), dtype=np.float32)
    score = np.zeros((ab_corners.shape[0]), dtype=np.int32)

    for i in range(ab_corners.shape[0]):
        idx_gt = np.where(gt_ids == ab_ids[i])
        iou_max = 0
        if (len(idx_gt[0]) != 0):
            gt_corner = gt_corners[idx_gt[0][0]]
            for j in range(len(gt_corner)):
                if (np.count_nonzero(gt_corner[j, 0] != 0.0) > 0):
                    iou_3d, iou_2d = box3d_iou(ab_corners[i], gt_corner[j])
                    if iou_3d > iou_max:
                        iou_max = iou_3d
            if iou_max < 0.2:
                score[i] = 0

            elif (iou_max > 0.35):
                score[i] = 1



    return score


def get_loss_seg_cls(mask_label, gt_corners, end_points):
    ''' Loss functions for 3D object detection.
    Input:
        mask_label: TF int32 tensor in shape (B,N)
        center_label: TF tensor in shape (B,3)
        heading_class_label: TF int32 tensor in shape (B,)
        heading_residual_label: TF tensor in shape (B,)
        size_class_label: TF tensor int32 in shape (B,)
        size_residual_label: TF tensor tensor in shape (B,)
        end_points: dict, outputs from our model
        corner_loss_weight: float scalar
        box_loss_weight: float scalar
    Output:
        total_loss: TF scalar tensor
            the total_loss is also added to the losses collection
    '''
    # 3D Segmentation loss
    mask_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits( \
        logits=end_points['mask_logits'], labels=mask_label))
    tf.summary.scalar('3d mask loss', mask_loss)
    score = tf.py_func(get_iou_AB,
                               [end_points['ab_corners'], end_points['ab_ids'], end_points['gt_ids'], gt_corners],
                               [tf.int32])
    #print("score",score.shape)
    score_=tf.reshape(score,[n_batches*n_AB])
    print("score", score_.shape)
    end_points['score_label']=score
    cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=end_points['ab_cls'], labels=score_))

    tf.summary.scalar('cls_loss', cls_loss)
    # Weighted sum of all losses
    total_loss = mask_loss + cls_loss
    tf.add_to_collection('losses', total_loss)

    return total_loss


def get_loss_bbox(center_label, \
                  heading_class_label, heading_residual_label, \
                  size_class_label, size_residual_label, \
                  end_points, \
                  corner_loss_weight=10.0, \
                  box_loss_weight=1.0):
    ''' Loss functions for 3D object detection.
    Input:
        mask_label: TF int32 tensor in shape (B,N)
        center_label: TF tensor in shape (B,3)
        heading_class_label: TF int32 tensor in shape (B,)
        heading_residual_label: TF tensor in shape (B,)
        size_class_label: TF tensor int32 in shape (B,)
        size_residual_label: TF tensor tensor in shape (B,)
        end_points: dict, outputs from our model
        corner_loss_weight: float scalar
        box_loss_weight: float scalar
    Output:
        total_loss: TF scalar tensor
            the total_loss is also added to the losses collection
    '''
    # 3D Segmentation loss
    # mask_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits( \
    #    logits=end_points['mask_logits'], labels=mask_label))
    # tf.summary.scalar('3d mask loss', mask_loss)

    # Center regression losses
    center_dist = tf.norm(center_label - end_points['center'], axis=-1)
    center_loss = huber_loss(center_dist, delta=2.0)
    tf.summary.scalar('center loss', center_loss)
    stage1_center_dist = tf.norm(center_label - \
                                 end_points['stage1_center'], axis=-1)
    stage1_center_loss = huber_loss(stage1_center_dist, delta=1.0)
    tf.summary.scalar('stage1 center loss', stage1_center_loss)

    # Heading loss
    heading_class_loss = tf.reduce_mean( \
        tf.nn.sparse_softmax_cross_entropy_with_logits( \
            logits=end_points['heading_scores'], labels=heading_class_label))
    tf.summary.scalar('heading class loss', heading_class_loss)

    hcls_onehot = tf.one_hot(heading_class_label,
                             depth=NUM_HEADING_BIN,
                             on_value=1, off_value=0, axis=-1)  # BxNUM_HEADING_BIN
    heading_residual_normalized_label = \
        heading_residual_label / (np.pi / NUM_HEADING_BIN)
    heading_residual_normalized_loss = huber_loss(tf.reduce_sum( \
        end_points['heading_residuals_normalized'] * tf.to_float(hcls_onehot), axis=1) - \
                                                  heading_residual_normalized_label, delta=1.0)
    tf.summary.scalar('heading residual normalized loss',
                      heading_residual_normalized_loss)

    # Size loss
    size_class_loss = tf.reduce_mean( \
        tf.nn.sparse_softmax_cross_entropy_with_logits( \
            logits=end_points['size_scores'], labels=size_class_label))
    tf.summary.scalar('size class loss', size_class_loss)

    scls_onehot = tf.one_hot(size_class_label,
                             depth=NUM_SIZE_CLUSTER,
                             on_value=1, off_value=0, axis=-1)  # BxNUM_SIZE_CLUSTER
    scls_onehot_tiled = tf.tile(tf.expand_dims( \
        tf.to_float(scls_onehot), -1), [1, 1, 3])  # BxNUM_SIZE_CLUSTERx3
    predicted_size_residual_normalized = tf.reduce_sum( \
        end_points['size_residuals_normalized'] * scls_onehot_tiled, axis=[1])  # Bx3

    mean_size_arr_expand = tf.expand_dims( \
        tf.constant(g_mean_size_arr, dtype=tf.float32), 0)  # 1xNUM_SIZE_CLUSTERx3
    mean_size_label = tf.reduce_sum( \
        scls_onehot_tiled * mean_size_arr_expand, axis=[1])  # Bx3
    size_residual_label_normalized = size_residual_label / mean_size_label
    size_normalized_dist = tf.norm( \
        size_residual_label_normalized - predicted_size_residual_normalized,
        axis=-1)
    size_residual_normalized_loss = huber_loss(size_normalized_dist, delta=1.0)
    tf.summary.scalar('size residual normalized loss',
                      size_residual_normalized_loss)

    # Corner loss
    # We select the predicted corners corresponding to the
    # GT heading bin and size cluster.
    corners_3d = get_box3d_corners(end_points['center'],
                                   end_points['heading_residuals'],
                                   end_points['size_residuals'])  # (B,NH,NS,8,3)
    gt_mask = tf.tile(tf.expand_dims(hcls_onehot, 2), [1, 1, NUM_SIZE_CLUSTER]) * \
              tf.tile(tf.expand_dims(scls_onehot, 1), [1, NUM_HEADING_BIN, 1])  # (B,NH,NS)
    corners_3d_pred = tf.reduce_sum( \
        tf.to_float(tf.expand_dims(tf.expand_dims(gt_mask, -1), -1)) * corners_3d,
        axis=[1, 2])  # (B,8,3)

    heading_bin_centers = tf.constant( \
        np.arange(0, 2 * np.pi, 2 * np.pi / NUM_HEADING_BIN), dtype=tf.float32)  # (NH,)
    heading_label = tf.expand_dims(heading_residual_label, 1) + \
                    tf.expand_dims(heading_bin_centers, 0)  # (B,NH)
    heading_label = tf.reduce_sum(tf.to_float(hcls_onehot) * heading_label, 1)
    mean_sizes = tf.expand_dims( \
        tf.constant(g_mean_size_arr, dtype=tf.float32), 0)  # (1,NS,3)
    size_label = mean_sizes + \
                 tf.expand_dims(size_residual_label, 1)  # (1,NS,3) + (B,1,3) = (B,NS,3)
    size_label = tf.reduce_sum(tf.expand_dims(tf.to_float(scls_onehot), -1) * size_label, axis=[1])  # (B,3)
    corners_3d_gt = get_box3d_corners_helper( \
        center_label, heading_label, size_label)  # (B,8,3)
    corners_3d_gt_flip = get_box3d_corners_helper( \
        center_label, heading_label + np.pi, size_label)  # (B,8,3)

    corners_dist = tf.minimum(tf.norm(corners_3d_pred - corners_3d_gt, axis=-1),
                              tf.norm(corners_3d_pred - corners_3d_gt_flip, axis=-1))
    corners_loss = huber_loss(corners_dist, delta=1.0)
    tf.summary.scalar('corners loss', corners_loss)

    # Weighted sum of all losses
    total_loss = (center_loss + heading_class_loss + size_class_loss + \
                  heading_residual_normalized_loss * 20 + \
                  size_residual_normalized_loss * 20 + \
                  stage1_center_loss + \
                  corner_loss_weight * corners_loss)
    tf.add_to_collection('losses', total_loss)
    return total_loss
