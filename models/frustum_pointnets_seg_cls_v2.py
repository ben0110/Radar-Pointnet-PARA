''' Frustum PointNets v2 Model.
'''
from __future__ import print_function

import sys
import os
import tensorflow as tf
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_sa_module_msg, pointnet_fp_module
from model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER, NUM_OBJECT_POINT
from model_util import extract_proposals,n_batches,n_radar_set,n_AB,n_points_AB,tf_gather_object_pc_feats
from model_util import placeholder_inputs_seg_cls, parse_output_to_tensors, get_loss_batch,get_loss,get_loss_seg,placeholder_inputs_seg_cls,get_loss_seg_cls


def get_instance_seg_v2_net(point_cloud, one_hot_vec,
                            is_training, bn_decay, end_points):
    ''' 3D instance segmentation PointNet v2 network.
    Input:
        point_cloud: TF tensor in shape (B,N,4)
            frustum point clouds with XYZ and intensity in point channels
            XYZs are in frustum coordinate
        one_hot_vec: TF tensor in shape (B,3)
            length-3 vectors indicating predicted object type
        is_training: TF boolean scalar
        bn_decay: TF float scalar
        end_points: dict
    Output:
        logits: TF tensor in shape (B,N,2), scores for bkg/clutter and object
        end_points: dict
    '''

    l0_xyz = tf.slice(point_cloud, [0,0,0], [-1,-1,3])
    l0_points = tf.slice(point_cloud, [0,0,3], [-1,-1,1])

    # Set abstraction layers
    l1_xyz, l1_points = pointnet_sa_module_msg(l0_xyz, l0_points,
        128, [0.2,0.4,0.8], [32,64,128],
        [[32,32,64], [64,64,128], [64 ,96,128]],
        is_training, bn_decay, scope='layer1')
    l2_xyz, l2_points = pointnet_sa_module_msg(l1_xyz, l1_points,
        32, [0.4,0.8,1.6], [64,64,128],
        [[64,64,128], [128,128,256], [128,128,256]],
        is_training, bn_decay, scope='layer2')
    l3_xyz, l3_points, _ = pointnet_sa_module(l2_xyz, l2_points,
        npoint=None, radius=None, nsample=None, mlp=[128,256,1024],
        mlp2=None, group_all=True, is_training=is_training,
        bn_decay=bn_decay, scope='layer3')

    # Feature Propagation layers
    l3_points = tf.concat([l3_points, tf.expand_dims(one_hot_vec, 1)], axis=2)
    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points,
        [128,128], is_training, bn_decay, scope='fa_layer1')
    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points,
        [128,128], is_training, bn_decay, scope='fa_layer2')
    l0_points = pointnet_fp_module(l0_xyz, l1_xyz,
        tf.concat([l0_xyz,l0_points],axis=-1), l1_points,
        [128,128], is_training, bn_decay, scope='fa_layer3')
    end_points['feats'] = l0_points
    # FC layers
    #print("l0_points",l0_points.shape)
    #print("l0_points",l0_points)
    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True,
        is_training=is_training, scope='conv1d-fc1', bn_decay=bn_decay)
    #print("l0_points",l0_points.shape)
    net = tf_util.dropout(net, keep_prob=0.7,
        is_training=is_training, scope='dp1')
    logits = tf_util.conv1d(net, 2, 1,
        padding='VALID', activation_fn=None, scope='conv1d-fc2')

    return logits, end_points

def get_3d_box_estimation_v2_net(object_point_cloud, one_hot_vec,
                                 is_training, bn_decay, end_points):
    ''' 3D Box Estimation PointNet v2 network.
    Input:
        object_point_cloud: TF tensor in shape (B,M,C)
            masked point clouds in object coordinate
        one_hot_vec: TF tensor in shape (B,3)
            length-3 vectors indicating predicted object type
    Output:
        output: TF tensor in shape (B,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4)
            including box centers, heading bin class scores and residuals,
            and size cluster scores and residuals
    '''
    # Gather object points
    print("one_hot_vec: ", one_hot_vec)
    batch_size = object_point_cloud.get_shape()[0].value

    l0_xyz = object_point_cloud
    l0_points = None
    # Set abstraction layers
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points,
        npoint=128, radius=0.2, nsample=64, mlp=[64,64,128],
        mlp2=None, group_all=False,
        is_training=is_training, bn_decay=bn_decay, scope='ssg-layer1')
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points,
        npoint=32, radius=0.4, nsample=64, mlp=[128,128,256],
        mlp2=None, group_all=False,
        is_training=is_training, bn_decay=bn_decay, scope='ssg-layer2')
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points,
        npoint=None, radius=None, nsample=None, mlp=[256,256,512],
        mlp2=None, group_all=True,
        is_training=is_training, bn_decay=bn_decay, scope='ssg-layer3')

    # Fully connected layers
    net = tf.reshape(l3_points, [batch_size, -1])
    net = tf.concat([net, one_hot_vec], axis=1)
    net = tf_util.fully_connected(net, 512, bn=True,
        is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True,
        is_training=is_training, scope='fc2', bn_decay=bn_decay)

    # The first 3 numbers: box center coordinates (cx,cy,cz),
    # the next NUM_HEADING_BIN*2:  heading bin class scores and bin residuals
    # next NUM_SIZE_CLUSTER*4: box cluster scores and residuals
    output = tf_util.fully_connected(net,
        3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4, activation_fn=None, scope='fc3')
    return output, end_points

def get_cls_net(point_cloud,ids,radar_rois,radar_set,logits,is_training,bn_decay,end_points):
    mask = tf.slice(logits, [0, 0, 0], [-1, -1, 1]) < \
           tf.slice(logits, [0, 0, 1], [-1, -1, 1])
    mask = tf.to_float(mask)
    mask = tf.squeeze(mask, axis=[2])  # BxN
    end_points['mask'] = mask
    #object_point_cloud, _ = tf_gather_object_pc(point_cloud,
    #                                            mask, NUM_OBJECT_POINT)
    #object_point_cloud,object_point_cloud_features,idx_list,radar_rois_list, _ = tf_gather_object_pc_feats(point_cloud,end_points['feats'],
    #                                            mask,radar_set,radar_rois,ids, NUM_OBJECT_POINT)
    #ab_pc,ab_pc_feat,ab_corners,ab_ids,non_zero_AB = extract_proposals(object_point_cloud,object_point_cloud_features,radar_rois_list,idx_list)
    ab_pc, ab_pc_feat, ab_corners, ab_ids = extract_proposals(point_cloud,end_points['feats'],
                                                mask,radar_set,radar_rois,ids, NUM_OBJECT_POINT)
    # get sure of ab_pc hs a static of batches
    end_points['ab_corners']= ab_corners
    end_points['ab_ids']= ab_ids
    #print("ab_pc_feat",ab_pc_feat)

    #net=ab_pc_feat.set_shape([100, 512, 3])
    #net = tf.reshape(ab_pc_feat, [n_batches*n_radar_set*n_AB,n_points_AB,128])
    #ab_pc_feat.set_shape([n_batches*n_radar_set*n_AB,n_points_AB,128])
    ab_pc.set_shape([n_batches*n_AB,n_points_AB,3])
    batch_size = ab_pc.get_shape()[0].value
    #net = ab_pc_feat
    #print("net",ab_pc.shape)
    #print("net", ab_pc  )
    l0_xyz = ab_pc
    l0_points = None
    # Set abstraction layers
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points,
                                                       npoint=128, radius=0.2, nsample=64, mlp=[64, 64, 128],
                                                       mlp2=None, group_all=False,
                                                       is_training=is_training, bn_decay=bn_decay, scope='ssg-layer1_rpn')
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points,
                                                       npoint=32, radius=0.4, nsample=64, mlp=[128, 128, 256],
                                                       mlp2=None, group_all=False,
                                                       is_training=is_training, bn_decay=bn_decay, scope='ssg-layer2_rpn')
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points,
                                                       npoint=None, radius=None, nsample=None, mlp=[256, 256, 512],
                                                       mlp2=None, group_all=True,
                                                       is_training=is_training, bn_decay=bn_decay, scope='ssg-layer3_rpn')
    #print("l3_points", l3_points.shape)
    #print("l3_points", l3_points)
    #print(batch_size)
    # Fully connected layers
    net = tf.reshape(l3_points, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True,
                                  is_training=is_training, scope='fc1_rpn', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True,
                                  is_training=is_training, scope='fc2_rpn', bn_decay=bn_decay)

    # The first 3 numbers: box center coordinates (cx,cy,cz),
    # the next NUM_HEADING_BIN*2:  heading bin class scores and bin residuals
    # next NUM_SIZE_CLUSTER*4: box cluster scores and residuals

    end_points['ab_cls'] = tf_util.fully_connected(net,
                                     2, activation_fn=None, scope='fc3_rpn')
    return end_points

def get_model(point_cloud,ids,radar_rois,radar_set, one_hot_vec, is_training, bn_decay=None):
    ''' Frustum PointNets model. The model predict 3D object masks and
    amodel bounding boxes for objects in frustum point clouds.

    Input:
        point_cloud: TF tensor in shape (B,N,4)
            frustum point clouds with XYZ and intensity in point channels
            XYZs are in frustum coordinate
        one_hot_vec: TF tensor in shape (B,3)
            length-3 vectors indicating predicted object type
        is_training: TF boolean scalar
        bn_decay: TF float scalar
    Output:
        end_points: dict (map from name strings to TF tensors)
    '''
    end_points = {}

    # 3D Instance Segmentation PointNet
    logits, end_points = get_instance_seg_v2_net(\
        point_cloud, one_hot_vec,
        is_training, bn_decay, end_points)
    end_points['mask_logits'] = logits
    end_points = get_cls_net(point_cloud,ids,radar_rois,radar_set,logits,is_training, bn_decay,end_points)
    end_points['gt_ids'] = ids

    return end_points

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,4))
        radar_rois = tf.ones((32,7))
        outputs = get_model(inputs,tf.ones((32,1)),radar_rois, tf.ones((32,3)), tf.constant(True))
        for key in outputs:
            print((key, outputs[key]))
        loss = get_loss(tf.ones((32,1024),dtype=tf.int32),
            tf.ones((32,3)), tf.ones((32,),dtype=tf.int32),
            tf.ones((32,)), tf.ones((32,),dtype=tf.int32),
            tf.ones((32,3)), outputs)
        print(loss)
