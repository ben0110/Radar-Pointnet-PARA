''' Evaluating Frustum PointNets.
Write evaluation results to KITTI format labels.
and [optionally] write results to pickle files.

Author: Charles R. Qi
Date: September 2017
'''
from __future__ import print_function

import os
import sys
import argparse
import importlib
import numpy as np
import tensorflow as tf
import cPickle as pickle
from shutil import copyfile
import json
import time
import math
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
#sys.path.append(os.path.join(ROOT_DIR, '/train/log_v2/15-02-2020-23:59:37/'))
from model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER
import provider_dfd_batch_train_overall as  provider
from train_util import get_batch_test
import box_util as box_util

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
parser.add_argument('--model', default='frustum_pointnets_v1', help='Model name [default: frustum_pointnets_v1]')
parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for inference [default: 32]')
parser.add_argument('--output', default='test_results', help='output file/folder name [default: test_results]')
parser.add_argument('--data_path', default=None, help='frustum dataset pickle filepath [default: None]')
parser.add_argument('--from_rgb_detection', action='store_true', help='test from dataset files from rgb detection.')
parser.add_argument('--idx_path', default=None, help='filename of txt where each line is a data idx, used for rgb detection -- write <id>.txt for all frames. [default: None]')
parser.add_argument('--dump_result', action='store_true', help='If true, also dump results to .pickle file')
FLAGS = parser.parse_args()

# Set training configurations
BATCH_SIZE = FLAGS.batch_size
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
NUM_POINT = FLAGS.num_point
sys.path.append("/")
MODEL = importlib.import_module(FLAGS.model)
NUM_CLASSES = 2
NUM_CHANNEL = 4

pathsplit= MODEL_PATH.split('/')

OUTPUT_FILE = os.path.join('/',pathsplit[1],pathsplit[2],pathsplit[3],pathsplit[4],pathsplit[5],'results/')
if not os.path.exists(OUTPUT_FILE):
    os.mkdir(OUTPUT_FILE)
# Load Frustum Datasets.
TEST_DATASET = provider.FrustumDataset('pc_radar_2',npoints=NUM_POINT, split='val',
    rotate_to_center=False, one_hot=True,all_batches = True,translate_radar_center=False,store_data=True,proposals_3 =True,no_color=True)

def get_session_and_ops(batch_size, num_point):
    ''' Define model graph, load model parameters,
    create session and return session handle and tensors
    '''
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, one_hot_vec_pl, labels_pl, centers_pl, \
            heading_class_label_pl, heading_residual_label_pl, \
            size_class_label_pl, size_residual_label_pl ,batch_mask_pl= \
                MODEL.placeholder_inputs_batch(batch_size, num_point)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            end_points = MODEL.get_model(pointclouds_pl, one_hot_vec_pl,
                is_training_pl)
            loss = MODEL.get_loss_batch(labels_pl,batch_mask_pl, centers_pl,
                heading_class_label_pl, heading_residual_label_pl,
                size_class_label_pl, size_residual_label_pl, end_points)
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)

        # Restore variables from disk.
        saver.restore(sess, MODEL_PATH)
        ops = {'pointclouds_pl': pointclouds_pl,
               'one_hot_vec_pl': one_hot_vec_pl,
               'labels_pl': labels_pl,
               'centers_pl': centers_pl,
               'heading_class_label_pl': heading_class_label_pl,
               'heading_residual_label_pl': heading_residual_label_pl,
               'size_class_label_pl': size_class_label_pl,
               'size_residual_label_pl': size_residual_label_pl,
               'is_training_pl': is_training_pl,
               'logits': end_points['mask_logits'],
               'center': end_points['center'],
               'end_points': end_points,
               'loss': loss}
        return sess, ops

def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape)-1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape)-1, keepdims=True)
    return probs



def write_detection_results(result_dir, id_list, type_list, box2d_list, center_list, \
                            heading_cls_list, heading_res_list, \
                            size_cls_list, size_res_list, \
                            rot_angle_list, score_list):
    ''' Write frustum pointnets results to KITTI format label files. '''
    if result_dir is None: return
    results = {} # map from idx to list of strings, each string is a line (without \n)
    for i in range(len(center_list)):
        idx = id_list[i]
        output_str = type_list[i] + " -1 -1 -10 "
        box2d = box2d_list[i]
        output_str += "%f %f %f %f " % (box2d[0],box2d[1],box2d[2],box2d[3])
        h,w,l,tx,ty,tz,ry = provider.from_prediction_to_label_format(center_list[i],
            heading_cls_list[i], heading_res_list[i],
            size_cls_list[i], size_res_list[i], rot_angle_list[i])
        score = score_list[i]
        output_str += "%f %f %f %f %f %f %f %f" % (h,w,l,tx,ty,tz,ry,score)
        if idx not in results: results[idx] = []
        results[idx].append(output_str)

    # Write TXT files
    if not os.path.exists(result_dir): os.mkdir(result_dir)
    output_dir = os.path.join(result_dir, 'data')
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    for idx in results:
        pred_filename = os.path.join(output_dir, '%06d.txt'%(idx))
        fout = open(pred_filename, 'w')
        for line in results[idx]:
            fout.write(line+'\n')
        fout.close()

def write_detection_results_kitti(result_dir, id_list,  center_list, \
                            heading_cls_list, heading_res_list, \
                            size_cls_list, size_res_list, \
                            rot_angle_list, score_list,batch_seg_sum_frames):
    ''' Write frustum pointnets results to KITTI format label files. '''
    if result_dir is None: return
    results = {} # map from idx to list of strings, each string is a line (without \n)
    for i in range(len(id_list)):
        for j in range(len(center_list[i])):
            print(id_list[i])
            if(batch_seg_sum_frames[i][j]<30):
                print("3asba")
                continue
            print("boom")
            idx = id_list[i]
            output_str ="Pedestrian -1 -1 -10 "
            output_str += "0.0 0.0 0.0 0.0 "
            h,w,l,tx,ty,tz,ry = provider.from_prediction_to_label_format(center_list[i][j],
                heading_cls_list[i][j], heading_res_list[i][j],
                size_cls_list[i][j], size_res_list[i][j], 0.0)
            score = score_list[i][j]
            x = tz
            y = -tx
            z = -ty
            th = h
            tw = w
            tl = l
            output_str += "%f %f %f %f %f %f %f %f" % (th,tl,tw,x,y,z,ry,score)
            if idx not in results: results[idx] = []
            results[idx].append(output_str)

    # Write TXT files
    if not os.path.exists(result_dir): os.mkdir(result_dir)
    output_dir = os.path.join(result_dir, 'data')
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    for idx in results:
        pred_filename = os.path.join(output_dir, '%06d.txt'%(idx))
        fout = open(pred_filename, 'w')
        for line in results[idx]:
            fout.write(line+'\n')
        fout.close()


def write_detection_results_3dbat(result_dir, id_list,  center_list, \
                            heading_cls_list, heading_res_list, \
                            size_cls_list, size_res_list, \
                            rot_angle_list, score_list):
    orig_path = "/root/3D_BoundingBox_Annotation_Tool_3D_BAT/input/NuScenes/ONE/"
    # copy files to 3d bbat
    i=0
    for idx in id_list:
        #copy pcd file
        copyfile("/root/frustum-pointnets_RSC/dataset/KITTI/object/training/velodyne/%06d.pcd" %idx ,orig_path+"/pointclouds/%06d.pcd" %i)
        copyfile("/root/frustum-pointnets_RSC/dataset/KITTI/object/training/image_left/%06d.jpg" %idx ,orig_path+"/images/CAM_FRONT_LEFT/%06d.jpg" %i)
        copyfile("/root/frustum-pointnets_RSC/dataset/KITTI/object/training/image_right/%06d.jpg" % idx,orig_path + "/images/CAM_FRONT/%06d.jpg" % i)
        i=i+1
        #copy image or create sym link zum image: watch out that you dont broke the links

    # generate json annotation
    annotation = []
    for i in range(len(center_list)):
        h,w,l,tx,ty,tz,ry = provider.from_prediction_to_label_format(center_list[i],
            heading_cls_list[i], heading_res_list[i],
            size_cls_list[i], size_res_list[i],0.0)
        x = tz
        y = -tx
        z = -ty

        th = h
        tw = l
        tl = w
        annotation.append([{"class":"Pedestrian","width":tw,"length":tl,"height":th,"x":x,"y":y,"z":z,"rotationY":ry,"trackId":1,"frameIdx":i}])

    with open(orig_path+ "/annotations/NuScenes_ONE_annotations.txt", 'w') as outfile:
        json.dump(annotation, outfile)

def write_detection_results_batch_3dbat(result_dir, id_list_frame,  center_list_frame, \
                            heading_cls_list_frame, heading_res_list_frame, \
                            size_cls_list_frame, size_res_list_frame, \
                            rot_angle_list_frame, score_list_frame,scores_mask_mean_frames,scores_heading_frames,scores_size_frames,batch_seg_sum_frames):
    orig_path = "/root/3D_BoundingBox_Annotation_Tool_3D_BAT/input/NuScenes/ONE/"
    kitti_orig_path = "/root/"
    i = 0

    # copy files to 3d bbat
    for idx in range(len(id_list_frame)):
        # copy pcd file
        print("id_list_frame[idx]:",id_list_frame[idx])
        copyfile("/root/frustum-pointnets_RSC/dataset/KITTI/object/training/velodyne/%06d.pcd" % id_list_frame[idx],
                 orig_path + "/pointclouds/%06d.pcd" % i)
        copyfile("/root/frustum-pointnets_RSC/dataset/KITTI/object/training/image_left/%06d.jpg" % id_list_frame[idx],
                 orig_path + "/images/CAM_FRONT_LEFT/%06d.jpg" % i)
        copyfile("/root/frustum-pointnets_RSC/dataset/KITTI/object/training/image_right/%06d.jpg" % id_list_frame[idx],
                 orig_path + "/images/CAM_FRONT/%06d.jpg" % i)
        i = i + 1

    annotation = []

    # generate json annotation
    for j in range(len(id_list_frame)):
        annotation_frame = []
        output_results_path=OUTPUT_FILE + "%06d.txt" % id_list_frame[j]
        line=[]
        for i in range(len(center_list_frame[j])):
            if(batch_seg_sum_frames<30):
                print("3asba")
            h, w, l, tx, ty, tz, ry = provider.from_prediction_to_label_format(center_list_frame[j][i],
                                                                               heading_cls_list_frame[j][i],
                                                                               heading_res_list_frame[j][i],
                                                                               size_cls_list_frame[j][i],
                                                                               size_res_list_frame[j][i],
                                                                               0.0)
            print(h,w,l,tx,ty,tz)
            x = tz
            z = -ty
            y = -tx
            th = h
            tw = l
            tl = w
            line.append("Pedestrian 0.0 0.0 0.0 0.0 ")
            annotation_frame.append(
                {"class": "Pedestrian", "width": tw, "length": tl, "height": th, "x": x, "y": y, "z": z,
                 "rotationY": ry, "trackId": i, "frameIdx": j})
        annotation.append(annotation_frame)


    with open(orig_path + "/annotations/NuScenes_ONE_annotations.txt", 'w') as outfile:
        json.dump(annotation, outfile)






def fill_files(output_dir, to_fill_filename_list):
    ''' Create empty files if not exist for the filelist. '''
    for filename in to_fill_filename_list:
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            fout = open(filepath, 'w')
            fout.close()



def NMS(result_dir, id_list_frame, center_list_frame, \
        heading_cls_list_frame, heading_res_list_frame, \
        size_cls_list_frame, size_res_list_frame, \
        rot_angle_list_frame, score_list_frame,scores_mask_mean_frames,batch_seg_sum_frames):

    # estimate corners for all detections in box
    for j in range(len(id_list_frame[:5])):
    # estimate 3DIoU for a box with other boxes for a batch
        print(center_list_frame[j])
        corners3d = np.zeros((len(center_list_frame[j]),8,3))
        for i in range(len(center_list_frame[j])):
            corners3d[i]=provider.get_3d_box_batch(center_list_frame[j][i],heading_cls_list_frame[j][i], heading_res_list_frame[j][i],
                                            size_cls_list_frame[j][i], size_res_list_frame[j][i],rot_angle_list_frame[j][i])
        bboxes = []
        center_list = []
        heading_cls_list =[]
        heading_res_list = []
        size_cls_list =[]
        size_res_list =[]
        rot_angle_list = []
        score_list = []
        #ind_sort = np.argsort(-scores_mask_mean_frames[j])
        ind_sort = np.argsort([x*-1.0 for x in scores_mask_mean_frames[j]])
        print(ind_sort)
        for i in range(len(scores_mask_mean_frames[j])):
            print( scores_mask_mean_frames[j][ind_sort[i]])
        #print(scores_mask_mean_frames[j][ind_sort])
        for i in range(len(corners3d)):
            bbox = corners3d[ind_sort[i]]
            flag = 1
            for k in range(i+1,len(corners3d)):
                if(batch_seg_sum_frames[j][ind_sort[i]]< 30):
                    flag = -1
                    break
                if(np.array_equal(bbox, corners3d[ind_sort[k]])):
                    flag = -1
                    break
                print("index ",ind_sort[i],scores_mask_mean_frames[j][ind_sort[i]], "index _comp: ", ind_sort[k], scores_mask_mean_frames[j][ind_sort[k]],"IoU: ",box_util.box3d_iou(bbox,corners3d[ind_sort[k]]))
                if box_util.box3d_iou(bbox,corners3d[ind_sort[k]])[1] > 0.3:
                    flag = -1
                    break
            if flag == 1:
                bboxes.append(bbox)
                center_list.append(center_list_frame[j][ind_sort[i]])
                heading_cls_list.append(heading_cls_list_frame[j][ind_sort[i]])
                heading_res_list.append(heading_res_list_frame[j][ind_sort[i]])
                size_cls_list.append(size_cls_list_frame[j][ind_sort[i]])
                size_res_list.append(size_res_list_frame[j][ind_sort[i]])
                rot_angle_list.append(rot_angle_list_frame[j][ind_sort[i]])
                #score_list.append(score_list_frame[j][i])
            print("boxes size:", len(bboxes))

        center_list_frame[j]= center_list
        heading_cls_list_frame[j] = heading_cls_list
        heading_res_list_frame[j] = heading_res_list
        size_cls_list_frame[j] = size_cls_list
        size_res_list_frame[j] = size_res_list
        rot_angle_list_frame[j] = rot_angle_list
        #score_list_frame[j] = score_list
    return center_list_frame, \
        heading_cls_list_frame, heading_res_list_frame, \
        size_cls_list_frame, size_res_list_frame, \
        rot_angle_list_frame, score_list_frame


def test_batch(output_filename, result_dir=None):
    ''' Test frustum pointnets with GT 2D boxes.
    Write test results to KITTI format label files.
    todo (rqi): support variable number of points.
    '''
    ps_list_frames = []
    seg_list_frames = []
    segp_list_frames = []
    center_list_frames = []
    heading_cls_list_frames = []
    heading_res_list_frames = []
    size_cls_list_frames = []
    size_res_list_frames = []
    rot_angle_list_frames = []
    score_list_frames = []
    scores_mask_mean_frames = []
    scores_heading_frames = []
    scores_size_frames = []
    batch_seg_sum_frames = []
    start_idx = 0
    for i in range(len(TEST_DATASET.idx_batch)):
        ps_list = []
        seg_list = []
        segp_list = []
        center_list = []
        heading_cls_list = []
        heading_res_list = []
        size_cls_list = []
        size_res_list = []
        rot_angle_list = []
        score_list = []
        scores_mask_mean_list = []
        scores_heading_list = []
        scores_size_list=[]
        batch_seg_sum_list =[]
        test_idxs = np.arange(0, len(TEST_DATASET))
        batch_size = TEST_DATASET.batch_size[i]
        #num_batches = len(TEST_DATASET)/batch_size

        sess, ops = get_session_and_ops(batch_size=batch_size, num_point=NUM_POINT)
        correct_cnt = 0
        print('batch idx: %d' % (i))
        print('batch start' ,start_idx)

        #start_idx = 0
        end_idx = start_idx + batch_size
        print('end_idx',end_idx)
        batch_data,batch_rot_angle,batch_label, batch_one_hot_vec, = \
            get_batch_test(TEST_DATASET, test_idxs, start_idx, end_idx,
                NUM_POINT, NUM_CHANNEL)

        batch_output, batch_center_pred, \
            batch_hclass_pred, batch_hres_pred, \
            batch_sclass_pred, batch_sres_pred, batch_scores,batch_scores_mask_mean,batch_scores_heading,batch_scores_size,batch_seg_sum = \
                    inference(sess, ops, batch_data,
                    batch_one_hot_vec, batch_size=batch_size)

        correct_cnt += np.sum(batch_output==batch_label)
        print("batch correct_cnt: ",correct_cnt)
        start_idx = start_idx+batch_size

        for i in range(batch_output.shape[0]):
            ps_list.append(batch_data[i,...])
            segp_list.append(batch_output[i,...])
            center_list.append(batch_center_pred[i,:])
            heading_cls_list.append(batch_hclass_pred[i])
            heading_res_list.append(batch_hres_pred[i])
            size_cls_list.append(batch_sclass_pred[i])
            size_res_list.append(batch_sres_pred[i,:])
            rot_angle_list.append(batch_rot_angle[i])
            score_list.append(batch_scores[i])
            scores_mask_mean_list.append(batch_scores_mask_mean[i])
            scores_heading_list.append(batch_scores_heading[i])
            scores_size_list.append(batch_scores_size[i])
            batch_seg_sum_list.append(batch_seg_sum[i])
        ps_list_frames.append(ps_list)
        seg_list_frames.append(seg_list)
        segp_list_frames.append(segp_list)
        center_list_frames.append(center_list)
        heading_cls_list_frames.append(heading_cls_list)
        heading_res_list_frames.append(heading_res_list)
        size_cls_list_frames.append(size_cls_list)
        size_res_list_frames.append(size_res_list)
        rot_angle_list_frames.append(rot_angle_list)
        score_list_frames.append(score_list)
        scores_mask_mean_frames.append(scores_mask_mean_list)
        scores_heading_frames.append(scores_heading_list)
        scores_size_frames.append(scores_size_list)
        batch_seg_sum_frames.append(batch_seg_sum)
    """
    center_list_frames,heading_cls_list_frames,\
    heading_res_list_frames,size_cls_list_frames,\
    size_res_list_frames,rot_angle_list_frames,\
    score_list_frames = NMS(result_dir, TEST_DATASET.id_list, center_list_frames,
                          heading_cls_list_frames, heading_res_list_frames,size_cls_list_frames,
                          size_res_list_frames,rot_angle_list_frames, score_list_frames,scores_mask_mean_frames,batch_seg_sum_frames)
    """
    # evaluation
    print("Segmentation accuracy: %f" % \
        (correct_cnt / float(len(TEST_DATASET.batch_size)*NUM_POINT)))
    # evaluate 3d box accuracy
    #box3d_accuracy(center_list_frames,heading_cls_list_frames,heading_res_list_frames,size_cls_list_frames,size_res_list_frames)

    # Write detection results for KITTI evaluation

    write_detection_results_batch_3dbat(result_dir, TEST_DATASET.idx_batch,
        center_list_frames,heading_cls_list_frames, heading_res_list_frames,
        size_cls_list_frames, size_res_list_frames, rot_angle_list_frames, score_list_frames,scores_mask_mean_frames,scores_heading_frames,scores_size_frames,batch_seg_sum_frames)
    write_detection_results_kitti(OUTPUT_FILE, TEST_DATASET.idx_batch, center_list_frames, \
                                  heading_cls_list_frames, heading_res_list_frames, \
                                  size_cls_list_frames, size_res_list_frames, \
                                  rot_angle_list_frames, score_list_frames,batch_seg_sum_frames)

    def test_batch(output_filename, result_dir=None):
        ''' Test frustum pointnets with GT 2D boxes.
        Write test results to KITTI format label files.
        todo (rqi): support variable number of points.
        '''
        ps_list_frames = []
        seg_list_frames = []
        segp_list_frames = []
        center_list_frames = []
        heading_cls_list_frames = []
        heading_res_list_frames = []
        size_cls_list_frames = []
        size_res_list_frames = []
        rot_angle_list_frames = []
        score_list_frames = []
        scores_mask_mean_frames = []
        scores_heading_frames = []
        scores_size_frames = []
        batch_seg_sum_frames = []
        start_idx = 0
        for i in range(len(TEST_DATASET.idx_batch)):
            ps_list = []
            seg_list = []
            segp_list = []
            center_list = []
            heading_cls_list = []
            heading_res_list = []
            size_cls_list = []
            size_res_list = []
            rot_angle_list = []
            score_list = []
            scores_mask_mean_list = []
            scores_heading_list = []
            scores_size_list = []
            batch_seg_sum_list = []
            test_idxs = np.arange(0, len(TEST_DATASET))
            batch_size = TEST_DATASET.batch_size[i]
            # num_batches = len(TEST_DATASET)/batch_size

            sess, ops = get_session_and_ops(batch_size=batch_size, num_point=NUM_POINT)
            correct_cnt = 0
            print('batch idx: %d' % (i))
            print('batch start', start_idx)

            # start_idx = 0
            end_idx = start_idx + batch_size
            print('end_idx', end_idx)
            batch_data, batch_rot_angle, batch_label, batch_one_hot_vec, = \
                get_batch_test(TEST_DATASET, test_idxs, start_idx, end_idx,
                               NUM_POINT, NUM_CHANNEL)

            batch_output, batch_center_pred, \
            batch_hclass_pred, batch_hres_pred, \
            batch_sclass_pred, batch_sres_pred, batch_scores, batch_scores_mask_mean, batch_scores_heading, batch_scores_size, batch_seg_sum = \
                inference(sess, ops, batch_data,
                          batch_one_hot_vec, batch_size=batch_size)

            correct_cnt += np.sum(batch_output == batch_label)
            print("batch correct_cnt: ", correct_cnt)
            start_idx = start_idx + batch_size

            for i in range(batch_output.shape[0]):
                ps_list.append(batch_data[i, ...])
                segp_list.append(batch_output[i, ...])
                center_list.append(batch_center_pred[i, :])
                heading_cls_list.append(batch_hclass_pred[i])
                heading_res_list.append(batch_hres_pred[i])
                size_cls_list.append(batch_sclass_pred[i])
                size_res_list.append(batch_sres_pred[i, :])
                rot_angle_list.append(batch_rot_angle[i])
                score_list.append(batch_scores[i])
                scores_mask_mean_list.append(batch_scores_mask_mean[i])
                scores_heading_list.append(batch_scores_heading[i])
                scores_size_list.append(batch_scores_size[i])
                batch_seg_sum_list.append(batch_seg_sum[i])
            ps_list_frames.append(ps_list)
            seg_list_frames.append(seg_list)
            segp_list_frames.append(segp_list)
            center_list_frames.append(center_list)
            heading_cls_list_frames.append(heading_cls_list)
            heading_res_list_frames.append(heading_res_list)
            size_cls_list_frames.append(size_cls_list)
            size_res_list_frames.append(size_res_list)
            rot_angle_list_frames.append(rot_angle_list)
            score_list_frames.append(score_list)
            scores_mask_mean_frames.append(scores_mask_mean_list)
            scores_heading_frames.append(scores_heading_list)
            scores_size_frames.append(scores_size_list)
            batch_seg_sum_frames.append(batch_seg_sum)
        """
        center_list_frames,heading_cls_list_frames,\
        heading_res_list_frames,size_cls_list_frames,\
        size_res_list_frames,rot_angle_list_frames,\
        score_list_frames = NMS(result_dir, TEST_DATASET.id_list, center_list_frames,
                              heading_cls_list_frames, heading_res_list_frames,size_cls_list_frames,
                              size_res_list_frames,rot_angle_list_frames, score_list_frames,scores_mask_mean_frames,batch_seg_sum_frames)
        """
        # evaluation
        print("Segmentation accuracy: %f" % \
              (correct_cnt / float(len(TEST_DATASET.batch_size) * NUM_POINT)))
        # evaluate 3d box accuracy
        # box3d_accuracy(center_list_frames,heading_cls_list_frames,heading_res_list_frames,size_cls_list_frames,size_res_list_frames)

        # Write detection results for KITTI evaluation

        # write_detection_results_batch_3dbat(result_dir, TEST_DATASET.idx_batch,
        #    center_list_frames,heading_cls_list_frames, heading_res_list_frames,
        #    size_cls_list_frames, size_res_list_frames, rot_angle_list_frames, score_list_frames,scores_mask_mean_frames,scores_heading_frames,scores_size_frames)
        write_detection_results_kitti(OUTPUT_FILE, TEST_DATASET.idx_batch, center_list_frames, \
                                      heading_cls_list_frames, heading_res_list_frames, \
                                      size_cls_list_frames, size_res_list_frames, \
                                      rot_angle_list_frames, score_list_frames)

def box3d_accuracy(center_list_frames,scores_heading,heading_res_list_frames,scores_size_frames,size_residuals):
    # get lables (size, residuals) for this frame:
    # get batch frame
    for j in range(len(TEST_DATASET.idx_batch[:5])):
        print(TEST_DATASET.idx_batch[j])
        gt_obj_list = TEST_DATASET.dataset_kitti.filtrate_objects(TEST_DATASET.dataset_kitti.get_label(TEST_DATASET.idx_batch[j]))
        gt_boxes3d = provider.kitti_utils.objs_to_boxes3d(gt_obj_list)
        corners3d_label = provider.kitti_utils.boxes3d_to_corners3d(gt_boxes3d, transform=True)


        iou2d, iou3d = provider.compute_box3d_iou_batch_test(np.array(center_list_frames[j]),np.array(scores_heading[j]), np.array(heading_res_list_frames[j]), np.array(scores_size_frames[j]), np.array(size_residuals[j]),
            corners3d_label)
        print(iou2d,iou3d)
        """
        provider.compute_box3d_iou( \
            center_list_frames, \
            end_points['heading_scores'], end_points['heading_residuals'], \
            end_points['size_scores'], end_points['size_residuals'], \
            centers_pl, \
            heading_class_label_pl, heading_residual_label_pl, \
            size_class_label_pl, size_residual_label_pl]) \
        [tf.float32, tf.float32]
        """
def inference(sess, ops, pc, one_hot_vec, batch_size):
    ''' Run inference for frustum pointnets in batch mode '''
    assert pc.shape[0]%batch_size == 0
    num_batches = pc.shape[0]/batch_size
    logits = np.zeros((pc.shape[0], pc.shape[1], NUM_CLASSES))
    centers = np.zeros((pc.shape[0], 3))
    heading_logits = np.zeros((pc.shape[0], NUM_HEADING_BIN))
    heading_residuals = np.zeros((pc.shape[0], NUM_HEADING_BIN))
    size_logits = np.zeros((pc.shape[0], NUM_SIZE_CLUSTER))
    size_residuals = np.zeros((pc.shape[0], NUM_SIZE_CLUSTER, 3))
    scores = np.zeros((pc.shape[0],)) # 3D box score
    scores_mask_mean =  np.zeros((pc.shape[0],))
    scores_heading = np.zeros((pc.shape[0],))
    scores_size = np.zeros((pc.shape[0],))
    batch_seg_mask_sum = np.zeros((pc.shape[0],))
    ep = ops['end_points']
    for i in range(num_batches):
        feed_dict = {\
            ops['pointclouds_pl']: pc[i*batch_size:(i+1)*batch_size,...],
            ops['one_hot_vec_pl']: one_hot_vec[i*batch_size:(i+1)*batch_size,:],
            ops['is_training_pl']: False}
        start = time.time()
        batch_logits, batch_centers, \
        batch_heading_scores, batch_heading_residuals, \
        batch_size_scores, batch_size_residuals = \
            sess.run([ops['logits'], ops['center'],
                ep['heading_scores'], ep['heading_residuals'],
                ep['size_scores'], ep['size_residuals']],
                feed_dict=feed_dict)
        end = time.time()
        print("inference time: ", end-start)
        logits[i*batch_size:(i+1)*batch_size,...] = batch_logits
        centers[i*batch_size:(i+1)*batch_size,...] = batch_centers
        heading_logits[i*batch_size:(i+1)*batch_size,...] = batch_heading_scores
        heading_residuals[i*batch_size:(i+1)*batch_size,...] = batch_heading_residuals
        size_logits[i*batch_size:(i+1)*batch_size,...] = batch_size_scores
        size_residuals[i*batch_size:(i+1)*batch_size,...] = batch_size_residuals

        # Compute scores
        batch_seg_prob = softmax(batch_logits)[:,:,1] # BxN
        #print("batch_seg_prob",batch_seg_prob)
        batch_seg_mask = np.argmax(batch_logits, 2) # BxN
        #print("batch_seg_mask:",batch_seg_mask)
        mask_mean_prob = np.sum(batch_seg_prob * batch_seg_mask, 1) # B,
        #print("mask_mean_prob: ",mask_mean_prob)
        #print("sum batch seg mask: ",np.sum(batch_seg_mask,1))
        mask_mean_prob = mask_mean_prob / np.sum(batch_seg_mask,1) # B,
        #print("mask_mean_prob: ", mask_mean_prob)
        heading_prob = np.max(softmax(batch_heading_scores),1) # B
        size_prob = np.max(softmax(batch_size_scores),1) # B,
        batch_scores = np.log(mask_mean_prob) + np.log(heading_prob) + np.log(size_prob)
        #print("mask_mean_prob",mask_mean_prob,  np.log(mask_mean_prob),"heading_prob", heading_prob,np.log(heading_prob), "size_prob",size_prob, np.log(size_prob))
        batch_seg_mask_sum[i*batch_size:(i+1)*batch_size] =  np.sum(batch_seg_mask,1)
        scores[i*batch_size:(i+1)*batch_size] = batch_scores
        scores_mask_mean[i * batch_size:(i + 1) * batch_size] =  mask_mean_prob
        scores_heading[i * batch_size:(i + 1) * batch_size] =  heading_prob
        scores_size[i * batch_size:(i + 1) * batch_size] = size_prob
        #print("scores",scores)
        # Finished computing scores

    heading_cls = np.argmax(heading_logits, 1) # B
    size_cls = np.argmax(size_logits, 1) # B
    heading_res = np.array([heading_residuals[i,heading_cls[i]] \
        for i in range(pc.shape[0])])
    size_res = np.vstack([size_residuals[i,size_cls[i],:] \
        for i in range(pc.shape[0])])

    return np.argmax(logits, 2), centers, heading_cls, heading_res, \
        size_cls, size_res, scores,scores_mask_mean,scores_heading,scores_size,batch_seg_mask_sum
def test(output_filename, result_dir=None):
    ''' Test frustum pointnets with GT 2D boxes.
    Write test results to KITTI format label files.
    todo (rqi): support variable number of points.
    '''
    ps_list_frames = []
    seg_list_frames = []
    segp_list_frames = []
    center_list_frames = []
    heading_cls_list_frames = []
    heading_res_list_frames = []
    size_cls_list_frames = []
    size_res_list_frames = []
    rot_angle_list_frames = []
    score_list_frames = []
    center_label_frames = []
    start_idx = 0
    for i in range(len(TEST_DATASET.batch_size)):
        ps_list = []
        seg_list = []
        segp_list = []
        center_list = []
        heading_cls_list = []
        heading_res_list = []
        size_cls_list = []
        size_res_list = []
        rot_angle_list = []
        score_list = []

        test_idxs = np.arange(0, len(TEST_DATASET))
        batch_size = TEST_DATASET.batch_size[i]
        #num_batches = len(TEST_DATASET)/batch_size

        sess, ops = get_session_and_ops(batch_size=batch_size, num_point=NUM_POINT)
        correct_cnt = 0
        print('batch idx: %d' % (i))
        print('batch start' ,start_idx)

        #start_idx = 0
        end_idx = start_idx + batch_size
        print('end_idx',end_idx)
        batch_data, batch_label, batch_center, \
        batch_hclass, batch_hres, batch_sclass, batch_sres, \
        batch_rot_angle, batch_one_hot_vec = \
            get_batch(TEST_DATASET, test_idxs, start_idx, end_idx,
                NUM_POINT, NUM_CHANNEL)

        batch_output, batch_center_pred, \
            batch_hclass_pred, batch_hres_pred, \
            batch_sclass_pred, batch_sres_pred, batch_scores = \
                inference(sess, ops, batch_data,
                    batch_one_hot_vec, batch_size=batch_size)

        correct_cnt += np.sum(batch_output==batch_label)

        start_idx = start_idx+batch_size
        print(batch_output.shape)
        for i in range(len(batch_output)):
            print(np.count_nonzero(batch_output[i] == 1))
        for i in range(batch_output.shape[0]):
            ps_list.append(batch_data[i,...])
            seg_list.append(batch_label[i,...])
            segp_list.append(batch_output[i,...])
            center_list.append(batch_center_pred[i,:])
            heading_cls_list.append(batch_hclass_pred[i])
            heading_res_list.append(batch_hres_pred[i])
            size_cls_list.append(batch_sclass_pred[i])
            size_res_list.append(batch_sres_pred[i,:])
            rot_angle_list.append(batch_rot_angle[i])
            score_list.append(batch_scores[i])
        ps_list_frames.append(ps_list)
        seg_list_frames.append(seg_list)
        segp_list_frames.append(segp_list)
        center_list_frames.append(center_list)
        heading_cls_list_frames.append(heading_cls_list)
        heading_res_list_frames.append(heading_res_list)
        size_cls_list_frames.append(size_cls_list)
        size_res_list_frames.append(size_res_list)
        rot_angle_list_frames.append(rot_angle_list)
        score_list_frames.append(score_list)


    #print("Segmentation accuracy: %f" % \
    #    (correct_cnt / float(batch_size*NUM_POINT)))
    ''''
    if FLAGS.dump_result:
        with open(output_filename, 'wp') as fp:
            pickle.dump(ps_list, fp)
            pickle.dump(seg_list, fp)
            pickle.dump(segp_list, fp)
            pickle.dump(center_list, fp)
            pickle.dump(heading_cls_list, fp)
            pickle.dump(heading_res_list, fp)
            pickle.dump(size_cls_list, fp)
            pickle.dump(size_res_list, fp)
            pickle.dump(rot_angle_list, fp)
            pickle.dump(score_list, fp)
    '''
    # Write detection results for KITTI evaluation
    NMS(result_dir, TEST_DATASET.id_list, center_list_frames,heading_cls_list_frames, heading_res_list_frames,size_cls_list_frames, size_res_list_frames,rot_angle_list_frames, score_list_frames)
    #write_detection_results_batch_3dbat(result_dir, TEST_DATASET.id_list,
    #    center_list_frames,heading_cls_list_frames, heading_res_list_frames,
    #    size_cls_list_frames, size_res_list_frames, rot_angle_list_frames, score_list_frames)


if __name__=='__main__':

    test_batch(FLAGS.output+'.pickle', FLAGS.output)
