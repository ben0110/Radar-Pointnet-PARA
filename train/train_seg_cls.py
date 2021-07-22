''' Training Frustum PointNets.

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
from datetime import datetime
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
import provider_seg as provider
from train_util import get_batch_seg_cls
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='frustum_pointnets_v1', help='Model name [default: frustum_pointnets_v1]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
parser.add_argument('--max_epoch', type=int, default=201, help='Epoch to run [default: 201]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--no_intensity', action='store_true', help='Only use XYZ for training')
parser.add_argument('--restore_model_path', default=None, help='Restore model path e.g. log/model.ckpt [default: None]')
FLAGS = parser.parse_args()

# Set training configurations
EPOCH_CNT = 0
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
NUM_CHANNEL = 3 if FLAGS.no_intensity else 4 # point feature channel
NUM_CLASSES = 2 # segmentation has two classes

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
datum = datetime.now()
datum = datum.strftime("%d-%m-%Y-%H:%M:%S")
LOG_DIR=os.path.join(LOG_DIR,datum)
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
    os.mkdir(LOG_DIR+"/ckpt")
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp %s %s' % (os.path.join(BASE_DIR, 'train_seg.py'), LOG_DIR))
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

# Load Frustum Datasets. Use default data paths.
TRAIN_DATASET = provider.RadarDataset_seg_per_ROI('pc_radar_2','KITTI',npoints=NUM_POINT, split='train',
    rotate_to_center=False, random_flip=False, random_shift=False, one_hot=True,all_batches = True,translate_radar_center=False,store_data=True,proposals_3=False ,no_color=True)
TEST_DATASET = provider.RadarDataset_seg_per_ROI('pc_radar_2','KITTI',npoints=NUM_POINT, split='val',rotate_to_center=False, one_hot=True,all_batches = True, translate_radar_center=False, store_data=True, proposals_3 =False ,no_color=True)
TEST_1_DATASET =  provider.RadarDataset_seg_per_ROI('pc_radar_2','KITTI_2',npoints=NUM_POINT, split='test',rotate_to_center=False, one_hot=True,all_batches = True, translate_radar_center=False, store_data=True, proposals_3 =False ,no_color=True)
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    #learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    ''' Main function for training and simple evaluation. '''
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, one_hot_vec_pl, labels_pl,ids_pl,radar_rois_pl,radar_set_pl,gt_corners_pl  = \
                MODEL.placeholder_inputs_seg_cls(BATCH_SIZE, NUM_POINT)

            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to increment the 'batch' parameter
            # for you every time it trains.
            batch = tf.get_variable('batch', [],
                initializer=tf.constant_initializer(0), trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and losses 
            end_points = MODEL.get_model(pointclouds_pl,ids_pl,radar_rois_pl,radar_set_pl, one_hot_vec_pl,
                is_training_pl, bn_decay=bn_decay)
            loss=\
                MODEL.get_loss_seg_cls(labels_pl,gt_corners_pl, end_points)
            tf.summary.scalar('loss', loss)

            losses = tf.get_collection('losses')
            total_loss = tf.add_n(losses, name='total_loss')
            tf.summary.scalar('total_loss', total_loss)

            # Write summaries of bounding box IoU and segmentation accuracies
            """iou2ds, iou3ds,box_det_nbr = tf.py_func(provider.compute_box3d_iou_batch, [end_points['mask_logits'],\
                end_points['center'], \
                end_points['heading_scores'], end_points['heading_residuals'], \
                end_points['size_scores'], end_points['size_residuals'], \
                centers_pl, \
                heading_class_label_pl, heading_residual_label_pl, \
                size_class_label_pl, size_residual_label_pl], \
                [tf.float32, tf.float32,tf.float32])
            end_points['iou2ds'] = iou2ds 
            end_points['iou3ds'] = iou3ds
            end_points['box_pred_nbr'] = box_det_nbr

            tf.summary.scalar('iou_2d', tf.reduce_mean(iou2ds))
            tf.summary.scalar('iou_3d', tf.reduce_mean(iou3ds))"""

            correct = tf.equal(tf.argmax(end_points['mask_logits'], 2),
                tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / \
                float(BATCH_SIZE*NUM_POINT)
            tf.summary.scalar('segmentation accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate,
                    momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
        
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        if FLAGS.restore_model_path is None:
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            saver.restore(sess, FLAGS.restore_model_path)

        ops = {'pointclouds_pl': pointclouds_pl,
               'ids':ids_pl,
               'radar_rois':radar_rois_pl,
               'radar_set':radar_set_pl,
               'one_hot_vec_pl': one_hot_vec_pl,
               'labels_pl': labels_pl,
               'gt_corners_pl':gt_corners_pl,
               'is_training_pl': is_training_pl,
               'logits': end_points['mask_logits'],
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points}

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
             
            ps_list,segp_list_train,batch_radar_mask_set,radar_rois_param_set,ids_train,ab_corners_train,ab_cls_train,ab_ids_train=train_one_epoch(sess, ops, train_writer)

            ps_list,segp_list_val,batch_radar_mask_set,radar_rois_param_set,ids_val,ab_corners_eval,ab_cls_eval,ab_ids_eval=eval_one_epoch(sess, ops, test_writer)
            ps_list, segp_list_test, batch_radar_mask_set, radar_rois_param_set, ids_test,ab_corners_test,ab_cls_test,ab_ids_test = test_one_epoch(sess, ops,
                                                                                                         test_writer)


            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR,"ckpt" ,"model_"+str(epoch)+".ckpt"))
                log_string("Model saved in file: %s" % save_path)
                save_data("train", segp_list_train, ids_train,ab_corners_train,ab_cls_train,ab_ids_train)
                save_data("val",  segp_list_val,ids_val,ab_corners_eval,ab_cls_eval,ab_ids_eval )
                save_data("test",segp_list_test,ids_test,ab_corners_test,ab_cls_test,ab_ids_test )


def save_data(src, segp_list, ids,ab_corners_list,ab_cls_list,ab_ids_list):
    output_filename = "/root/frustum-pointnets_RSC_RADAR_fil_PC_batch_para/dataset/RSC/" + src +"_seg_per_ROI_cls_iter_method.pickle"
    with open(output_filename, 'wb') as fp:
        pickle.dump(ids,fp)
        pickle.dump(segp_list,fp)
        pickle.dump(ab_corners_list,fp)
        pickle.dump(ab_cls_list, fp)
        pickle.dump(ab_ids_list, fp)
    fp.close()
def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape) - 1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape) - 1, keepdims=True)
    return probs

def train_one_epoch(sess, ops, train_writer):
    ''' Training for one epoch on the frustum dataset.
    ops is dict mapping from string to tf ops
    '''
    is_training = False
    log_string(str(datetime.now()))
    
    # Shuffle train samples
    train_idxs = np.arange(0, len(TRAIN_DATASET))
    np.random.shuffle(train_idxs)
    num_batches = len(TRAIN_DATASET)/BATCH_SIZE

    # To collect statistics
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    iou2ds_sum = 0
    iou3ds_sum = 0
    iou3d_correct_cnt = 0
    box_pred_nbr_sum = 0
    # Training with batches

    total_ab_correct=0
    total_ab_seen=0

    ps_list = []
    seg_list = []
    segp_list = []
    score_list = []
    batch_radar_mask_set = []
    radar_rois_param_set = []
    logits_list = []
    ids_list=[]
    pc_seg_list=[]

    ab_corners_list =[]
    ab_cls_list=[]
    ab_ids_list=[]

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        batch_data, batch_label,gt_corners_batch, batch_one_hot_vec,batch_radar_mask_list,radar_rois_param,id_list = \
            get_batch_seg_cls(TRAIN_DATASET, train_idxs, start_idx, end_idx,
                NUM_POINT, NUM_CHANNEL)
        batch_label_nbr=[]
        for i in range(len(batch_label)):
            batch_label_nbr.append(np.count_nonzero(batch_label[i]==1))

        print("batches_label:", batch_label_nbr)
        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['one_hot_vec_pl']: batch_one_hot_vec,
                     ops['ids']: id_list,
                     ops['radar_rois']: radar_rois_param,
                     ops['radar_set']:batch_radar_mask_list,
                     ops['labels_pl']: batch_label,
                     ops['gt_corners_pl']:gt_corners_batch,
                     ops['is_training_pl']: is_training}

        summary, step, _, loss_val, logits_val, Ab_cls,score,ab_corners ,ab_ids= \
            sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'],
                      ops['logits'], ops['end_points']['ab_cls'], ops['end_points']['score_label'],ops['end_points']['ab_corners'],ops['end_points']['ab_ids']],
                     feed_dict=feed_dict)

        train_writer.add_summary(summary, step)
        print("Ab_cls",Ab_cls)
        print("score",score)
        preds_val = np.argmax(logits_val, 2)
        correct = np.sum(preds_val == batch_label)
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += loss_val
        print("preds_val", np.sum(preds_val, 1))
        for l in range(NUM_CLASSES):
            total_seen_class[l] += np.sum(batch_label==l)
            total_correct_class[l] += (np.sum((preds_val==l) & (batch_label==l)))
        for i in range(BATCH_SIZE):
            segp = preds_val[i,:]
            segl = batch_label[i,:]
            part_ious = [0.0 for _ in range(NUM_CLASSES)]
            for l in range(NUM_CLASSES):
                if (np.sum(segl==l) == 0) and (np.sum(segp==l) == 0):
                    part_ious[l] = 1.0 # class not present
                else:
                    part_ious[l] = np.sum((segl==l) & (segp==l)) / \
                        float(np.sum((segl==l) | (segp==l)))
        ab_cls =np.argmax(Ab_cls, 1)
        ab_correct = np.sum(ab_cls==score)
        total_ab_correct += ab_correct
        total_ab_seen += len(ab_cls)
        for i in range(BATCH_SIZE):
            ps_list.append(batch_data[i, ...])
            seg_list.append(batch_label[i, ...])

            segp_list.append(preds_val[i, ...])
            seg_idx = np.argwhere(preds_val[i] == 1)
            pc = batch_data[i, ...]
            pc_seg = pc[seg_idx.reshape(-1)]
            pc_seg_list.append(pc_seg)
            batch_radar_mask_set.append(batch_radar_mask_list[i])
            radar_rois_param_set.append(radar_rois_param[i])
            logits_list.append(logits_val[i])
            ids_list.append(id_list[i])

        for i in range(len(ab_corners)):
            ab_corners_list.append(ab_corners[i])
            ab_cls_list.append(Ab_cls[i])
            ab_ids_list.append(ab_ids[i])



    log_string(' -- %03d / %03d --' % (batch_idx+1, num_batches))
    log_string('mean loss: %f' % (loss_sum / 10))
    log_string('segmentation accuracy: %f' % \
                (total_correct / float(total_seen)))
    log_string('eval segmentation avg class acc: %f' % \
               (np.mean(np.array(total_correct_class) / \
                        np.array(total_seen_class, dtype=np.float))))
    log_string('rpn accuracy: %f' % (total_ab_correct/float(total_ab_seen)))
                #log_string('box IoU (ground/3D): %f / %f' % \
                #    (iou2ds_sum / max(float(box_pred_nbr_sum),1.0), iou3ds_sum / max(float(box_pred_nbr_sum),1.0)))
                #log_string('box estimation accuracy (IoU=0.5): %f' % \
                #    (float(iou3d_correct_cnt)/max(float(box_pred_nbr_sum),1.0)))
                #total_correct = 0
                #total_seen = 0
                #loss_sum = 0
                #iou2ds_sum = 0
                #iou3ds_sum = 0
                #iou3d_correct_cnt = 0
    return ps_list,pc_seg_list,batch_radar_mask_set,radar_rois_param_set,ids_list,ab_corners_list,ab_cls_list,ab_ids_list
        
def eval_one_epoch(sess, ops, test_writer):
    ''' Simple evaluation for one epoch on the frustum dataset.
    ops is dict mapping from string to tf ops """
    '''
    global EPOCH_CNT
    is_training = False
    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))
    test_idxs = np.arange(0, len(TEST_DATASET))
    num_batches = len(TEST_DATASET)/BATCH_SIZE

    # To collect statistics
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    iou2ds_sum = 0
    iou3ds_sum = 0
    iou3d_correct_cnt = 0
    box_pred_nbr_sum = 0
    ps_list = []
    seg_list = []
    segp_list = []
    score_list = []
    batch_radar_mask_set = []
    radar_rois_param_set = []
    logits_list = []
    ids_list=[]
    pc_seg_list=[]


    total_ab_correct=0
    total_ab_seen=0

    ab_corners_list = []
    ab_cls_list = []
    ab_ids_list = []
    # Simple evaluation with batches 
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        batch_data, batch_label, gt_corners_batch, batch_one_hot_vec, batch_radar_mask_list, radar_rois_param, id_list = \
            get_batch_seg_cls(TEST_DATASET, test_idxs, start_idx, end_idx,
                              NUM_POINT, NUM_CHANNEL)

        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['one_hot_vec_pl']: batch_one_hot_vec,
                     ops['radar_rois']: radar_rois_param,
                     ops['radar_set']: batch_radar_mask_list,
                     ops['labels_pl']: batch_label,
                     ops['ids']: id_list,
                     ops['gt_corners_pl']: gt_corners_batch,
                     ops['is_training_pl']: is_training}

        #summary, step, loss_val, logits_val, iou2ds, iou3ds,box_pred_nbr = \
        #    sess.run([ops['merged'], ops['step'],
        #        ops['loss'], ops['logits'],
        #        ops['end_points']['iou2ds'], ops['end_points']['iou3ds'],ops['end_points']['box_pred_nbr']],
        #        feed_dict=feed_dict)
        summary, step, _, loss_val, logits_val, Ab_cls, score, ab_corners, ab_ids = \
            sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'],
                      ops['logits'], ops['end_points']['ab_cls'], ops['end_points']['score_label'],
                      ops['end_points']['ab_corners'], ops['end_points']['ab_ids']],
                     feed_dict=feed_dict)

        test_writer.add_summary(summary, step)

        preds_val = np.argmax(logits_val, 2)
        correct = np.sum(preds_val == batch_label)
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += loss_val
        for l in range(NUM_CLASSES):
            total_seen_class[l] += np.sum(batch_label==l)
            total_correct_class[l] += (np.sum((preds_val==l) & (batch_label==l)))

        for i in range(BATCH_SIZE):
            segp = preds_val[i,:]
            segl = batch_label[i,:] 
            part_ious = [0.0 for _ in range(NUM_CLASSES)]
            for l in range(NUM_CLASSES):
                if (np.sum(segl==l) == 0) and (np.sum(segp==l) == 0): 
                    part_ious[l] = 1.0 # class not present
                else:
                    part_ious[l] = np.sum((segl==l) & (segp==l)) / \
                        float(np.sum((segl==l) | (segp==l)))

        ab_cls = np.argmax(Ab_cls, 1)
        ab_correct = np.sum(ab_cls==score)
        total_ab_correct += ab_correct
        total_ab_seen += len(ab_cls)

        for i in range(BATCH_SIZE):
            ps_list.append(batch_data[i, ...])
            seg_list.append(batch_label[i, ...])
            segp_list.append(preds_val[i, ...])
            seg_idx = np.argwhere(preds_val[i] == 1)
            pc= batch_data[i,...]
            pc_seg = pc[seg_idx.reshape(-1)]
            pc_seg_list.append(pc_seg)
            batch_radar_mask_set.append(batch_radar_mask_list[i])
            radar_rois_param_set.append(radar_rois_param[i])
            ids_list.append(id_list[i])

        for i in range(len(ab_corners)):
            ab_corners_list.append(ab_corners[i])
            ab_cls_list.append(Ab_cls[i])
            ab_ids_list.append(ab_ids[i])
    log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('eval segmentation accuracy: %f'% \
        (total_correct / float(total_seen)))
    log_string('eval segmentation avg class acc: %f' % \
        (np.mean(np.array(total_correct_class) / \
            np.array(total_seen_class,dtype=np.float))))
    log_string('rpn accuracy: %f' % (total_ab_correct/float(total_ab_seen)))

    #log_string('eval box IoU (ground/3D): %f / %f' % \
    #    (iou2ds_sum / max(float(box_pred_nbr_sum),1.0), iou3ds_sum / \
    #        max(float(box_pred_nbr_sum),1.0)))
    #log_string('eval box estimation accuracy (IoU=0.5): %f' % \
    #    (float(iou3d_correct_cnt)/max(float(box_pred_nbr_sum),1.0)))
         


    return ps_list,pc_seg_list,batch_radar_mask_set,radar_rois_param_set,ids_list,ab_corners_list,ab_cls_list,ab_ids_list


def test_one_epoch(sess, ops, test_writer):
    ''' Simple evaluation for one epoch on the frustum dataset.
    ops is dict mapping from string to tf ops """
    '''
    global EPOCH_CNT
    is_training = False
    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d TEST ----' % (EPOCH_CNT))
    test_idxs = np.arange(0, len(TEST_1_DATASET))
    num_batches = len(TEST_1_DATASET) / BATCH_SIZE

    # To collect statistics
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    iou2ds_sum = 0
    iou3ds_sum = 0
    iou3d_correct_cnt = 0
    box_pred_nbr_sum = 0
    ps_list = []
    seg_list = []
    segp_list = []
    score_list = []
    batch_radar_mask_set = []
    radar_rois_param_set = []
    logits_list = []
    ids_list = []
    pc_seg_list = []


    total_ab_correct=0
    total_ab_seen=0

    ab_corners_list = []
    ab_cls_list = []
    ab_ids_list = []
    # Simple evaluation with batches
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE

        batch_data, batch_label, gt_corners_batch, batch_one_hot_vec, batch_radar_mask_list, radar_rois_param, id_list = \
            get_batch_seg_cls(TEST_1_DATASET, test_idxs, start_idx, end_idx,
                              NUM_POINT, NUM_CHANNEL)

        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['ids']: id_list,
                     ops['one_hot_vec_pl']: batch_one_hot_vec,
                     ops['radar_rois']: radar_rois_param,
                     ops['radar_set']: batch_radar_mask_list,
                     ops['labels_pl']: batch_label,
                     ops['gt_corners_pl']: gt_corners_batch,
                     ops['is_training_pl']: is_training}

        # summary, step, loss_val, logits_val, iou2ds, iou3ds,box_pred_nbr = \
        #    sess.run([ops['merged'], ops['step'],
        #        ops['loss'], ops['logits'],
        #        ops['end_points']['iou2ds'], ops['end_points']['iou3ds'],ops['end_points']['box_pred_nbr']],
        #        feed_dict=feed_dict)
        summary, step, _, loss_val, logits_val, Ab_cls, score, ab_corners, ab_ids = \
            sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'],
                      ops['logits'], ops['end_points']['ab_cls'], ops['end_points']['score_label'],
                      ops['end_points']['ab_corners'], ops['end_points']['ab_ids']],
                     feed_dict=feed_dict)

        test_writer.add_summary(summary, step)

        preds_val = np.argmax(logits_val, 2)
        correct = np.sum(preds_val == batch_label)
        total_correct += correct
        total_seen += (BATCH_SIZE * NUM_POINT)
        loss_sum += loss_val
        for l in range(NUM_CLASSES):
            total_seen_class[l] += np.sum(batch_label == l)
            total_correct_class[l] += (np.sum((preds_val == l) & (batch_label == l)))
        print(Ab_cls)

        for i in range(BATCH_SIZE):
            segp = preds_val[i, :]
            segl = batch_label[i, :]
            part_ious = [0.0 for _ in range(NUM_CLASSES)]
            for l in range(NUM_CLASSES):
                if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):
                    part_ious[l] = 1.0  # class not present
                else:
                    part_ious[l] = np.sum((segl == l) & (segp == l)) / \
                                   float(np.sum((segl == l) | (segp == l)))

        ab_cls = np.argmax(Ab_cls, 1)
        ab_correct = np.sum(ab_cls==score)
        total_ab_correct += ab_correct
        total_ab_seen += len(ab_cls)

        for i in range(BATCH_SIZE):
            ps_list.append(batch_data[i, ...])
            seg_list.append(batch_label[i, ...])
            segp_list.append(preds_val[i, ...])
            seg_idx = np.argwhere(preds_val[i] == 1)
            pc = batch_data[i, ...]
            pc_seg = pc[seg_idx.reshape(-1)]
            pc_seg_list.append(pc_seg)
            batch_radar_mask_set.append(batch_radar_mask_list[i])
            radar_rois_param_set.append(radar_rois_param[i])
            ids_list.append(id_list[i])

        for i in range(len(ab_corners)):
            ab_corners_list.append(ab_corners[i])
            ab_cls_list.append(Ab_cls[i])
            ab_ids_list.append(ab_ids[i])

    log_string('test mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('test segmentation accuracy: %f' % \
               (total_correct / float(total_seen)))
    log_string('test segmentation avg class acc: %f' % \
               (np.mean(np.array(total_correct_class) / \
                        np.array(total_seen_class, dtype=np.float))))
    log_string('rpn accuracy: %f' % (total_ab_correct/float(total_ab_seen)))
    # log_string('eval box IoU (ground/3D): %f / %f' % \
    #    (iou2ds_sum / max(float(box_pred_nbr_sum),1.0), iou3ds_sum / \
    #        max(float(box_pred_nbr_sum),1.0)))
    # log_string('eval box estimation accuracy (IoU=0.5): %f' % \
    #    (float(iou3d_correct_cnt)/max(float(box_pred_nbr_sum),1.0)))
    EPOCH_CNT += 1

    return ps_list, pc_seg_list, batch_radar_mask_set, radar_rois_param_set, ids_list,ab_corners_list,ab_cls_list,ab_ids_list

if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
