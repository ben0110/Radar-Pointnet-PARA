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
from train_util import get_batch_seg_all
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
os.system('cp %s %s' % (os.path.join(BASE_DIR, 'train_seg_all.py'), LOG_DIR))
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

# Load Frustum Datasets. Use default data paths.
TRAIN_DATASET = provider.Dataset_seg('pc_radar_2',npoints=NUM_POINT, split='train',
    rotate_to_center=False, random_flip=False, random_shift=False, one_hot=True,all_batches = True,translate_radar_center=False,store_data=True,proposals_3=False ,no_color=True)
TEST_DATASET = provider.Dataset_seg('pc_radar_2',npoints=NUM_POINT, split='val',rotate_to_center=False, one_hot=True,all_batches = True, translate_radar_center=False, store_data=True, proposals_3 =False ,no_color=True)

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
            pointclouds_pl, one_hot_vec_pl, labels_pl  = \
                MODEL.placeholder_inputs_seg(BATCH_SIZE, NUM_POINT)

            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to increment the 'batch' parameter
            # for you every time it trains.
            batch = tf.get_variable('batch', [],
                initializer=tf.constant_initializer(0), trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and losses 
            end_points = MODEL.get_model(pointclouds_pl, one_hot_vec_pl,
                is_training_pl, bn_decay=bn_decay)
            loss=\
                MODEL.get_loss_seg(labels_pl, end_points)
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
               'one_hot_vec_pl': one_hot_vec_pl,
               'labels_pl': labels_pl,
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
             
            ps_list_train,segp_list_train,ids_train=train_one_epoch(sess, ops, train_writer)

            ps_list_val,segp_list_val,ids_val=eval_one_epoch(sess, ops, test_writer)

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR,"ckpt" ,"model_"+str(epoch)+".ckpt"))
                log_string("Model saved in file: %s" % save_path)
            if epoch == 10:
                save_data("train_all", ps_list_train, segp_list_train, ids_train)
                save_data("val_all", ps_list_val, segp_list_val,  ids_val)

def save_data(src, ps_list, segp_list,ids):
    """id_data = []
    pc_roi = []
    rot_angle = []
    for i in range(len(ps_list)):
        id = ids[i]
        for j in range(len(batch_radar_mask_set[i])):
            labels_per_roi = batch_radar_mask_set[i][j] * segp_list[i]
            if (np.count_nonzero(labels_per_roi == 1) > 10):
                pos_indices = np.where(labels_per_roi == 1)[0]
                point_set = ps_list[i][pos_indices, :]
                id_data.append(id)
                pc_roi.append(point_set)
                rot_angle.append(radar_rois_param_set[i][j][6])"""
    output_filename = "/root/frustum-pointnets_RSC_RADAR_fil_PC_batch_para/dataset/RSC/seg_rois_" + src + ".pickle"
    with open(output_filename, 'wb') as fp:
        pickle.dump(ids,fp)
        pickle.dump(segp_list, fp)

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

    ps_list = []
    seg_list = []
    segp_list = []
    score_list = []
    batch_radar_mask_set = []
    radar_rois_param_set = []
    logits_list = []
    ids_list=[]
    pc_seg_list=[]

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        batch_data, batch_label, batch_one_hot_vec,id_list = \
            get_batch_seg_all(TRAIN_DATASET, train_idxs, start_idx, end_idx,
                NUM_POINT, NUM_CHANNEL)
        batch_label_nbr=[]
        for i in range(len(batch_label)):
            batch_label_nbr.append(np.count_nonzero(batch_label[i]==1))

        print("batches_label:", batch_label_nbr)
        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['one_hot_vec_pl']: batch_one_hot_vec,
                     ops['labels_pl']: batch_label,
                     ops['is_training_pl']: is_training}

        summary, step, _, loss_val,logits_val,= \
            sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'],ops['logits']],
                feed_dict=feed_dict)

        train_writer.add_summary(summary, step)

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

        for i in range(BATCH_SIZE):
            ps_list.append(batch_data[i, ...])
            seg_list.append(batch_label[i, ...])
            segp_list.append(preds_val[i, ...])
            logits_list.append(logits_val[i])
            ids_list.append(id_list[i])
            seg_idx = np.argwhere(preds_val[i] == 1)
            pc = batch_data[i, ...]
            pc_seg = pc[seg_idx.reshape(-1)]
            pc_seg_list.append(pc_seg)


    log_string(' -- %03d / %03d --' % (batch_idx+1, num_batches))
    log_string('mean loss: %f' % (loss_sum / 10))
    log_string('segmentation accuracy: %f' % \
                (total_correct / float(total_seen)))
    log_string('eval segmentation avg class acc: %f' % \
               (np.mean(np.array(total_correct_class) / \
                        np.array(total_seen_class, dtype=np.float))))
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
    return ps_list,pc_seg_list,ids_list
        
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
    ids_list=[]
    batch_radar_mask_set = []
    radar_rois_param_set = []
    logits_list = []
    pc_seg_list=[]
    # Simple evaluation with batches 
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        batch_data, batch_label, batch_one_hot_vec,id_list = \
            get_batch_seg_all(TEST_DATASET, test_idxs, start_idx, end_idx,
                          NUM_POINT, NUM_CHANNEL)

        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['one_hot_vec_pl']: batch_one_hot_vec,
                     ops['labels_pl']: batch_label,
                     ops['is_training_pl']: is_training}

        #summary, step, loss_val, logits_val, iou2ds, iou3ds,box_pred_nbr = \
        #    sess.run([ops['merged'], ops['step'],
        #        ops['loss'], ops['logits'],
        #        ops['end_points']['iou2ds'], ops['end_points']['iou3ds'],ops['end_points']['box_pred_nbr']],
        #        feed_dict=feed_dict)
        summary, step, _, loss_val,logits_val, = \
            sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'],
                      ops['logits']],
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
        for i in range(BATCH_SIZE):
            ps_list.append(batch_data[i, ...])
            seg_list.append(batch_label[i, ...])
            segp_list.append(preds_val[i, ...])
            ids_list.append(id_list[i])
            seg_idx = np.argwhere(preds_val[i] == 1)
            pc = batch_data[i, ...]
            pc_seg = pc[seg_idx.reshape(-1)]
            pc_seg_list.append(pc_seg)
    log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('eval segmentation accuracy: %f'% \
        (total_correct / float(total_seen)))
    log_string('eval segmentation avg class acc: %f' % \
        (np.mean(np.array(total_correct_class) / \
            np.array(total_seen_class,dtype=np.float))))
    #log_string('eval box IoU (ground/3D): %f / %f' % \
    #    (iou2ds_sum / max(float(box_pred_nbr_sum),1.0), iou3ds_sum / \
    #        max(float(box_pred_nbr_sum),1.0)))
    #log_string('eval box estimation accuracy (IoU=0.5): %f' % \
    #    (float(iou3d_correct_cnt)/max(float(box_pred_nbr_sum),1.0)))
         
    EPOCH_CNT += 1

    return ps_list,pc_seg_list,ids_list

if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
