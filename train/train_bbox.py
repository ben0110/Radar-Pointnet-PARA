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
from train_util import get_batch_bbox

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
NUM_CHANNEL = 3 #if FLAGS.no_intensity else 4 # point feature channel
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
OUTPUT_FILE = os.path.join(LOG_DIR,
                           'results')
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp %s %s' % (os.path.join(BASE_DIR, 'train_bbox.py'), LOG_DIR))
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

#pathsplit = FLAGS.restore_model_path.split('/')
#OUTPUT_FILE = os.path.join('/', pathsplit[1], pathsplit[2], pathsplit[3], pathsplit[4], pathsplit[5],
#                           'results_train_eval_190/')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

# Load Frustum Datasets. Use default data paths.
TRAIN_DATASET = provider.RadarDataset_bbox('pc_radar_2','KITTI',npoints=NUM_POINT, split='train',
    rotate_to_center=False, random_flip=False, random_shift=False, one_hot=True,all_batches = True,translate_radar_center=False,store_data=True,proposals_3=True ,no_color=True)
EVAL_DATASET = provider.RadarDataset_bbox('pc_radar_2','KITTI',npoints=NUM_POINT, split='val',rotate_to_center=False, one_hot=True,all_batches = True, translate_radar_center=False, store_data=True, proposals_3 =True ,no_color=True)
TEST_DATASET = provider.RadarDataset_bbox('pc_radar_2','KITTI_2',npoints=NUM_POINT, split='test',rotate_to_center=False, one_hot=True,all_batches = True, translate_radar_center=False, store_data=True, proposals_3 =True ,no_color=True)

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
            pointclouds_pl,pointcloud_center_pl, one_hot_vec_pl,  centers_pl, \
            heading_class_label_pl, heading_residual_label_pl, \
            size_class_label_pl, size_residual_label_pl = \
                MODEL.placeholder_inputs_bbox(BATCH_SIZE, NUM_POINT)

            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to increment the 'batch' parameter
            # for you every time it trains.
            batch = tf.get_variable('batch', [],
                initializer=tf.constant_initializer(0), trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and losses 
            end_points = MODEL.get_model(pointclouds_pl,pointcloud_center_pl, one_hot_vec_pl,
                is_training_pl, bn_decay=bn_decay)
            loss=\
                MODEL.get_loss_bbox(centers_pl,
                heading_class_label_pl, heading_residual_label_pl,
                size_class_label_pl, size_residual_label_pl, end_points)
            tf.summary.scalar('loss', loss)

            losses = tf.get_collection('losses')
            total_loss = tf.add_n(losses, name='total_loss')
            tf.summary.scalar('total_loss', total_loss)

            # Write summaries of bounding box IoU and segmentation accuracies
            iou2ds, iou3ds,box_det_nbr = tf.py_func(provider.compute_box3d_iou_bbox, [end_points['center'], \
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
            tf.summary.scalar('iou_3d', tf.reduce_mean(iou3ds))

            #correct = tf.equal(tf.argmax(end_points['mask_logits'], 2),
            #    tf.to_int64(labels_pl))
            #accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / \
            #    float(BATCH_SIZE*NUM_POINT)
            #tf.summary.scalar('segmentation accuracy', accuracy)

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
        eval_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        if FLAGS.restore_model_path is None:
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            saver.restore(sess, FLAGS.restore_model_path)

        ops = {'pointclouds_pl': pointclouds_pl,
               'pointcloud_center_pl':pointcloud_center_pl,
               'one_hot_vec_pl': one_hot_vec_pl,
               'centers_pl': centers_pl,
               'heading_class_label_pl': heading_class_label_pl,
               'heading_residual_label_pl': heading_residual_label_pl,
               'size_class_label_pl': size_class_label_pl,
               'size_residual_label_pl': size_residual_label_pl,
               'is_training_pl': is_training_pl,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points}

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
             
            train_one_epoch(sess, ops, train_writer)
            eval_one_epoch(sess, ops, eval_writer,EVAL_DATASET,"KITTI","val")
            eval_one_epoch(sess,ops,test_writer,TEST_DATASET,"KITTI_2","test")
            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR,"ckpt" ,"model_"+str(epoch)+".ckpt"))
                log_string("Model saved in file: %s" % save_path)



def train_one_epoch(sess, ops, train_writer):
    ''' Training for one epoch on the frustum dataset.
    ops is dict mapping from string to tf ops
    '''
    is_training = True
    log_string(str(datetime.now()))
    
    # Shuffle train samples
    train_idxs = np.arange(0, len(TRAIN_DATASET))
    np.random.shuffle(train_idxs)
    num_batches = len(TRAIN_DATASET)/BATCH_SIZE

    # To collect statistics
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    iou2ds_sum = 0
    iou3ds_sum = 0
    iou3d_correct_cnt = 0
    box_pred_nbr_sum = 0
    # Training with batches
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        batch_data,batch_center_data,batch_one_hot_vec , batch_center, \
        batch_hclass, batch_hres, \
        batch_sclass, batch_sres = \
            get_batch_bbox(TRAIN_DATASET, train_idxs, start_idx, end_idx,
                NUM_POINT, NUM_CHANNEL)
        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['pointcloud_center_pl']:batch_center_data,
                     ops['one_hot_vec_pl']: batch_one_hot_vec,
                     ops['centers_pl']: batch_center,
                     ops['heading_class_label_pl']: batch_hclass,
                     ops['heading_residual_label_pl']: batch_hres,
                     ops['size_class_label_pl']: batch_sclass,
                     ops['size_residual_label_pl']: batch_sres,
                     ops['is_training_pl']: is_training}

        summary, step, _, loss_val,iou2ds, iou3ds,box_pred_nbr = \
            sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'],\
                ops['end_points']['iou2ds'], ops['end_points']['iou3ds'],ops['end_points']['box_pred_nbr']],
                feed_dict=feed_dict)

        train_writer.add_summary(summary, step)
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += loss_val
        iou2ds_sum += np.sum(iou2ds)
        iou3ds_sum += np.sum(iou3ds)
        box_pred_nbr_sum += np.sum(box_pred_nbr)
        print("box_pred_nbr", box_pred_nbr, "/", BATCH_SIZE)
        iou3d_correct_cnt += np.sum(iou3ds>=0.5)

        if (batch_idx+1)%10 == 0:
            log_string(' -- %03d / %03d --' % (batch_idx+1, num_batches))
            log_string('mean loss: %f' % (loss_sum / 10))
            log_string('box IoU (ground/3D): %f / %f' % \
                (iou2ds_sum / max(float(box_pred_nbr_sum),1.0), iou3ds_sum / max(float(box_pred_nbr_sum),1.0)))
            log_string('box estimation accuracy (IoU=0.5): %f' % \
                (float(iou3d_correct_cnt)/max(float(box_pred_nbr_sum),1.0)))
            loss_sum = 0
            iou2ds_sum = 0
            iou3ds_sum = 0
            iou3d_correct_cnt = 0
def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape) - 1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape) - 1, keepdims=True)
    return probs


def eval_one_epoch(sess, ops,test_writer, dataset, database, split):
    ''' Simple evaluation for one epoch on the frustum dataset.
    ops is dict mapping from string to tf ops """
    '''
    global EPOCH_CNT
    is_training = False
    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----' % (EPOCH_CNT))
    test_idxs = np.arange(0, len(dataset))
    num_batches = len(dataset) / BATCH_SIZE

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
    center_list = []
    heading_cls_list = []
    heading_res_list = []
    size_cls_list = []
    size_res_list = []
    rot_angle_list = []
    score_list = []
    batch_mask_list = []
    center_GT = []
    heading_class_GT = []
    heading_res_GT = []
    size_class_GT = []
    size_residual_GT = []

    # Simple evaluation with batches
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE

        batch_data, batch_center_data, batch_one_hot_vec, batch_center, \
        batch_hclass, batch_hres, \
        batch_sclass, batch_sres = \
            get_batch_bbox(dataset, test_idxs, start_idx, end_idx,
                           NUM_POINT, NUM_CHANNEL)

        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['pointcloud_center_pl']: batch_center_data,
                     ops['one_hot_vec_pl']: batch_one_hot_vec,
                     ops['centers_pl']: batch_center,
                     ops['heading_class_label_pl']: batch_hclass,
                     ops['heading_residual_label_pl']: batch_hres,
                     ops['size_class_label_pl']: batch_sclass,
                     ops['size_residual_label_pl']: batch_sres,
                     ops['is_training_pl']: is_training}

        # summary, step, loss_val, logits_val, iou2ds, iou3ds,box_pred_nbr = \
        #    sess.run([ops['merged'], ops['step'],
        #        ops['loss'], ops['logits'],
        #        ops['end_points']['iou2ds'], ops['end_points']['iou3ds'],ops['end_points']['box_pred_nbr']],
        #        feed_dict=feed_dict)
        summary, step, _, loss_val, iou2ds, iou3ds, box_pred_nbr, centers_pred_val, heading_scores, heading_residuals, size_scores, size_residuals = \
            sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'],
                      ops['end_points']['iou2ds'], ops['end_points']['iou3ds'], ops['end_points']['box_pred_nbr'],
                      ops['end_points']['center'], ops['end_points']['heading_scores'], ops['end_points']['heading_residuals'], \
                      ops['end_points']['size_scores'], ops['end_points']['size_residuals']],
                     feed_dict=feed_dict)

        test_writer.add_summary(summary, step)

        total_seen += (BATCH_SIZE * NUM_POINT)
        loss_sum += loss_val
        iou2ds_sum += np.sum(iou2ds)
        iou3ds_sum += np.sum(iou3ds)
        iou3d_correct_cnt += np.sum(iou3ds >= 0.5)
        box_pred_nbr_sum += np.sum(box_pred_nbr)

        heading_cls = np.argmax(heading_scores, 1)  # B
        size_cls = np.argmax(size_scores, 1)  # B
        heading_res = np.array([heading_residuals[i, heading_cls[i]] \
                                for i in range(batch_data.shape[0])])
        size_res = np.vstack([size_residuals[i, size_cls[i], :] \
                              for i in range(batch_data.shape[0])])
        # mask_mean_prob = np.sum(batch_seg_prob * batch_seg_mask, 1)  # B,
        # mask_mean_prob = mask_mean_prob / np.sum(batch_seg_mask, 1)
        heading_prob = np.max(softmax(heading_scores), 1)  # B
        size_prob = np.max(softmax(size_scores), 1)  # B,
        batch_scores = np.log(heading_prob) + np.log(size_prob)

        for i in range(batch_data.shape[0]):
            center_list.append(centers_pred_val[i, :])
            heading_cls_list.append(heading_cls[i])
            heading_res_list.append(heading_res[i])
            size_cls_list.append(size_cls[i])
            size_res_list.append(size_res[i, :])
            # rot_angle_list.append(batch_rot_angle[i])
            score_list.append(batch_scores[i])
            center_GT.append(batch_center[i])
            heading_class_GT.append(batch_hclass[i])
            heading_res_GT.append(batch_hres[i])
            size_class_GT.append(batch_sclass[i])
            size_residual_GT.append(batch_sres[i])
            # batch_mask_list.append(batch_mask[i])

    log_string(split+' mean loss: %f' % (loss_sum / float(num_batches)))
    log_string(split+' box IoU (ground/3D): %f / %f' % \
               (iou2ds_sum / max(float(box_pred_nbr_sum), 1.0), iou3ds_sum / \
                max(float(box_pred_nbr_sum), 1.0)))
    log_string(split+' box estimation accuracy (IoU=0.5): %f' % \
               (float(iou3d_correct_cnt) / max(float(box_pred_nbr_sum), 1.0)))

    EPOCH_CNT += 1
    IOU3d, GT_box_list, pred_box_list = compare_box_iou(dataset.id_list,
                                                        size_residual_GT, size_class_GT, heading_res_GT,
                                                        heading_class_GT, center_GT, score_list,
                                                        size_res_list, size_cls_list, heading_res_list,
                                                        heading_cls_list, center_list)

    accuracy,recall=eval_per_frame(dataset.id_list, dataset.indice_box, GT_box_list,
                   pred_box_list, IOU3d, score_list, database, split)

    write_detection_results_test("", dataset.id_list,
                                 center_list,
                                 heading_cls_list, heading_res_list,
                                 size_cls_list, size_res_list, rot_angle_list, segp_list,score_list, split)
    log_string( split + ' accuracy(0.5): %f' % accuracy)
    log_string( split + ' reca(0.5): %f' % recall)


def compare_box_iou(id_list, size_residual_GT, size_class_GT, heading_res_GT, heading_class_GT, center_GT,
                    score_list, size_res_list, size_cls_list, heading_res_list, heading_cls_list, center_list):
    # file1 = open(OUTPUT_FILE+"/results.txt" , "w")
    IoU = []
    GT_box_list = []
    pred_box_list = []
    for i in range(len(size_residual_GT)):
        GT_box = provider.get_3d_box(provider.class2size(size_class_GT[i], size_residual_GT[i]),
                                     provider.class2angle(heading_class_GT[i], heading_res_GT[i], 12), center_GT[i])
        pred_box = provider.get_3d_box(provider.class2size(size_cls_list[i], size_res_list[i]),
                                       provider.class2angle(heading_cls_list[i], heading_res_list[i], 12),
                                       center_list[i])
        GT_box_list.append(GT_box)
        pred_box_list.append(pred_box)
        iou_3d, iou_2d = provider.box3d_iou(pred_box, GT_box)
        IoU.append(iou_3d)
        """file1.write("3D box %f \n" % id_list[i])
        file1.write("iou %f  ,score %f \n "% (float(iou_3d) ,score_list[i]))
        #file1.write("label seg number: %f \n" % np.count_nonzero(seg_list[i] == 1))
        #file1.write("det seg number: %f\n" % np.count_nonzero(segp_list[i] == 1))
        #file1.write("correct per seen: %d" %np.sum(seg_list[i] == segp_list[i] )/len(seg_list[i])[0])
        file1.write("center: %f , %f, %f\n" % (center_list[i][0],center_list[i][1],center_list[i][2]))
        file1.write("center_GT: %f , %f , %f\n" % (center_GT[i][0], center_GT[i][1], center_GT[i][2]))"""
        size_pred = provider.class2size(size_cls_list[i], size_res_list[i])
        # file1.write("size pred: %f , %f , %f\n" % (size_pred[0],size_pred[1],size_pred[2]))
        size_GT = provider.class2size(size_class_GT[i], size_residual_GT[i])
        """file1.write("size GT: %f, %f , %f\n" %(size_GT[0],size_GT[1],size_GT[2]))
        file1.write("rotation pred %f\n" % provider.class2angle(heading_cls_list[i],heading_res_list[i],12))
        file1.write("rotation GT %f\n" % provider.class2angle(heading_class_GT[i], heading_res_GT[i], 12))"""

    # file1.close()
    return IoU, GT_box_list, pred_box_list


def eval_per_frame(id_list, indice_box_list, GT_box_list, pred_box_list, IOU3d, score_list, database, split):
    seg_list_frame = []
    segp_list_frame = []
    IoU_frame = []
    GT_box_frame = []
    pred_box_frame = []
    score_frame = []
    segp_sum_frame = []
    seg_sum_GT_frame = []
    indice_box_frame = []
    id = id_list[0]
    m = 1
    id_list_frame = []

    for i in range(1, len(score_list)):
        if id == id_list[i]:
            m = m + 1
        if id != id_list[i] or i == len(id_list) - 1:
            seg_prov = []
            segp_prov = []
            score_prov = []
            GT_box_prov = []
            pred_box_prov = []
            segp_sum = []
            seg_sum = []
            iou_prov = []
            indice_box_prov = []
            for j in range(i - m, i):
                indice_box_prov.append(indice_box_list[j])
                pred_box_prov.append(pred_box_list[j])
                score_prov.append(score_list[j])
                GT_box_prov.append(GT_box_list[j])
                iou_prov.append(IOU3d[j])
            id_list_frame.append(id)
            seg_list_frame.append(seg_prov)
            segp_list_frame.append(segp_prov)
            score_frame.append(score_prov)
            IoU_frame.append(iou_prov)
            GT_box_frame.append(GT_box_prov)
            pred_box_frame.append(pred_box_prov)
            indice_box_frame.append(indice_box_prov)
            segp_sum_frame.append(segp_sum)
            seg_sum_GT_frame.append(seg_sum)
            m = 1
            id = id_list[i]

    # score_list_frame, \
    # bboxes_frame,score_new_frame,id_new_frame,indices_frame,iou_new_frame = NMS(id_list_frame,pred_box_frame,IoU_frame,segp_sum_frame,score_frame,indice_box_frame)
    # load_GT
    #print("id_list_frame[len(id_list_frame)-1]", id_list_frame[len(id_list_frame) - 1])
    corners_GT_frame, id_list_GT = provider.load_GT_eval(id_list_frame[len(id_list_frame) - 1], database, split)
    #print("****************************************************************")
    #print(len(id_list_frame), len(id_list_GT))
    #print(id_list_frame[len(id_list_frame) - 1], id_list_GT[len(id_list_GT) - 1])
    accuracy,recall= precision_recall(id_list_frame, pred_box_frame, corners_GT_frame, score_frame, IoU_frame, indice_box_frame,
                     id_list_GT)
    return accuracy,recall

def NMS_unique(iou, corners_unique, scores_unique):
    ind_sort = np.argsort([x for x in scores_unique])
    bboxes = []
    score_list = []
    id_list = []
    indice = []
    iou_prov = []
    for i in range(len(corners_unique)):
        bbox = corners_unique[ind_sort[i]]
        flag = 1
        for k in range(i + 1, len(corners_unique)):
            #print("index ", ind_sort[i], scores_unique[ind_sort[i]], "index _comp: ", ind_sort[k],
            #      scores_unique[ind_sort[k]], "IoU: ", provider.box3d_iou(bbox, corners_unique[ind_sort[k]]))
            if provider.box3d_iou(bbox, corners_unique[ind_sort[k]])[1] > 0.2:
                flag = -1
                break
        if flag == 1:
            bboxes.append(bbox)
            # id_list.append(id_list_frame[j][ind_sort[i]])

            iou_prov.append(iou[ind_sort[i]])
            score_list.append(scores_unique[ind_sort[i]])

    for i in range(len(iou_prov) - 1):
        iou_prov[i] = 0.0
    return iou_prov


def precision_recall(id_list_frame, corners_frame, corners_GT_frame, scores, iou_frame, indice_box_frame, id_list_GT):
    IoU = []
    gt_box_num = []
    iou_3d_frame = []
    iou_2d_frame = []
    accuracy_5 = 0.0
    recall_5 = 0.0
    accuracy_4 = 0.0
    recall_4 = 0.0
    accuracy_35 = 0.0
    recall_35 = 0.0
    accuracy_3 = 0.0
    recall_3 = 0.0
    accuracy_25 = 0.0
    recall_25 = 0.0

    corners_GT_orig = corners_GT_frame
    corners_orig = corners_frame
    id_list_frame = np.asarray(id_list_frame)
    for i in range(len(id_list_GT)):
        frame = np.where(id_list_frame == id_list_GT[i])

        #print("*******************id*************: ", id_list_GT[i])
        #print("indices", frame[0])
        #print("nbr of GT boxes", len(corners_GT_frame[i]))
        if (frame[0].size == 0):
            accuracy_frame = 0
            recall_frame = 0
        else:
            frame_id = frame[0][0]
            #print("nbr of predicted boxes: ", len(corners_frame[frame_id]))
            #print("nbr of GT boxes", len(corners_GT_frame[frame_id]))
            #print("indices", indice_box_frame[frame_id])
            #print("iou_frame", iou_frame[frame_id])
            old_iou = []
            indices = []
            scores_ = []
            corners_ = []
            iou = []
            for j in range(len(indice_box_frame[frame_id])):
                old_iou.append(iou_frame[frame_id][j])
                indices.append(indice_box_frame[frame_id][j])
                scores_.append(scores[frame_id][j])
                corners_.append(corners_frame[frame_id][j])

            unique = np.unique(indices)
            #print("old_iou", old_iou)
            for j in range(len(unique)):
                indices_unique = np.argwhere(indices == unique[j])

                #print("indice_unique", indices_unique[0])
                # max_iou=np.max(old_iou[indices_unique[0]])

                abs = [old_iou[x] for x in indices_unique[:, 0]]
                corners_unique = [corners_[x] for x in indices_unique[:, 0]]
                scores_unique = [scores_[x] for x in indices_unique[:, 0]]
                #print("scores_unique", scores_unique)
                #print("iou unique", abs)
                iou_nms = NMS_unique(abs, corners_unique, scores_unique)
                #print("iou nms", iou_nms)
                # max_iou = np.max(abs)
                # iou.append(max_iou)
                if len(iou_nms) > 1:
                    iou_nms.sort()
                    for z in range(1, len(iou_nms)):
                        iou_nms[z] = 0.0
                for c in range(len(iou_nms)):
                    iou.append(iou_nms[c])
            #print("new iou", iou)
            TP_5 = 0
            for m in range(len(iou)):
                if (iou[m] > 0.5):
                    TP_5 += 1.0
            TP_4 = 0
            for m in range(len(iou)):
                if (iou[m] > 0.4):
                    TP_4 += 1.0

            TP_35 = 0
            for m in range(len(iou)):
                if (iou[m] > 0.35):
                    TP_35 += 1.0
            TP_3 = 0
            for m in range(len(iou)):
                if (iou[m] > 0.4):
                    TP_3 += 1.0

            TP_25 = 0
            for m in range(len(iou)):
                if (iou[m] > 0.25):
                    TP_25 += 1.0

            accuracy_frame_5 = TP_5 / float(max(1.0, float(len(iou))))
            recall_frame_5 = TP_5 / float(len(corners_GT_orig[i]))

            accuracy_frame_4 = TP_4 / float(max(1.0, float(len(iou))))
            recall_frame_4 = TP_4 / float(len(corners_GT_orig[i]))

            accuracy_frame_35 = TP_35 / float(max(1.0, float(len(iou))))
            recall_frame_35 = TP_35 / float(len(corners_GT_orig[i]))

            accuracy_frame_3 = TP_3 / float(max(1.0, float(len(iou))))
            recall_frame_3 = TP_3 / float(len(corners_GT_orig[i]))

            accuracy_frame_25 = TP_25 / float(max(1.0, float(len(iou))))
            recall_frame_25 = TP_25 / float(len(corners_GT_orig[i]))

            accuracy_5 += accuracy_frame_5
            recall_5 += recall_frame_5
            accuracy_4 += accuracy_frame_4
            recall_4 += recall_frame_4
            accuracy_35 += accuracy_frame_35
            recall_35 += recall_frame_35
            accuracy_3 += accuracy_frame_3
            recall_3 += recall_frame_3
            accuracy_25 += accuracy_frame_25
            recall_25 += recall_frame_25

        # iou_3d_frame.append(iou_3d_prov)

    print("accuracy_5", accuracy_5 / max(len(corners_frame), 1))
    print("recall_5", recall_5 / max(len(corners_GT_orig), 1))
    print("accuracy_4", accuracy_4 / max(len(corners_frame), 1))
    print("recall_4", recall_4 / max(len(corners_GT_orig), 1))
    print("accuracy_35", accuracy_35 / max(len(corners_frame), 1))
    print("recall_35", recall_35 / max(len(corners_GT_orig), 1))
    print("accuracy_3", accuracy_3 / max(len(corners_frame), 1))
    print("recall_3", recall_3 / max(len(corners_GT_orig), 1))
    print("accuracy_25", accuracy_25 / max(len(corners_frame), 1))
    print("recall_25", recall_25 / max(len(corners_GT_orig), 1))

    print("")
    return accuracy_5,recall_5

def write_detection_results_test(result_dir, id_list, center_list, \
                                 heading_cls_list, heading_res_list, \
                                 size_cls_list, size_res_list, \
                                 rot_angle_list, segp_list,score_list, split):
    ''' Write frustum pointnets results to KITTI format label files. '''
    result_dir = OUTPUT_FILE + split
    if result_dir is None: return
    results = {}  # map from idx to list of strings, each string is a line (without \n)

    for i in range(len(center_list)):
        """if np.count_nonzero(segp_list[i] == 1) < 5:
            continue"""
        idx = id_list[i]

        output_str = "Pedestrian -1 -1 -10 "
        output_str += "0.0 0.0 0.0 0.0 "
        h, w, l, tx, ty, tz, ry = provider.from_prediction_to_label_format(center_list[i],
                                                                           heading_cls_list[i], heading_res_list[i],
                                                                           size_cls_list[i], size_res_list[i], 0.0)
        output_str += "%f %f %f %f %f %f %f %f" % (h, w, l, tx, ty, tz, ry, score_list[i])
        if idx not in results: results[idx] = []
        results[idx].append(output_str)

    # Write TXT files
    if not os.path.exists(result_dir): os.mkdir(result_dir)
    output_dir = os.path.join(result_dir, 'data')

    if not os.path.exists(output_dir): os.mkdir(output_dir)
    for idx in results:
        pred_filename = os.path.join(output_dir, '%06d.txt' % (idx))
        print(pred_filename)
        fout = open(pred_filename, 'w')
        for line in results[idx]:
            fout.write(line + '\n')
        fout.close()
if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
