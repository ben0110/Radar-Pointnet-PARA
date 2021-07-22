from __future__ import print_function

import os
import sys
import argparse
import importlib
import numpy as np
#import tensorflow as tf
from datetime import datetime
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
import provider_dfd_batch_train_overall as provider
from train_util import get_batch

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

#MODEL = importlib.import_module(FLAGS.model) # import network module
#MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')
#LOG_DIR = FLAGS.log_dir
#if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
#os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
#os.system('cp %s %s' % (os.path.join(BASE_DIR, 'train.py'), LOG_DIR))
#LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
#LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

# Load Frustum Datasets. Use default data paths.
TRAIN_DATASET = provider.FrustumDataset("pc_radar_2",npoints=NUM_POINT, split='val',
                                        random_flip=False, random_shift=False, rotate_to_center=False,
                                        overwritten_data_path=None, from_rgb_detection=False, one_hot=False,
                                        all_batches=False,translate_radar_center=True,store_data=True,proposals_3 = True)
#TEST_DATASET = provider.FrustumDataset(npoints=NUM_POINT, split='val',
#    rotate_to_center=True, random_flip=True, random_shift=True, one_hot=True)
#TEST_DATASET = provider.FrustumDataset(npoints=NUM_POINT, split='val',
#    rotate_to_center=True, one_hot=True)