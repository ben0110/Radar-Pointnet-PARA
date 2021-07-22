''' Provider class and helper functions for Frustum PointNets.

Author: Charles R. Qi
Date: September 2017
'''
from __future__ import print_function

#import cPickle as pickle
#import pcl
import sys
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from box_util import box3d_iou
#from model_util import g_type2class, g_class2type, g_type2onehotclass
#from model_util import g_type_mean_size
#from model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER
from dataset import KittiDataset
from collections import Counter
import kitti_utils
try:
    raw_input  # Python 2
except NameError:
    raw_input = input  # Python 3


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
def get_radar_mask(input,input_radar):
    radar_mask = np.zeros((input.shape[0]), dtype=np.float32)
    gt_boxes3d = np.zeros((len(input_radar), 7), dtype=np.float32)
    for k in range(len(input_radar)):
        gt_boxes3d[k,0:3]= input_radar[k,0:3]
        gt_boxes3d[k, 3]= 4.0
        gt_boxes3d[k, 4]= 4.0
        gt_boxes3d[k, 5]= 4.0 # np.tan(45*np.pi/180)*input_radar[k,2]*2
    gt_corners = kitti_utils.boxes3d_to_corners3d(gt_boxes3d, transform=False)
    for k in range(len(gt_corners)):
        box_corners = gt_corners[k]
        fg_pt_flag = kitti_utils.in_hull(input[:,0:3], box_corners)
        radar_mask[fg_pt_flag] = 1.0
    return radar_mask,gt_boxes3d[k, 5]





class FrustumDataset(object):
    ''' Dataset class for Frustum PointNets training/evaluation.
    Load prepared KITTI data from pickled files, return individual data element
    [optional] along with its annotations.
    '''

    def __init__(self, npoints, split,
                 random_flip=False, random_shift=False, rotate_to_center=False,
                 overwritten_data_path=None, from_rgb_detection=False, one_hot=False):
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
        self.dataset_kitti = KittiDataset(root_dir='/home/amben/frustum-pointnets_RSC/dataset/', mode='TRAIN', split=split)

        self.npoints = npoints
        self.random_flip = random_flip
        self.random_shift = random_shift
        self.rotate_to_center = rotate_to_center
        self.one_hot = one_hot
        if overwritten_data_path is None:
            overwritten_data_path = os.path.join(ROOT_DIR,
                                                 'kitti/frustum_carpedcyc_%s.pickle' % (split))

        self.from_rgb_detection = from_rgb_detection
        if from_rgb_detection:
            with open(overwritten_data_path, 'rb') as fp:
                self.id_list = pickle.load(fp)
                self.box2d_list = pickle.load(fp)
                self.input_list = pickle.load(fp)
                self.type_list = pickle.load(fp)
                # frustum_angle is clockwise angle from positive x-axis
                self.frustum_angle_list = pickle.load(fp)
                self.prob_list = pickle.load(fp)
        else:
            #with open(overwritten_data_path, 'rb') as fp:
            #load list of frames
            self.id_list = self.dataset_kitti.sample_id_list
            print("id_list",len(self.id_list))
            #fil = np.zeros((len(self.id_list)))
            #for i in range(len(self.id_list)):
            #    print(self.id_list[i])
            #    gt_obj_list = self.dataset_kitti.filtrate_objects(self.dataset_kitti.get_label(self.id_list[i]))
            #    print(len(gt_obj_list))
                #gt_boxes3d = kitti_utils.objs_to_boxes3d(gt_obj_list)
                #print(gt_boxes3d)
            #    if(len(gt_obj_list)==1):
            #        fil[i]=1

            #self.id_list= np.extract(fil,self.id_list)
            self.index_batch = []
            self.box_present = []
            self.radar_OI = []
            self.radar_mask_len = []
            self.cls_labels_len = []
            self.cls_labels_orig = []
            self.width_list=[]

            for i in range(len(self.id_list)):
                pc_input = self.dataset_kitti.get_lidar(self.id_list[i])
                pc_radar = self.dataset_kitti.get_radar(self.id_list[i])
                gt_obj_list = self.dataset_kitti.filtrate_objects(self.dataset_kitti.get_label(self.id_list[i]))

                gt_boxes3d = kitti_utils.objs_to_boxes3d(gt_obj_list)
                corners3d = kitti_utils.boxes3d_to_corners3d(gt_boxes3d, transform=True)

                cls_label = np.zeros((pc_input.shape[0]), dtype=np.int32)
                gt_corners = kitti_utils.boxes3d_to_corners3d(gt_boxes3d, transform=True)
                for k in range(gt_boxes3d.shape[0]):
                    box_corners = gt_corners[k]
                    fg_pt_flag = kitti_utils.in_hull(pc_input[:, 0:3], box_corners)
                    cls_label[fg_pt_flag] = k+1
                print("indice",self.id_list[i] )
                print("number of boxes",gt_boxes3d.shape[0])
                print("cls_label nubmer", np.count_nonzero(cls_label > 0 ))
                print("number of radar pts present in frame",len(pc_radar))
                for j in range(len(pc_radar)):
                    radar_mask,width = get_radar_mask(pc_input, pc_radar[j].reshape(-1, 3))
                    # check label present in radar ROI
                    print("radar_mask",np.count_nonzero(radar_mask== 1))
                    label_radar_intersection = radar_mask*cls_label

                    print("intersection", np.count_nonzero(label_radar_intersection > 0))
                                        # if there's one
                    labels_present=[]
                    for m in range(gt_boxes3d.shape[0]):
                          if(np.isin(m+1,label_radar_intersection)):
                              labels_present.append(m+1)
                    print("labels present", labels_present)
                    if (len(labels_present)==1):
                        self.radar_mask_len.append(np.count_nonzero(radar_mask == 1))
                        self.index_batch.append(self.id_list[i])
                        self.radar_OI.append(j)
                        self.box_present.append(labels_present[0])
                        self.cls_labels_len.append(np.count_nonzero(label_radar_intersection > 0))
                        self.cls_labels_orig.append(np.count_nonzero(cls_label == labels_present[0]))
                        self.width_list.append(width)
            print("retained indices",self.index_batch)
            print("len retained indices", len(self.index_batch))
            with open('radar_batches_stats_'+split+'_static_4mw.csv',mode='w') as csv_file:
                fieldnames= ['index_batch','radar_mask_len','radar_OI','box_present','cls_labels_len','width']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                for i in range(len(self.index_batch)):
                    writer.writerow({'index_batch':self.index_batch[i],'radar_mask_len': self.radar_mask_len[i], 'radar_OI': self.radar_OI[i], 'box_present': self.box_present[i],'cls_labels_len':self.cls_labels_len[i],'width':self.width_list[i]})
                    # keep this as a batch
                    # if there's isn't
                    # than forget about it


            #print("id_list_filtered", len(self.id_list))
            """
                self.input_list=[]
                self.box3d_list=[]
                self.label_list=[]
                self.type_list=[]
                self.heading_list=[]
                self.size_list=[]
                self.frustum_angle_list=[]
                for i in range(len(self.id_list)):

                    #BOX3D_IN_CORNERS FORMAT
                    gt_obj_list = dataset_kitti.filtrate_objects(dataset_kitti.get_label(self.id_list[i]))
                    gt_boxes3d = kitti_utils.objs_to_boxes3d(gt_obj_list)
                    print(gt_boxes3d)
                    self.box3d_list.append(kitti_utils.boxes3d_to_corners3d(gt_boxes3d, transform=True))

                    #INPUT_DATA_LIST
                    input = dataset_kitti.get_lidar(self.id_list[i])
                    self.input_list.append(input)

                    #LABEL_LIST
                    cls_label = np.zeros((self.input_list[i].shape[0]), dtype=np.int32)
                    gt_corners = kitti_utils.boxes3d_to_corners3d(gt_boxes3d, transform=True)
                    for k in range(gt_boxes3d.shape[0]):
                        box_corners = gt_corners[k]
                        fg_pt_flag = kitti_utils.in_hull(self.input_list[i][:,0:3], box_corners)
                        cls_label[fg_pt_flag] = 1

                    #print(cls_label.shape)
                    print("cls_label", (np.count_nonzero(cls_label == 1)))
                    
                    label_pts = np.ndarray((cls_label_count, 3))
                    j = 0
                    c = np.ndarray((len(input), 3))

                    for i in range(len(input)):
                        if (cls_label[i] == 1):
                            c[i] = np.array([1.0, 0.0, 0.0])
                            label_pts[j] = input[i,0:3]
                            j = j + 1
                        else:
                            c[i] = np.array([0.0, 0.0, 1.0])

                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection="3d")
                    ax.scatter(label_pts[:, 0], label_pts[:, 1], label_pts[:, 2])
                    plt.show()

                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection="3d")
                    ax.scatter(input[:, 0], input[:, 1], input[:, 2], c=c, s=1)
                    plt.show()
                    
                    self.label_list.append(cls_label)

                    #TYPE_LIST
                    self.type_list.append("Pedestrian")
                    #HEADING_LIST
                    self.heading_list.append(gt_boxes3d[:,6])

                    #SIZE_LIST l,w,h
                    self.size_list.append(gt_boxes3d[:,3:6])
                    #frustum_angle with 0.0 populate
                    self.frustum_angle_list.append(0.0)
                """
                # box2d in corners format
                #self.box2d_list = pickle.load(fp)
                # box3d in corners format
                #self.box3d_list = pickle.load(fp)
                # point cloud, hole or frustum filtered? looks like frustrum filtered because number of pc is too small
                #self.input_list = pickle.load(fp)
                # from frustrum point cloud which one belongs to label
                #self.label_list = pickle.load(fp)
                # for each 2d box/frustrum point cloud, detected object
                #self.type_list = pickle.load(fp)
                # rotation of 3d label box (ry)
                #self.heading_list = pickle.load(fp)
                # array of l,w,h
                #self.size_list = pickle.load(fp)
                # frustum_angle is clockwise angle from positive x-axis
                #self.frustum_angle_list = pickle.load(fp)


    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        ''' Get index-th element from the picked file dataset. '''
        # ------------------------------ INPUTS ----------------------------
        #rot_angle = self.get_center_view_rot_angle(index)
        # load radar points
        input_radar = self.dataset_kitti.get_radar(self.id_list[index])
        input = self.dataset_kitti.get_lidar(self.id_list[index])
        radar_mask = get_radar_mask(input, input_radar)
        num_point_fil = np.count_nonzero(radar_mask == 1)
        radar_idx =np.argwhere(radar_mask==1)
        input = input[radar_idx.reshape(-1)]
        print(input.shape)

        pts_rect = input[:, 0:3]
        pts_intensity = input[:, 3:]
        if self.npoints < len(pts_rect):

            # print(len(pts_rect))
            print(pts_rect.shape)
            pts_depth = pts_rect[:, 2]
            pts_near_flag = pts_depth < 20.0
            far_idxs_choice = np.where(pts_near_flag == 0)[0]
            near_idxs = np.where(pts_near_flag == 1)[0]
            # print(len(pts_depth),len(far_idxs_choice),len(near_idxs),self.npoints, self.npoints - len(far_idxs_choice))
            near_idxs_choice = np.random.choice(near_idxs, self.npoints - len(far_idxs_choice), replace=False)

            choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                if len(far_idxs_choice) > 0 else near_idxs_choice
            np.random.shuffle(choice)
        else:
            if (self.npoints / 2) > len(pts_rect):
                diff = int(self.npoints / 2 - len(pts_rect))
                # print(diff)
                add_pts = np.zeros((diff, 3), dtype=np.float32)
                add_int = np.zeros((diff, 3), dtype=np.float32)
                # print("add_int", add_int[0])
                pts_rect = np.concatenate((pts_rect, add_pts), axis=0)
                pts_intensity = np.concatenate((pts_intensity, add_int), axis=0)
            choice = np.arange(0, len(pts_rect), dtype=np.int32)
            if self.npoints > len(pts_rect):
                # print(len(pts_rect),self.npoints - len(pts_rect))
                extra_choice = np.random.choice(choice, self.npoints - len(pts_rect), replace=False)
                choice = np.concatenate((choice, extra_choice), axis=0)
        np.random.shuffle(choice)
        # print(len(pts_rect))
        ret_pts_rect = pts_rect[choice, :]
        # ret_pts_rect=pts_rect
        # TODO don't use intensity feature or try a method to add rgb
        # ret_pts_intensity = pts_intensity[choice] - 0.5  # translate intensity to [-0.5, 0.5]
        ret_pts_intensity = pts_intensity[choice]
        pts_features = [ret_pts_intensity.reshape(-1, 3)]
        ret_pts_features = np.concatenate(pts_features, axis=1) if pts_features.__len__() > 1 else pts_features[0]
        input = np.concatenate((ret_pts_rect, ret_pts_features), axis=1)

        radar_mask = get_radar_mask(input,input_radar)


        gt_obj_list = self.dataset_kitti.filtrate_objects(self.dataset_kitti.get_label(self.id_list[index]))
        gt_boxes3d = kitti_utils.objs_to_boxes3d(gt_obj_list)
        corners3d = kitti_utils.boxes3d_to_corners3d(gt_boxes3d, transform=True)

        cls_label = np.zeros((input.shape[0]), dtype=np.int32)
        gt_corners = kitti_utils.boxes3d_to_corners3d(gt_boxes3d, transform=True)
        for k in range(gt_boxes3d.shape[0]):
            box_corners = gt_corners[k]
            fg_pt_flag = kitti_utils.in_hull(input[:, 0:3], box_corners)
            cls_label[fg_pt_flag] = 1

        type="Pedestrian"
        center = gt_boxes3d[:,0:3]
        #closest_radar_point = get_closest_radar_point(center,input_radar)
        heading = gt_boxes3d[:, 6]

        size=gt_boxes3d[:, 3:6]
        # frustum_angle with 0.0 populate
        frustum_angle=0.0


        rot_angle=0.0
        # Compute one hot vector
        if self.one_hot:
            cls_type = type
            assert (cls_type in ['Car', 'Pedestrian', 'Cyclist'])
            one_hot_vec = np.zeros((3))
            one_hot_vec[g_type2onehotclass[cls_type]] = 1

        # Get point cloud
        if self.rotate_to_center:
            point_set = self.get_center_view_point_set(input,frustum_angle)
        else:
            point_set = input
        # Resample

        #print(point_set.shape[0],self.npoints)
        #choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        #print(len(choice))
        #point_set = point_set[choice, :]

        if self.from_rgb_detection:
            if self.one_hot:
                return point_set, rot_angle, self.prob_list[index], one_hot_vec
            else:
                return point_set, rot_angle, self.prob_list[index]

        # ------------------------------ LABELS ----------------------------
        seg = cls_label
        #seg = seg[choice]
        #print("batch seg 3asba:", np.count_nonzero(seg == 1))

        # Get center point of 3D box
        if self.rotate_to_center:
            box3d_center = self.get_center_view_box3d_center(corners3d,frustum_angle)
        else:
            box3d_center = self.get_box3d_center(corners3d)

        # Heading
        if self.rotate_to_center:
            heading_angle = heading - rot_angle
        else:
            heading_angle = heading

        # Size
        size_class, size_residual = size2class(size,
                                               type)

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
        #print(angle_class,angle_residual)
        if self.one_hot:
            return point_set, seg, box3d_center, angle_class, angle_residual, \
                   size_class, size_residual, rot_angle, one_hot_vec,radar_mask
        else:
            return point_set, seg, box3d_center, angle_class, angle_residual, \
                   size_class, size_residual, rot_angle,radar_mask

    def get_center_view_rot_angle(self, frustrum_angle):
        ''' Get the frustum rotation angle, it isshifted by pi/2 so that it
        can be directly used to adjust GT heading angle '''
        return 0.0 #np.pi / 2.0 + self.frustum_angle_list[index]

    def get_box3d_center(self, corners3d):
        ''' Get the center (XYZ) of 3D bounding box. '''
        corners3d= corners3d.reshape((8,3))
        box3d_center = (corners3d[0, :] + \
                        corners3d[6, :]) / 2.0
        return box3d_center

    def get_center_view_box3d_center(self, box3d,frustrum_angle):
        ''' Frustum rotation of 3D bounding box center. '''
        box3d= box3d.reshape((8,3))
        box3d_center = (box3d[0, :] + box3d[6, :]) / 2.0
        rotate_pc_along_y(np.expand_dims(box3d_center, 0), self.get_center_view_rot_angle(frustrum_angle)).squeeze()

        return rotate_pc_along_y(np.expand_dims(box3d_center, 0), self.get_center_view_rot_angle(frustrum_angle)).squeeze()

    def get_center_view_box3d(self, index):
        ''' Frustum rotation of 3D bounding box corners. '''
        box3d = self.box3d_list[index]
        box3d_center_view = np.copy(box3d)
        return rotate_pc_along_y(box3d_center_view, \
                                 self.get_center_view_rot_angle(index))

    def get_center_view_point_set(self, input,frustrum_angle ):
        ''' Frustum rotation of point clouds.
        NxC points with first 3 channels as XYZ
        z is facing forward, x is left ward, y is downward
        '''
        # Use np.copy to avoid corrupting original data
        point_set = np.copy(input)
        return rotate_pc_along_y(point_set, \
                                 self.get_center_view_rot_angle(frustrum_angle))


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
    l, w, h = box_size
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2];
    y_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2];
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2];
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + center[0];
    corners_3d[1, :] = corners_3d[1, :] + center[1];
    corners_3d[2, :] = corners_3d[2, :] + center[2];
    corners_3d = np.transpose(corners_3d)
    return corners_3d


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
    ty += h / 2.0
    return h, w, l, tx, ty, tz, ry


if __name__ == '__main__':
    import mayavi.mlab as mlab

    sys.path.append(os.path.join(ROOT_DIR, 'mayavi'))
    from viz_util import draw_lidar, draw_gt_boxes3d

    median_list = []
    dataset = FrustumDataset(1024, split='val',
                             rotate_to_center=True, random_flip=True, random_shift=True)
    for i in range(len(dataset)):
        data = dataset[i]
        print(('Center: ', data[2], \
               'angle_class: ', data[3], 'angle_res:', data[4], \
               'size_class: ', data[5], 'size_residual:', data[6], \
               'real_size:', g_type_mean_size[g_class2type[data[5]]] + data[6]))
        print(('Frustum angle: ', dataset.frustum_angle_list[i]))
        median_list.append(np.median(data[0][:, 0]))
        print((data[2], dataset.box3d_list[i], median_list[-1]))
        box3d_from_label = get_3d_box(class2size(data[5], data[6]), class2angle(data[3], data[4], 12), data[2])

        ps = data[0]
        seg = data[1]
        fig = mlab.figure(figure=None, bgcolor=(0.4, 0.4, 0.4), fgcolor=None, engine=None, size=(1000, 500))
        mlab.points3d(ps[:, 0], ps[:, 1], ps[:, 2], seg, mode='point', colormap='gnuplot', scale_factor=1, figure=fig)
        mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2, figure=fig)
        draw_gt_boxes3d([box3d_from_label], fig, color=(1, 0, 0))
        mlab.orientation_axes()
        raw_input()
    print(np.mean(np.abs(median_list)))
