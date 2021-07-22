''' Util functions for training and evaluation.

Author: Charles R. Qi
Date: September 2017
'''

import numpy as np

def get_batch(dataset, idxs, start_idx, end_idx,
              num_point, num_channel,
              from_rgb_detection=False):
    ''' Prepare batch data for training/evaluation.
    batch size is determined by start_idx-end_idx

    Input:
        dataset: an instance of FrustumDataset class
        idxs: a list of data element indices
        start_idx: int scalar, start position in idxs
        end_idx: int scalar, end position in idxs
        num_point: int scalar
        num_channel: int scalar
        from_rgb_detection: bool
    Output:
        batched data and label
    '''
    if from_rgb_detection:
        return get_batch_from_rgb_detection(dataset, idxs, start_idx, end_idx,
            num_point, num_channel)

    bsize = end_idx-start_idx
    batch_data = np.zeros((bsize, num_point, num_channel))
    batch_label = np.zeros((bsize, num_point), dtype=np.int32)
    batch_center = np.zeros((bsize, 3))
    batch_heading_class = np.zeros((bsize,), dtype=np.int32)
    batch_heading_residual = np.zeros((bsize,))
    batch_size_class = np.zeros((bsize,), dtype=np.int32)
    batch_size_residual = np.zeros((bsize, 3))
    batch_rot_angle = np.zeros((bsize,))
    batch_radar_mask = np.zeros((bsize,num_point),dtype=np.float32)
    if dataset.one_hot:
        batch_one_hot_vec = np.zeros((bsize,3)) # for car,ped,cyc
    for i in range(bsize):
        if dataset.one_hot:
            ps,seg,center,hclass,hres,sclass,sres,rotangle,onehotvec,radar_mask = \
                dataset[idxs[i+start_idx]]
            batch_one_hot_vec[i] = onehotvec
        else:
            ps,seg,center,hclass,hres,sclass,sres,rotangle = \
                dataset[idxs[i+start_idx]]
        batch_data[i,...] = ps[:,0:num_channel]
        batch_label[i,:] = seg

        batch_center[i,:] = center
        batch_heading_class[i] = hclass
        batch_heading_residual[i] = hres
        batch_size_class[i] = sclass
        batch_size_residual[i] = sres
        batch_rot_angle[i] = rotangle
        #print(batch_label.shape)
        #print(batch_radar_mask.shape)
        batch_radar_mask[i,:]= radar_mask
    if dataset.one_hot:
        return batch_data, batch_label, batch_center, \
            batch_heading_class, batch_heading_residual, \
            batch_size_class, batch_size_residual, \
            batch_rot_angle, batch_one_hot_vec
    else:
        return batch_data, batch_label, batch_center, \
            batch_heading_class, batch_heading_residual, \
            batch_size_class, batch_size_residual, batch_rot_angle

def get_batch_mask(dataset, idxs, start_idx, end_idx,
              num_point, num_channel,
              from_rgb_detection=False):
    ''' Prepare batch data for training/evaluation.
    batch size is determined by start_idx-end_idx

    Input:
        dataset: an instance of FrustumDataset class
        idxs: a list of data element indices
        start_idx: int scalar, start position in idxs
        end_idx: int scalar, end position in idxs
        num_point: int scalar
        num_channel: int scalar
        from_rgb_detection: bool
    Output:
        batched data and label
    '''
    if from_rgb_detection:
        return get_batch_from_rgb_detection(dataset, idxs, start_idx, end_idx,
            num_point, num_channel)

    bsize = end_idx-start_idx
    batch_data = np.zeros((bsize, num_point, num_channel))
    batch_label = np.zeros((bsize, num_point), dtype=np.int32)
    batch_center = np.zeros((bsize, 3))
    batch_heading_class = np.zeros((bsize,), dtype=np.int32)
    batch_heading_residual = np.zeros((bsize,))
    batch_size_class = np.zeros((bsize,), dtype=np.int32)
    batch_size_residual = np.zeros((bsize, 3))
    batch_rot_angle = np.zeros((bsize,))
    batch_mask = np.zeros((bsize,1))
    if dataset.one_hot:
        batch_one_hot_vec = np.zeros((bsize,3)) # for car,ped,cyc
    for i in range(bsize):
        if dataset.one_hot:
            ps,seg,center,hclass,hres,sclass,sres,rotangle,onehotvec,label_mask = \
                dataset[idxs[i+start_idx]]
            batch_one_hot_vec[i] = onehotvec
        else:
            ps,seg,center,hclass,hres,sclass,sres,rotangle = \
                dataset[idxs[i+start_idx]]
        batch_data[i,...] = ps[:,0:num_channel]
        batch_label[i,:] = seg
        batch_center[i,:] = center
        batch_heading_class[i] = hclass
        batch_heading_residual[i] = hres
        batch_size_class[i] = sclass
        batch_size_residual[i] = sres
        batch_rot_angle[i] = rotangle
        batch_mask[i] = label_mask
        print("batch_mask",batch_mask)
    if dataset.one_hot:
        return batch_data, batch_label, batch_center, \
            batch_heading_class, batch_heading_residual, \
            batch_size_class, batch_size_residual, \
            batch_rot_angle, batch_one_hot_vec,batch_mask
    else:
        return batch_data, batch_label, batch_center, \
            batch_heading_class, batch_heading_residual, \
            batch_size_class, batch_size_residual, batch_rot_angle,batch_mask

def get_batch_bbox(dataset, idxs, start_idx, end_idx,
              num_point, num_channel,
              from_rgb_detection=False):
    ''' Prepare batch data for training/evaluation.
    batch size is determined by start_idx-end_idx

    Input:
        dataset: an instance of FrustumDataset class
        idxs: a list of data element indices
        start_idx: int scalar, start position in idxs
        end_idx: int scalar, end position in idxs
        num_point: int scalar
        num_channel: int scalar
        from_rgb_detection: bool
    Output:
        batched data and label
    '''
    if from_rgb_detection:
        return get_batch_from_rgb_detection(dataset, idxs, start_idx, end_idx,
            num_point, num_channel)

    bsize = end_idx-start_idx
    batch_data = np.zeros((bsize, num_point, num_channel))
    batch_center_mean = np.zeros((bsize, 3))
    batch_center = np.zeros((bsize, 3))
    batch_heading_class = np.zeros((bsize,), dtype=np.int32)
    batch_heading_residual = np.zeros((bsize,))
    batch_size_class = np.zeros((bsize,), dtype=np.int32)
    batch_size_residual = np.zeros((bsize, 3))
    if dataset.one_hot:
        batch_one_hot_vec = np.zeros((bsize,3)) # for car,ped,cyc
    for i in range(bsize):
        if dataset.one_hot:
            ps,mean_center,onehotvec, center,\
            heading_class,heading_residual,size_class,size_residual,id_AB = \
                dataset[idxs[i+start_idx]]
            batch_one_hot_vec[i] = onehotvec
        else:
            ps,mean_center,center,\
            heading_class,heading_residual,size_class,size_residual,id_AB = \
                dataset[idxs[i+start_idx]]
        batch_data[i,...] = ps[:,0:num_channel]
        batch_center_mean[i,:]= mean_center
        batch_center[i, :] = center
        batch_heading_class[i] = heading_class
        batch_heading_residual[i] = heading_residual
        batch_size_class[i] = size_class
        batch_size_residual[i] = size_residual
    if dataset.one_hot:
        return batch_data,batch_center_mean, batch_one_hot_vec,\
               batch_center,batch_heading_class,batch_heading_residual,\
                batch_size_class,batch_size_residual
    else:
        return batch_data,batch_center_mean,batch_center,batch_heading_class,batch_heading_residual,\
                batch_size_class,batch_size_residual

def get_batch_seg(dataset, idxs, start_idx, end_idx,
              num_point, num_channel,
              from_rgb_detection=False):
    ''' Prepare batch data for training/evaluation.
    batch size is determined by start_idx-end_idx

    Input:
        dataset: an instance of FrustumDataset class
        idxs: a list of data element indices
        start_idx: int scalar, start position in idxs
        end_idx: int scalar, end position in idxs
        num_point: int scalar
        num_channel: int scalar
        from_rgb_detection: bool
    Output:
        batched data and label
    '''
    if from_rgb_detection:
        return get_batch_from_rgb_detection(dataset, idxs, start_idx, end_idx,
            num_point, num_channel)

    bsize = end_idx-start_idx
    batch_data = np.zeros((bsize, num_point, num_channel))
    batch_label = np.zeros((bsize, num_point), dtype=np.int32)
    batch_radar_mask_list =[] #np.zeros((bsize, 32, num_point), dtype=np.int32)
    radar_rois = [] #np.zeros((bsize, 32, 7), dtype=np.float32)
    batch_rot_angle = np.zeros((bsize,))
    ids = []
    if dataset.one_hot:
        batch_one_hot_vec = np.zeros((bsize,3)) # for car,ped,cyc
    for i in range(bsize):
        if dataset.one_hot:
            ps,seg,onehotvec,radar_mask_list,radar_roi_param,idx = \
                dataset[idxs[i+start_idx]]
            batch_one_hot_vec[i] = onehotvec
        else:
            ps,seg,radar_mask_list,radar_roi_param,idx = \
                dataset[idxs[i+start_idx]]
        batch_data[i,...] = ps[:,0:num_channel]
        batch_label[i,:] = seg
        batch_radar_mask_list.append(radar_mask_list)
        radar_rois.append(radar_roi_param)
        #radar_rois.append(radar_roi_param)
        ids.append(idx)
    if dataset.one_hot:
        return batch_data, batch_label, batch_one_hot_vec,batch_radar_mask_list,radar_rois,ids
    else:
        return batch_data, batch_label,batch_radar_mask_list,radar_rois,ids

def get_batch_seg_cls(dataset, idxs, start_idx, end_idx,
              num_point, num_channel,
              from_rgb_detection=False):
    ''' Prepare batch data for training/evaluation.
    batch size is determined by start_idx-end_idx

    Input:
        dataset: an instance of FrustumDataset class
        idxs: a list of data element indices
        start_idx: int scalar, start position in idxs
        end_idx: int scalar, end position in idxs
        num_point: int scalar
        num_channel: int scalar
        from_rgb_detection: bool
    Output:
        batched data and label
    '''
    if from_rgb_detection:
        return get_batch_from_rgb_detection(dataset, idxs, start_idx, end_idx,
            num_point, num_channel)

    bsize = end_idx-start_idx
    batch_data = np.zeros((bsize, num_point, num_channel))
    batch_label = np.zeros((bsize, num_point), dtype=np.int32)
    batch_radar_mask_list = np.zeros((bsize, 16, num_point), dtype=np.int32)
    radar_rois = np.zeros((bsize, 16, 7), dtype=np.float32)
    batch_rot_angle = np.zeros((bsize,))
    gt_corners_batch = np.zeros((bsize,5,8,3),dtype=np.float32)
    ids = np.zeros((bsize,1))
    if dataset.one_hot:
        batch_one_hot_vec = np.zeros((bsize,3)) # for car,ped,cyc
    for i in range(bsize):
        if dataset.one_hot:
            ps,seg,onehotvec,radar_mask_list,radar_roi_param,idx,gt_corners = \
                dataset[idxs[i+start_idx]]
            batch_one_hot_vec[i] = onehotvec
        else:
            ps,seg,radar_mask_list,radar_roi_param,idx,gt_corners = \
                dataset[idxs[i+start_idx]]
        batch_data[i,...] = ps[:,0:num_channel]
        batch_label[i,:] = seg
        batch_radar_mask_list[i,...]=radar_mask_list
        radar_rois[i,:]=radar_roi_param
        gt_corners_batch[i,:]=gt_corners
        #radar_rois.append(radar_roi_param)
        ids[i,:]=idx
    if dataset.one_hot:
        return batch_data, batch_label,gt_corners_batch, batch_one_hot_vec,batch_radar_mask_list,radar_rois,ids
    else:
        return batch_data, batch_label,gt_corners_batch,batch_radar_mask_list,radar_rois,ids

def get_batch_seg_all(dataset, idxs, start_idx, end_idx,
              num_point, num_channel,
              from_rgb_detection=False):
    ''' Prepare batch data for training/evaluation.
    batch size is determined by start_idx-end_idx

    Input:
        dataset: an instance of FrustumDataset class
        idxs: a list of data element indices
        start_idx: int scalar, start position in idxs
        end_idx: int scalar, end position in idxs
        num_point: int scalar
        num_channel: int scalar
        from_rgb_detection: bool
    Output:
        batched data and label
    '''
    if from_rgb_detection:
        return get_batch_from_rgb_detection(dataset, idxs, start_idx, end_idx,
            num_point, num_channel)

    bsize = end_idx-start_idx
    batch_data = np.zeros((bsize, num_point, num_channel))
    batch_label = np.zeros((bsize, num_point), dtype=np.int32)
    ids = []
    if dataset.one_hot:
        batch_one_hot_vec = np.zeros((bsize,3)) # for car,ped,cyc
    for i in range(bsize):
        if dataset.one_hot:
            ps,seg,onehotvec,id = \
                dataset[idxs[i+start_idx]]
            batch_one_hot_vec[i] = onehotvec
        else:
            ps,seg,id = \
                dataset[idxs[i+start_idx]]
        batch_data[i,...] = ps[:,0:num_channel]
        batch_label[i,:] = seg
        ids.append(id)

    if dataset.one_hot:
        return batch_data, batch_label, batch_one_hot_vec,ids
    else:
        return batch_data, batch_label,ids


def get_batch_test(dataset, idxs, start_idx, end_idx,
              num_point, num_channel,
              from_rgb_detection=False):
    ''' Prepare batch data for training/evaluation.
    batch size is determined by start_idx-end_idx

    Input:
        dataset: an instance of FrustumDataset class
        idxs: a list of data element indices
        start_idx: int scalar, start position in idxs
        end_idx: int scalar, end position in idxs
        num_point: int scalar
        num_channel: int scalar
        from_rgb_detection: bool
    Output:
        batched data and label
    '''
    if from_rgb_detection:
        return get_batch_from_rgb_detection(dataset, idxs, start_idx, end_idx,
            num_point, num_channel)

    bsize = (end_idx-start_idx)+1
    batch_data = np.zeros((bsize, num_point, num_channel))
    batch_label = np.zeros((bsize, num_point), dtype=np.int32)
    batch_center = np.zeros((bsize, 3))
    batch_heading_class = np.zeros((bsize,), dtype=np.int32)
    batch_heading_residual = np.zeros((bsize,))
    batch_size_class = np.zeros((bsize,), dtype=np.int32)
    batch_size_residual = np.zeros((bsize, 3))
    batch_rot_angle = np.zeros((bsize,))
    batch_mask = np.zeros((bsize,1))
    if dataset.one_hot:
        batch_one_hot_vec = np.zeros((bsize,3)) # for car,ped,cyc
    for i in range(bsize):
        if dataset.one_hot:
            ps,seg,center,hclass,hres,sclass,sres,rotangle,onehotvec,label_mask = \
                dataset[idxs[i+start_idx]]
            batch_one_hot_vec[i] = onehotvec
        else:
            ps,seg,center,hclass,hres,sclass,sres,rotangle = \
                dataset[idxs[i+start_idx]]
        batch_data[i,...] = ps[:,0:num_channel]
        batch_label[i,:] = seg
        batch_center[i,:] = center
        batch_heading_class[i] = hclass
        batch_heading_residual[i] = hres
        batch_size_class[i] = sclass
        batch_size_residual[i] = sres
        batch_rot_angle[i] = rotangle
        batch_mask[i] = label_mask
    if dataset.one_hot:
        return batch_data, batch_label, batch_center, \
            batch_heading_class, batch_heading_residual, \
            batch_size_class, batch_size_residual, \
            batch_rot_angle, batch_one_hot_vec,batch_mask
    else:
        return batch_data, batch_label, batch_center, \
            batch_heading_class, batch_heading_residual, \
            batch_size_class, batch_size_residual, batch_rot_angle,batch_mask


def get_batch_from_rgb_detection(dataset, idxs, start_idx, end_idx,
                                 num_point, num_channel):
    bsize = end_idx-start_idx
    batch_data = np.zeros((bsize, num_point, num_channel))
    batch_rot_angle = np.zeros((bsize,))
    batch_prob = np.zeros((bsize,))
    if dataset.one_hot:
        batch_one_hot_vec = np.zeros((bsize,3)) # for car,ped,cyc
    for i in range(bsize):
        if dataset.one_hot:
            ps,rotangle,prob,onehotvec = dataset[idxs[i+start_idx]]
            batch_one_hot_vec[i] = onehotvec
        else:
            ps,rotangle,prob = dataset[idxs[i+start_idx]]
        batch_data[i,...] = ps[:,0:num_channel]
        batch_rot_angle[i] = rotangle
        batch_prob[i] = prob
    if dataset.one_hot:
        return batch_data, batch_rot_angle, batch_prob, batch_one_hot_vec
    else:
        return batch_data, batch_rot_angle, batch_prob


