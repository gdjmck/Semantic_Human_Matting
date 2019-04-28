"""
    dataset create
Author: Zhengwei Li
Date  : 2018/12/24
"""
import cv2
import os
import random as r
import numpy as np
import imutils
import scipy.stats

import torch
import torch.nn as nn
import torch.utils.data as data


def read_files(data_dir, file_name={}):

    image_name = os.path.join(data_dir, 'image', file_name['image'])
    trimap_name = os.path.join(data_dir, 'trimap', file_name['trimap'])
    alpha_name = os.path.join(data_dir, 'alpha', file_name['alpha'])

    image = cv2.imread(image_name, -1)
    trimap = cv2.imread(trimap_name, -1)
    alpha = cv2.imread(alpha_name, -1)
    if (image is None) or (trimap is None) or (alpha is None):
        print(image_name, ' missing')

    return image, trimap, alpha


def random_scale_and_creat_patch(image, trimap, alpha, patch_size):
    # short side scale to patch_size
    h, w, c = image.shape
    scale = max(patch_size*1.0/h, patch_size*1.0/w)
    target_shape = (int(np.round(w*scale)), int(np.round(h*scale)))
    assert target_shape[0] == patch_size or target_shape[1] == patch_size
    assert target_shape[0] >= patch_size and target_shape[1] >= patch_size
    image = cv2.resize(image, target_shape, interpolation=cv2.INTER_CUBIC)
    trimap = cv2.resize(trimap, target_shape, interpolation=cv2.INTER_NEAREST)
    alpha = cv2.resize(alpha, target_shape, interpolation=cv2.INTER_CUBIC)
    if (0 in list(image.shape)) or (0 in list(trimap.shape)) or (0 in list(alpha.shape)):
        print('random scale makes Null image')
    # creat patch
    h, w, c = image.shape
    x = r.randrange(0, w - patch_size) if w > patch_size else 0
    y = r.randrange(0, h - patch_size) if h > patch_size else 0
    image = image[y:y + patch_size, x:x+patch_size, :]
    trimap = trimap[y:y + patch_size, x:x+patch_size]
    alpha = alpha[y:y+patch_size, x:x+patch_size, :]
    assert image.shape[0] == patch_size and image.shape[1] == patch_size
    if (0 in list(image.shape)) or (0 in list(trimap.shape)) or (0 in list(alpha.shape)):
        print('create patch makes Null image')

    return image, trimap, alpha


def random_flip(image, trimap, alpha):
    '''
    if r.random() < 0.5:
        image = cv2.flip(image, 0)
        trimap = cv2.flip(trimap, 0)
        alpha = cv2.flip(alpha, 0)
    '''
    if r.random() < 0.5:
        image = cv2.flip(image, 1)
        trimap = cv2.flip(trimap, 1)
        alpha = cv2.flip(alpha, 1)
    return image, trimap, alpha
       
def np2Tensor(array):
    ts = (2, 0, 1)
    tmp = array.copy()
    tensor = torch.FloatTensor(tmp.transpose(ts).astype(float))    
    return tensor

class human_matting_data(data.Dataset):
    """
    human_matting
    """

    def __init__(self, root_dir, imglist, patch_size, rotate=None):
        super().__init__()
        self.data_root = root_dir
        self.rotate = rotate
        if rotate:
            self.rotate = abs(self.rotate)

        self.patch_size = patch_size
        '''
        with open(imglist) as f:
            self.imgID = f.readlines()
        '''
        self.imgID = os.listdir(os.path.join(root_dir, 'trimap'))
        self.num = len(self.imgID)
        print("Dataset : file number %d"% self.num)




    def __getitem__(self, index):
        # read files
        image, trimap, alpha = read_files(self.data_root, 
                                          file_name={'image': self.imgID[index].strip()[:-4]+'.jpg',
                                                     'trimap': self.imgID[index],
                                                     'alpha': self.imgID[index]})
        assert alpha.shape[0] == trimap.shape[0]
        assert alpha.shape[1] == trimap.shape[1]
        if image.shape[0] != alpha.shape[0]:
            image_ratio = image.shape[0] / image.shape[1]
            alpha_ratio = alpha.shape[0] / alpha.shape[1]
            if np.fabs(image_ratio - alpha_ratio) < 1e-2:
                image = cv2.resize(image, (alpha.shape[1], alpha.shape[0]))
                assert image.shape[0] == alpha.shape[0]
                assert image.shape[1] == alpha.shape[1]
            else:
                print(self.imgID[index], ' NEEDS TO BE ELIMINATED')
        if 0 in list(image.shape):
            print(image.shape)
            print(self.imgID[index], ' Image is None')
        if 0 in list(trimap.shape):
            print(trimap.shape)
            print(self.imgID[index], ' trimap is None')
        if 0 in list(alpha.shape):
            print(alpha.shape)
            print(self.imgID[index], ' alpha is None')
        # NOTE ! ! !
        # trimap should be 3 classes : fg, bg. unsure
        trimap[trimap==0] = 0
        trimap[trimap==128] = 1
        trimap[trimap==255] = 2

        # augmentation
        image, trimap, alpha = random_scale_and_creat_patch(image, trimap, alpha, self.patch_size)
        image, trimap, alpha = random_flip(image, trimap, alpha)
        trimap = np.expand_dims(trimap, -1)


        # normalize
        image = (image.astype(np.float32)  - (114., 121., 134.,)) / 255.0
        alpha = alpha.astype(np.float32) / 255.0
        # rotate angle sampled from truncated normal distribution
        if self.rotate:
            rot_angle = scipy.stats.truncnorm.rvs(-self.rotate, self.rotate, 0, 3)
            image = imutils.rotate(image, rot_angle)
            alpha = imutils.rotate(alpha, rot_angle)
            trimap = imutils.rotate(trimap, rot_angle)
        # to tensor
        image = np2Tensor(image)
        trimap = np2Tensor(trimap)
        alpha = np2Tensor(alpha)

        trimap = trimap[0,:,:].unsqueeze_(0)
        alpha = alpha[-1,:,:].unsqueeze_(0)

        sample = {'image': image, 'trimap': trimap, 'alpha': alpha}

        return sample

    def __len__(self):
        return self.num
