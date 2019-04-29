"""
    dataset create
Author: Zhengwei Li
Date  : 2018/12/24
"""
import cv2
import os
import random as r
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.utils.data as data


def read_files(data_dir, file_name={}):

    image_name = os.path.join(data_dir, 'image', file_name['image'])
    alpha_name = os.path.join(data_dir, 'alpha', file_name['alpha'])

    image = cv2.imread(image_name, -1)
    alpha = cv2.imread(alpha_name, -1)

    return image, alpha


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


def border_fillin(img):
    h, w = img.shape[:2]
    pad_len = abs(h-w)
    if pad_len == 0:
        return img
    elif h > w:
        return cv2.copyMakeBorder(img, 0, 0, int(pad_len/2.), pad_len-int(pad_len/2.), borderType=cv2.BORDER_CONSTANT, value=0)
    else:
        return cv2.copyMakeBorder(img, int(pad_len/2.), pad_len-int(pad_len/2.), 0, 0, borderType=cv2.BORDER_CONSTANT, value=0)


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
    if len(array.shape) == 2:
        array = np.expand_dims(array, -1)
    ts = (2, 0, 1)
    tmp = array.copy()
    tensor = torch.FloatTensor(tmp.transpose(ts).astype(float))    
    return tensor

class human_matting_data(data.Dataset):
    """
    human_matting
    """

    def __init__(self, root_dir, imglist, patch_size, anomalist='anonymous.pkl'):
        super().__init__()
        self.data_root = root_dir

        self.patch_size = patch_size
        with open(anomalist, 'rb') as f:
            self.anomalist = pickle.load(f)
        self.imgID = os.listdir(os.path.join(root_dir, 'alpha'))
        print('number of anonymous: ', len(self.anomalist), '\t', len(self.anomalist)/len(self.img))
        self.num = len(self.imgID)
        print("Dataset : file number %d"% self.num)


    def __getitem__(self, index):
        # read files
        anomaly = torch.Tensor([1]) if self.imgID[index] in self.anomalist else torch.Tensor([0])
        image, alpha = read_files(self.data_root, 
                                          file_name={'image': self.imgID[index].strip()[:-4]+'.jpg',
                                                     'alpha': self.imgID[index]})
        if alpha.shape[-1] == 4:
            alpha = alpha[..., -1:]
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
        if 0 in list(alpha.shape):
            print(alpha.shape)
            print(self.imgID[index], ' alpha is None')

        image = border_fillin(image)
        image = cv2.resize(image, (self.patch_size, self.patch_size), interpolation=cv2.INTER_CUBIC)
        alpha = border_fillin(alpha)
        alpha = cv2.resize(alpha, (self.patch_size, self.patch_size), interpolation=cv2.INTER_CUBIC)
        # normalize
        image = (image.astype(np.float32)  - (114., 121., 134.,)) / 255.0
        alpha = alpha.astype(np.float32) / 255.0
        # to tensor
        image = np2Tensor(image)
        alpha = np2Tensor(alpha)

        alpha = alpha[-1,:,:].unsqueeze_(0)

        sample = {'image': image, 'alpha': alpha, 'anomaly': anomaly}

        return sample

    def __len__(self):
        return self.num
