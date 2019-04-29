"""
    train

Author: Zhengwei Li
Date  : 2018/12/24
"""

from tensorboardX import SummaryWriter
import numpy as np
import argparse
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import time
import os
from data import dataset
from model import network, utils, Anomaly_Net
import torch.nn.functional as F


def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Fast portrait matting !')
    parser.add_argument('--dataDir', default='./DATA/', help='dataset directory')
    parser.add_argument('--saveDir', default='./ckpt', help='model save dir')
    parser.add_argument('--trainData', default='human_matting_data', help='train dataset name')
    parser.add_argument('--trainList', default='./data/list.txt', help='train img ID')
    parser.add_argument('--anomalist', default='./segments/anonymous.pkl', help='anonymous list file')
    parser.add_argument('--load', default= 'human_matting', help='save model')

    parser.add_argument('--finetuning', action='store_true', default=False, help='finetuning the training')
    parser.add_argument('--without_gpu', action='store_true', default=False, help='no use gpu')

    parser.add_argument('--nThreads', type=int, default=4, help='number of threads for data loading')
    parser.add_argument('--train_batch', type=int, default=8, help='input batch size for train')
    parser.add_argument('--patch_size', type=int, default=256, help='patch size for train')


    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--lrDecay', type=int, default=100)
    parser.add_argument('--lrdecayType', default='keep')
    parser.add_argument('--nEpochs', type=int, default=300, help='number of epochs to train')
    parser.add_argument('--save_epoch', type=int, default=1, help='number of epochs to save model')

    parser.add_argument('--train_phase', default= 'end_to_end', help='train phase')


    args = parser.parse_args()
    print(args)
    return args


def set_lr(args, epoch, optimizer):

    lrDecay = args.lrDecay
    decayType = args.lrdecayType
    if decayType == 'keep':
        lr = args.lr
    elif decayType == 'step':
        epoch_iter = (epoch + 1) // lrDecay
        lr = args.lr / 2**epoch_iter
    elif decayType == 'exp':
        k = math.log(2) / lrDecay
        lr = args.lr * math.exp(-k * epoch)
    elif decayType == 'poly':
        lr = args.lr * math.pow((1 - epoch / args.nEpochs), 0.9)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

  

class Train_Log():
    def __init__(self, args):
        self.args = args

        self.save_dir = os.path.join(args.saveDir, args.load)
        self.summary = SummaryWriter(self.save_dir)
        self.step_cnt = 1
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.save_dir_model = os.path.join(self.save_dir, 'model')
        if not os.path.exists(self.save_dir_model):
            os.makedirs(self.save_dir_model)

        if os.path.exists(self.save_dir + '/log.txt'):
            self.logFile = open(self.save_dir + '/log.txt', 'a')
        else:
            self.logFile = open(self.save_dir + '/log.txt', 'w')

    def add_scalar(self, scalar_name, scalar, step=None):
        if step is None:
            step = self.step_cnt
        self.summary.add_scalar(scalar_name, scalar, step)
        
    def add_histogram(self, var_name, value, step=None):
        if step is None:
            step = self.step_cnt
        self.summary.add_histogram(var_name, value, step)
        
    def add_trimap(self, image):
        image = image[0, :, :, :].detach().cpu().numpy().copy()
        bg = (image[0, :, :] > image[1, :, :]) & (image[0, :, :] > image[2, :, :])
        fg = (image[2, :, :] > image[0, :, :]) & (image[2, :, :] > image[1, :, :])
        figure_fg = np.zeros((image.shape[1], image.shape[2]))
        figure_unsure = np.zeros((image.shape[1], image.shape[2]))
        figure_fg[fg] = 128
        figure_unsure[(~bg)&(~fg)] = 128
        self.summary.add_image('trimap-fg', figure_fg, self.step_cnt, dataformats='HW')
        self.summary.add_image('trimap-unsure', figure_unsure, self.step_cnt, dataformats='HW')
        
    def add_trimap_gt(self, image):
        image = image.detach().cpu().numpy().copy()
        if len(image.shape) > 4:
            print('image shape too large', image.shape)
        if len(image.shape) == 4:
            image = image[0, :, :, :]
        assert image.shape[0] == 1
        image = image[0, :, :]
        figure_fg = image.copy()
        figure_unsure = image.copy()
        figure_unsure[figure_unsure!=1] = 0
        figure_unsure[figure_unsure==1] = 128
        figure_fg[image!=2] = 0
        figure_fg[image==2] = 128
        self.summary.add_image('trimap_gt_unsure', figure_unsure, self.step_cnt, dataformats='HW')
        self.summary.add_image('trimap_gt_fg', figure_fg, self.step_cnt, dataformats='HW')
        
    def add_image(self, tag, image):
        if isinstance(image, torch.autograd.Variable):
            image = image.data
        if len(image.shape) == 4:
            image = image[0, :, :, :]
        image = image.cpu().numpy()
        self.summary.add_image(tag, image, self.step_cnt)
        
    def step(self):
        self.step_cnt += 1

    def save_model(self, model, epoch):

        # epoch_out_path = "{}/ckpt_e{}.pth".format(self.save_dir_model, epoch)
        # print("Checkpoint saved to {}".format(epoch_out_path))

        # torch.save({
        #     'epoch': epoch,
        #     'state_dict': model.state_dict(),
        # }, epoch_out_path)

        lastest_out_path = "{}/ckpt_lastest.pth".format(self.save_dir_model)
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'step': self.step_cnt
        }, lastest_out_path)

        model_out_path = "{}/model_obj.pth".format(self.save_dir_model)
        torch.save(
            model,
            model_out_path)

    def load_model(self, model):

        lastest_out_path = "{}/ckpt_lastest.pth".format(self.save_dir_model)
        if self.args.without_gpu: # 用cpu载入模型到内存
            ckpt = torch.load(lastest_out_path, map_location='cpu')
        else: # 模型载入到显存
            ckpt = torch.load(lastest_out_path)
        state_dict = ckpt['state_dict'].copy()
        for key in ckpt['state_dict']:
            if key not in model.state_dict():
                print('missing key:\t', key)
                state_dict.pop(key)
        ckpt['state_dict'] = state_dict
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['state_dict'], strict=False)
        self.step_cnt = ckpt['step']
        #self.step_cnt = 1
        print("=> loaded checkpoint '{}' (epoch {}  total step {})".format(lastest_out_path, ckpt['epoch'], self.step_cnt))

        return start_epoch, model


    def save_log(self, log):
        self.logFile.write(log + '\n')


def main():

    print("=============> Loading args")
    args = get_args()

    print("============> Environment init")
    if args.without_gpu:
        print("use CPU !")
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            print("No GPU is is available !")

    print("============> Building model ...")
    model = Anomaly_Net.AnomalyNet()    
    model.to(device)

    print("============> Loading datasets ...")
    train_data = getattr(dataset, args.trainData)(root_dir = args.dataDir, \
                                                  imglist = args.trainList, \
                                                  patch_size = args.patch_size,
                                                  anomalist = args.anomalist)
    trainloader = DataLoader(train_data, 
                             batch_size=args.train_batch, 
                             drop_last=True, 
                             shuffle=True, 
                             num_workers=args.nThreads, 
                             pin_memory=False)
    model.train() 

    print('============> Loss function ', args.train_phase)
    print("============> Set optimizer ...")
    lr = args.lr
    
    L2_criterion = nn.MSELoss()
    BCE_criterion = nn.BCELoss()
    optimizer_encoder = optim.Adam(filter(lambda p: p.requires_grad, list(model.encoder.parameters())+list(model.decoder.parameters())), \
                                   lr=lr, betas=(0.9, 0.999), 
                                   weight_decay=0.0005)
    optimizer_discriminator = optim.Adam(filter(lambda p: p.requires_grad, model.classifier.parameters()),
                                    lr=lr, betas=(0.9, 0.999),
                                    weight_decay=0.0005)

    print("============> Start Train ! ...")
    start_epoch = 1
    trainlog = Train_Log(args)
    if args.finetuning:
        start_epoch, model = trainlog.load_model(model) 

    for epoch in range(start_epoch, args.nEpochs+1):

        loss_ = 0
        L_alpha_ = 0
        L_composition_ = 0
        L_cross_, L2_bg_ = 0, 0
        loss_array = []
        IOU_t_bg_, IOU_t_unsure_, IOU_t_fg_ = 0, 0, 0
        IOU_alpha_ = 0

        t0 = time.time()
        for i, sample_batched in enumerate(trainloader):
            print('batch ', i)
            img, alpha_gt, label = sample_batched['image'], sample_batched['alpha'], sample_batched['anomaly']
            img_in = torch.cat((img, alpha_gt), 1)
            matting = alpha_gt.repeat(1, 3, 1, 1) * img
            img_in, matting, label = img_in.to(device), matting.to(device), label.to(device)

            # update auto encoder
            matting_replica, _ = model(img_in)
            loss_encoder = L2_criterion(matting_replica, matting)
            
            optimizer_encoder.zero_grad()
            loss_encoder.backward()
            optimizer_encoder.step()

            # update discriminator
            _, probs = model(img_in)
            loss_discrim = BCE_criterion(probs, label)

            optimizer_discriminator.zero_grad()
            loss_discrim.backward()
            optimizer_discriminator.step()


if __name__ == "__main__":
    main()
