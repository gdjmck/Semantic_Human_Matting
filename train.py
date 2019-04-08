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
from torch.utils.data import DataLoader
import time
import os
from data import dataset
from model import network, utils
import torch.nn.functional as F


def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Fast portrait matting !')
    parser.add_argument('--dataDir', default='./DATA/', help='dataset directory')
    parser.add_argument('--saveDir', default='./ckpt', help='model save dir')
    parser.add_argument('--trainData', default='human_matting_data', help='train dataset name')
    parser.add_argument('--trainList', default='./data/list.txt', help='train img ID')
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

    def add_scalar(self, scalar_name, scalar):
        self.summary.add_scalar(scalar_name, scalar, self.step_cnt)
        
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
        
    def add_image(self, image):
        if isinstance(image, torch.autograd.Variable):
            image = image.data
        image = image.cpu().numpy()
        self.summary.add_image('origin_image', image[0, :, :, :], self.step_cnt)
        
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
        ckpt = torch.load(lastest_out_path)
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['state_dict'])
        self.step_cnt = ckpt['step']
        #self.step_cnt = 1
        print("=> loaded checkpoint '{}' (epoch {}  total step {})".format(lastest_out_path, ckpt['epoch'], self.step_cnt))

        return start_epoch, model    


    def save_log(self, log):
        self.logFile.write(log + '\n')



def loss_function(args, img, trimap_pre, trimap_gt, alpha_pre, alpha_gt):

    # -------------------------------------
    # classification loss L_t
    # ------------------------
    # Cross Entropy 
    # criterion = nn.BCELoss()
    # trimap_pre = trimap_pre.contiguous().view(-1)
    # trimap_gt = trimap_gt.view(-1)
    # L_t = criterion(trimap_pre, trimap_gt)

    criterion = nn.CrossEntropyLoss()
    assert trimap_gt.shape[1] == 1
    L_t = criterion(trimap_pre, trimap_gt[:,0,:,:].long())
    IOU_t = utils.iou_pytorch((trimap_pre[:, 2, :, :]>trimap_pre[:, 0, :, :]) & (trimap_pre[:, 2, :, :]>trimap_pre[:, 1, :, :]), (trimap_gt[:, 0, :, :]==2))

    # -------------------------------------
    # prediction loss L_p
    # ------------------------
    eps = 1e-6
    # l_alpha
    L_alpha = torch.sqrt(torch.pow(alpha_pre - alpha_gt, 2.) + eps).mean()
    IOU_alpha = utils.iou_pytorch(alpha_pre>1e-5, alpha_gt>1e-5)

    # L_composition
    fg = torch.cat((alpha_gt, alpha_gt, alpha_gt), 1) * img
    fg_pre = torch.cat((alpha_pre, alpha_pre, alpha_pre), 1) * img

    L_composition = torch.sqrt(torch.pow(fg - fg_pre, 2.) + eps).mean()

    L_p = 0.5*L_alpha + 0.5*L_composition

    # train_phase
    if args.train_phase == 'pre_train_t_net':
        loss = L_t
    if args.train_phase == 'end_to_end':
        loss = L_p + 0.01*L_t
        
    return loss, L_alpha, L_composition, L_t, IOU_t, IOU_alpha


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
    model = network.net()    
    model.to(device)

    print("============> Loading datasets ...")
    train_data = getattr(dataset, args.trainData)(root_dir = args.dataDir, \
                                                  imglist = args.trainList, \
                                                  patch_size = args.patch_size)
    trainloader = DataLoader(train_data, 
                             batch_size=args.train_batch, 
                             drop_last=True, 
                             shuffle=True, 
                             num_workers=args.nThreads, 
                             pin_memory=False)

    print('============> Loss function ', args.train_phase)
    print("============> Set optimizer ...")
    lr = args.lr
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters() if args.train_phase == 'end_to_end' else model.t_net.parameters()), \
                                   lr=lr, betas=(0.9, 0.999), 
                                   weight_decay=0.0005)    

    print("============> Start Train ! ...")
    start_epoch = 1
    trainlog = Train_Log(args)
    if args.finetuning:
        start_epoch, model = trainlog.load_model(model) 

    model.train() 
    for epoch in range(start_epoch, args.nEpochs+1):

        loss_ = 0
        L_alpha_ = 0
        L_composition_ = 0
        L_cross_ = 0
        loss_array = []
        IOU_t_ = 0
        IOU_alpha_ = 0
        if args.lrdecayType != 'keep':
            lr = set_lr(args, epoch, optimizer)

        t0 = time.time()
        for i, sample_batched in enumerate(trainloader):
            print('batch ', i)
            img, trimap_gt, alpha_gt = sample_batched['image'], sample_batched['trimap'], sample_batched['alpha']
            img, trimap_gt, alpha_gt = img.to(device), trimap_gt.to(device), alpha_gt.to(device)

            trimap_pre, alpha_pre = model(img)
            loss, L_alpha, L_composition, L_cross, IOU_t, IOU_alpha = loss_function(args, 
                                                                  img,
                                                                  trimap_pre, 
                                                                  trimap_gt, 
                                                                  alpha_pre, 
                                                                  alpha_gt)
            print('Loss calculated %.4f, %.4f, %.4f' %(L_cross.item(), IOU_t.item(), IOU_alpha.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_ += loss.item()
            L_alpha_ += L_alpha.item()
            L_composition_ += L_composition.item()
            L_cross_ += L_cross.item()
            trainlog.add_scalar('T_net_loss', loss.item())
            IOU_t_ += IOU_t.item()
            trainlog.add_scalar('IOU_t', IOU_t.item())
            IOU_alpha_ += IOU_alpha.item()
            loss_array.append(loss.item())

            if i % 100 == 0:
                trainlog.add_trimap(trimap_pre)
                trainlog.add_trimap_gt(trimap_gt)
                trainlog.add_image(img)
            
            trainlog.step()

        print('Done iterating all training data')
        t1 = time.time()

        if epoch % args.save_epoch == 0:

            # speed = (t1 - t0) / 60 

            loss_ = loss_ / (i+1)
            L_alpha_ = L_alpha_ / (i+1)
            L_composition_ = L_composition_ / (i+1)
            L_cross_ = L_cross_ / (i+1)
            loss_var = np.var(loss_array)
            IOU_t_ = IOU_t_ / (i+1)
            IOU_alpha_ = IOU_alpha_ / (i+1)
            trainlog.add_scalar('avg_t_loss', L_cross_)
            trainlog.add_scalar('avg_t_loss_var', loss_var)
            trainlog.add_scalar('avg_IOU_t', IOU_t_)

            log = "[{} / {}] \tLr: {:.5f}\nloss: {:.5f}\tloss_p: {:.5f}\tloss_t: {:.5f}\tloss_var: {:.5f}\tIOU_t: {:.5f}\tIOU_alpha: {:.5f}\t" \
                     .format(epoch, args.nEpochs, 
                            lr, 
                            loss_, 
                            L_alpha_+L_composition_, 
                            L_cross_,
                            loss_var,
                            IOU_t_,
                            IOU_alpha_)
            print(log)
            trainlog.save_log(log)
            trainlog.save_model(model, epoch)


if __name__ == "__main__":
    main()
