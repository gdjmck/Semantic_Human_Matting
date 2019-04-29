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
from operator import add
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

alpha_iou_threshold = [0.1, 0.3, 0.5, 0.7, 0.9]
def loss_function(args, img, trimap_pre, trimap_gt, alpha_pre, alpha_gt):

    criterion = nn.CrossEntropyLoss()
    L1 = nn.L1Loss()
    # -------------------------------------
    # classification loss L_t
    # ------------------------
    # Cross Entropy 
    # criterion = nn.BCELoss()
    # trimap_pre = trimap_pre.contiguous().view(-1)
    # trimap_gt = trimap_gt.view(-1)
    # L_t = criterion(trimap_pre, trimap_gt)
    if args.train_phase != 'pre_train_m_net':
        assert trimap_gt.shape[1] == 1

        L1_t = L1(F.softmax(trimap_pre, dim=1)[:, 0, :, :], (trimap_gt[:, 0, :, :]==0).type(torch.FloatTensor))
        L_t = criterion(trimap_pre, trimap_gt[:,0,:,:].long())
        IOU_t = [utils.iou_pytorch((trimap_pre[:, 0, :, :]>trimap_pre[:, 1, :, :]) & (trimap_pre[:, 0, :, :]>trimap_pre[:, 2, :, :]), trimap_gt[:, 0, :, :]==0),
                 utils.iou_pytorch((trimap_pre[:, 1, :, :]>=trimap_pre[:, 0, :, :]) & (trimap_pre[:, 1, :, :]>=trimap_pre[:, 2, :, :]), trimap_gt[:, 0, :, :]==1),
                 utils.iou_pytorch((trimap_pre[:, 2, :, :]>trimap_pre[:, 0, :, :]) & (trimap_pre[:, 2, :, :]>trimap_pre[:, 1, :, :]), trimap_gt[:, 0, :, :]==2)]
    else: # train_phase == 'pre_train_m_net', L2_t = L_t = IOU_t = tensor(0.)
        L1_t = L_t = torch.Tensor([0.])
        IOU_t = [torch.Tensor([0.])]*3
    # -------------------------------------
    # prediction loss L_p
    # ------------------------
    # l_alpha
    L_alpha = L1(alpha_pre, alpha_gt)
    IOU_alpha = [utils.iou_pytorch(alpha_pre>thr, alpha_gt>thr).item() for thr in alpha_iou_threshold]

    # L_composition
    fg = torch.cat((alpha_gt, alpha_gt, alpha_gt), 1) * img
    fg_pre = torch.cat((alpha_pre, alpha_pre, alpha_pre), 1) * img

    L_composition = L1(fg_pre, fg)

    #L_p = 0.5*L_alpha + 0.5*L_composition
    L_p = L_alpha

    # train_phase
    if args.train_phase == 'pre_train_t_net':
        loss = L_t
    if args.train_phase == 'end_to_end':
        loss = L_p + 0.01*L_t
    if args.train_phase == 'pre_train_m_net':
        loss = L_p
        
    return loss, L_alpha, L_composition, L_t, L1_t, IOU_t, IOU_alpha


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
    model.train() 

    print('============> Loss function ', args.train_phase)
    print("============> Set optimizer ...")
    lr = args.lr
    train_params = model.parameters()
    target_network = model
    if args.train_phase == 'pre_train_t_net':
        train_params = model.t_net.parameters()
        target_network = model.t_net
    elif args.train_phase == 'pre_train_m_net':
        train_params = model.m_net.parameters()
        target_network = model.m_net
        model.t_net.eval()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, train_params), \
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
        IOU_alpha_ = [0] * 5
        if args.lrdecayType != 'keep':
            lr = set_lr(args, epoch, optimizer)

        t0 = time.time()
        for i, sample_batched in enumerate(trainloader):
            print('batch ', i)
            img, trimap_gt, alpha_gt = sample_batched['image'], sample_batched['trimap'], sample_batched['alpha']
            img, trimap_gt, alpha_gt = img.to(device), trimap_gt.to(device), alpha_gt.to(device)

            # end_to_end  or  pre_train_t_net
            if args.train_phase != 'pre_train_m_net':
                trimap_pre, alpha_pre = model(img)
                loss, L_alpha, L_composition, L_cross, L2_cross, IOU_t, IOU_alpha = loss_function(args, 
                                                                    img,
                                                                    trimap_pre, 
                                                                    trimap_gt, 
                                                                    alpha_pre, 
                                                                    alpha_gt)
                print("Loss calculated %.4f\nL2: %.2f\nbg IOU: %.2f\nunsure IOU: %.2f\nfg IOU: %.2f"%(L_cross.item(), L2_cross.item(), IOU_t[0].item(), IOU_t[1].item(), IOU_t[2].item()))
            else: # pre_train_m_net
                trimap_softmax = torch.zeros([trimap_gt.shape[0], 3, trimap_gt.shape[2], trimap_gt.shape[3]], dtype=torch.float32)
                trimap_softmax.scatter_(1, trimap_gt.long().data.cpu(), 1)
                trimap_softmax = trimap_softmax.to(device)
                #trimap_softmax = F.softmax(trimap_gt, dim=1)
                bg_gt, unsure_gt, fg_gt = torch.split(trimap_softmax, 1, dim=1)
                m_net_input = torch.cat((img, trimap_softmax), 1).to(device)
                alpha_r = model.m_net(m_net_input)
                alpha_p = fg_gt + unsure_gt * alpha_r
                loss, L_alpha, L_composition, L_cross, L2_cross, IOU_t, IOU_alpha = loss_function(args,
                                                                            img, 
                                                                            trimap_gt,
                                                                            trimap_gt, 
                                                                            alpha_p,
                                                                            alpha_gt)
                print('loss: %.5f\tL_composision: %.5f\tL_alpha: %.5f'%(loss.item(), L_composition.item(), L_alpha.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_ += loss.item()
            L_alpha_ += L_alpha.item()
            L_composition_ += L_composition.item()
            L_cross_ += L_cross.item()
            L2_bg_ += L2_cross.item()
            IOU_t_bg_ += IOU_t[0].item()
            IOU_t_unsure_ += IOU_t[1].item()
            IOU_t_fg_ += IOU_t[2].item()
            IOU_alpha_ = list(map(add, IOU_alpha_, IOU_alpha))
            loss_array.append(loss.item())
            
            # TENSORBOARD SCALARS
            trainlog.add_scalar('loss', loss.item())
            trainlog.add_scalar('T_net_loss', L_cross.item())
            trainlog.add_scalar('T_net_bg_L2', L2_cross.item())
            trainlog.add_scalar('M_net_alpha', L_alpha.item())
            trainlog.add_scalar('M_net_composition', L_composition.item())
            trainlog.add_scalar('IOU_t_bg', IOU_t[0].item())
            trainlog.add_scalar('IOU_t_unsure', IOU_t[1].item())
            trainlog.add_scalar('IOU_t_fg', IOU_t[2].item())
            if (i+1) % 100 == 0:
                for var_name, value in target_network.named_parameters():
                    # ignore unused parameters
                    if not hasattr(value.grad, 'data'):
                        continue
                    var_name = var_name.replace('.', '/')
                    trainlog.add_histogram(var_name, value.data.cpu().numpy())
                    trainlog.add_histogram(var_name+'/grad', value.grad.data.cpu().numpy())

            # TENSORBOARD IMAGE
            if (i+1) % 1000 == 0 and args.train_phase == 'pre_train_m_net':
                trainlog.add_image('fg_gt', vutils.make_grid(fg_gt, normalize=True, nrow=4))
                trainlog.add_image('unsure_gt', vutils.make_grid(unsure_gt, normalize=True, nrow=4))
                trainlog.add_image('alpha_p', vutils.make_grid(alpha_p, normalize=True, nrow=4))
                trainlog.add_image('alpha_r', vutils.make_grid(alpha_r, normalize=True, nrow=4))
                trainlog.add_image('alpha_gt', vutils.make_grid(alpha_gt, normalize=True, nrow=4))
            if (i+1) % 1000 == 0 and args.train_phase != 'pre_train_m_net':
                trainlog.add_trimap(trimap_pre)
                trainlog.add_trimap_gt(trimap_gt)
                trainlog.add_image('origin_image', vutils.make_grid(img, normalize=True, nrow=4))
            
            trainlog.step()

        print('Done iterating all training data')
        t1 = time.time()

        if epoch % args.save_epoch == 0:

            # speed = (t1 - t0) / 60 

            loss_ = loss_ / (i+1)
            L_alpha_ = L_alpha_ / (i+1)
            L_composition_ = L_composition_ / (i+1)
            L_cross_ = L_cross_ / (i+1)
            L2_bg_ = L2_bg_ / (i+1)
            loss_var = np.var(loss_array)
            IOU_t_bg_ = IOU_t_bg_ / (i+1)
            IOU_t_unsure_ = IOU_t_unsure_ / (i+1)
            IOU_t_fg_ = IOU_t_fg_ / (i+1)
            IOU_alpha_ = IOU_alpha_ / (i+1)
            trainlog.add_scalar('avg_loss', loss_, epoch)
            trainlog.add_scalar('avg_t_loss', L_cross_, epoch)
            trainlog.add_scalar('avg_t_L2_bg', L2_bg_, epoch)
            trainlog.add_scalar('avg_t_loss_var', loss_var, epoch)
            trainlog.add_scalar('avg_IOU_t_bg', IOU_t_bg_, epoch)
            trainlog.add_scalar('avg_IOU_t_unsure', IOU_t_unsure_, epoch)
            trainlog.add_scalar('avg_IOU_t_fg', IOU_t_fg_, epoch)
            trainlog.add_scalar('avg_L_alpha', L_alpha_, epoch)
            trainlog.add_scalar('avg_L_composition', L_composition_, epoch)
            for j in range(len(alpha_iou_threshold)):
                trainlog.add_scalar('avg_iou_alpha_'+str(alpha_iou_threshold[j]), IOU_alpha_[j], epoch)

            log = "[{} / {}] \tLr: {:.5f}\nloss: {:.5f}\tloss_p: {:.5f}\tloss_t: {:.5f}\tloss_var: {:.5f}\tIOU_t_bg: {:.5f}\tIOU_t_unsure: {:.5f}\tIOU_t_fg: {:.5f}\tIOU_alpha_mean: {:.5f}\t" \
                     .format(epoch, args.nEpochs, 
                            lr, 
                            loss_, 
                            L_alpha_+L_composition_, 
                            L_cross_,
                            loss_var,
                            IOU_t_bg_,
                            IOU_t_unsure_,
                            IOU_t_fg_,
                            IOU_alpha_.mean())
            print(log)
            trainlog.save_log(log)
            trainlog.save_model(model, epoch)


if __name__ == "__main__":
    main()
