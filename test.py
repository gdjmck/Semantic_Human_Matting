import cv2
import torch 
import argparse
import numpy as np
import os 
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader
import PIL.Image as Image
from model.M_Net import M_net
from model import network
from data import dataset


parser = argparse.ArgumentParser(description='human matting')
parser.add_argument('--model', default='./', help='preTrained model')
parser.add_argument('--data', default='./', help='test data folder or test date itself')
parser.add_argument('--size', type=int, default=256, help='input size')
parser.add_argument('--without_gpu', action='store_true', default=False, help='no use gpu')
parser.add_argument('--subnet', default='None', help='choose subset of model')
parser.add_argument('--save_dir', default='./test_result', help='directory to save test result')

args = parser.parse_args()

torch.set_grad_enabled(False)


#################################
#----------------
if args.without_gpu:
    print("use CPU !")
    device = torch.device('cpu')
else:
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        print("----------------------------------------------------------")
        print("|       use GPU !      ||   Available GPU number is {} !  |".format(n_gpu))
        print("----------------------------------------------------------")

        device = torch.device('cuda:0,1')

#################################
#---------------
def model_param_count(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

def load_model(args, model):
    print('Loading model from {}...'.format(args.model))
    ckpt_file = "{}/ckpt_lastest.pth".format(args.model)
    if args.without_gpu:
        ckpt = torch.load(ckpt_file, map_location=lambda storage, loc: storage)
    else:
        ckpt = torch.load(ckpt_file)
    state_dict = ckpt['state_dict'].copy()
    for key in ckpt['state_dict']:
        if key not in model.state_dict():
            print('missing key:\t', key)
            state_dict.pop(key)
    model.load_state_dict(state_dict, strict=False)
    if args.subnet == 't_net':
        model = model.t_net

    model.eval()
    model.to(device)
    
    return model

class Dataset(torch.utils.data.Dataset):
    def __init__(self, args, patchsize=256):
        self.args = args
        self.patchsize = patchsize
        if args.subnet == 'm_net':
            trimap_folder = os.path.join(args.data, 'trimap')
            image_folder = os.path.join(args.data, 'image')
            assert os.path.exists(trimap_folder), 'Please make sure trimap folder in testdata folder'
            assert os.path.exists(image_folder), 'Please make sure image folder in testdata folder'
            self.image = [os.path.join(args.data, 'image', imgfile) for imgfile in os.listdir(image_folder)]
            absent_file = []
            for i, imgfile in enumerate(self.image):
                trimapfile = imgfile.replace('image', 'trimap', 1).strip()[:-3]+'png'
                if not os.path.isfile(trimapfile):
                    absent_file.append(imgfile)
                    continue
                # make sure two files of approximately same shape
                with Image.open(trimapfile) as trimap:
                    with Image.open(imgfile) as image:
                        if np.fabs(image.size[0]/image.size[1] - trimap.size[0]/trimap.size[1]) > 1e-2:
                            absent_file.append(imgfile)
                if i - len(absent_file) > 10:
                    break
                #assert os.path.isfile(trimapfile), '{} is not a valid file'.format(trimapfile)
            for f in absent_file:
                self.image.remove(f)
        else:
            self.image = [os.path.join(self.args.data, 'image', imgfile) for imgfile in os.listdir(os.path.join(self.args.data, 'image'))]

    def center_crop(self, image, type='image'):
        interpolation = cv2.INTER_CUBIC if type == 'image' else cv2.INTER_NEAREST
        h, w = image.shape[0], image.shape[1]
        scale = self.patchsize*1.0 / min(h, w)
        resize_shape = (int(np.round(w*scale)), int(np.round(h*scale)))
        image = cv2.resize(image, resize_shape, interpolation=interpolation)
        h, w = image.shape[0], image.shape[1]
        y_blanking = int((h-self.patchsize)/2)
        x_blanking = int((w-self.patchsize)/2)
        image = image[y_blanking:y_blanking+self.patchsize, x_blanking: x_blanking+self.patchsize, ...]
        return image

    def __getitem__(self, index):
        image = cv2.imread(self.image[index], -1)
        if self.args.subnet == 'm_net':
            trimap = cv2.imread(self.image[index].replace('image', 'trimap', 1).strip()[:-3]+'png', -1)
            trimap[trimap==128] = 1
            trimap[trimap==255] = 2
            assert len(np.unique(trimap)) == 3, 'Groundtruth trimap should only contain 3 values in pixel value, {}'.format(self.image[index])
            # assert trimap.shape == image.shape, 'Trimap should has same shape as input image'
            if trimap.shape != image.shape:
                image = cv2.resize(image, (trimap.shape[1], trimap.shape[0]))
            trimap = self.center_crop(trimap, type='trimap')
            trimap = np.expand_dims(trimap, -1)
            trimap = dataset.np2Tensor(trimap)
        else:
            trimap = None
        image = self.center_crop(image)

        image = (image.astype(np.float32) - (114., 121., 134.)) / 255.
        image = dataset.np2Tensor(image)
        return {'image': image, 'trimap': trimap, 'filename': self.image[index].split('/')[-1]}

    def __len__(self):
        return len(self.image)


def main(args):
    model = network.net()
    model = load_model(args, model)
    test_data = Dataset(args)
    testloader = DataLoader(test_data, batch_size=1, drop_last=False, shuffle=True)
    for i, sample in enumerate(testloader):
        if i > 10:
            break
        postfix = sample['filename'].split('.')[-1]
        if args.subnet == 'm_net':
            trimap = sample['trimap']
            print('sample trimap.shape: ', trimap.shape, sample['image'].shape)
            trimap_softmax = torch.zeros([3, trimap.shape[-2], trimap.shape[-1]], dtype=torch.float32)
            trimap_softmax.scatter_(0, trimap.long().data.cpu(), 1)
            m_net_input = torch.cat((sample['image'], trimap_softmax), 0)
            m_net_input = m_net_input.unsqueeze(0)
            alpha_r = model.m_net(m_net_input)
            alpha_r = alpha_r.squeeze()
            print(trimap_softmax.shape, alpha_r.shape)
            result = trimap_softmax[2, ...] + trimap_softmax[1, ...]*alpha_r
            cv2.imwrite(os.path.join(args.save_dir, sample['filename']), result.data.cpu().numpy()*255)
        elif args.subnet == 'end_to_end':
            net_input = sample['image']
            net_input = net_input.unsqueeze(0)
            alpha = model(net_input)[1]
            alpha = alpha.squeeze()
            print('end_to_end alpha:', alpha.shape, type(alpha))
            cv2.imwrite(os.path.join(args.save_dir, sample['filename']).replace('.'+postfix, '_alpha.'+postfix), alpha.data.cpu().numpy()*255)
        elif args.subnet == 't_net':
            net_input = sample['image']
            net_input = net_input.unsqueeze(0)
            trimap = model(net_input)
            trimap_softmax = F.softmax(trimap, dim=1)
            print('trimap shape:', trimap_softmax.shape)
            cv2.imwrite(os.path.join(args.save_dir, sample['filename']).replace('.'+postfix, '_trimap.'+postfix), np.moveaxis(trimap_softmax.squeeze().data.cpu().numpy()*255, (0, 1, 2), (-1, 0, 1)))
        cv2.imwrite(os.path.join(args.save_dir, sample['filename']).replace('.'+postfix, '_img.'+postfix), np.moveaxis(sample['image'].data.cpu().numpy()*255, (0, 1, 2), (-1, 0, 1)) + (114., 121., 134.))
        

if __name__ == '__main__':
    main(args)