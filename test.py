import cv2
import torch 
import argparse
import numpy as np
import os 
import torch.nn.functional as F
from model.M_Net import M_net
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

def load_model(args):
    print('Loading model from {}...'.format(args.model))
    if args.without_gpu:
        myModel = torch.load(args.model, map_location=lambda storage, loc: storage)
    else:
        myModel = torch.load(args.model)

    if args.subnet == 'm_net':
        myModel = myModel.m_net
    elif args.subnet == 't_net':
        myModel = myModel.t_net

    myModel.eval()
    myModel.to(device)
    
    return myModel

class Dataset(torch.utils.data.Dataset):
    def __init__(self, args, patchsize=256):
        self.args = args
        self.patchsize = patchsize
        if args.subnet == 'm_net':
            trimap_folder = os.path.join(args.data, 'trimap')
            image_folder = os.path.join(args.data, 'image')
            assert os.path.exists(trimap_folder), 'Please make sure trimap folder in testdata folder'
            assert os.path.exists(image_folder), 'Please make sure image folder in testdata folder'
            self.image = [os.path.join(args.data, 'image', imgfile) for imgfile in image_folder]
            for imgfile in self.image:
                trimapfile = imgfile.replace('image', 'trimap', 1).strip()[:-3]+'png'
                assert os.path.isfile(trimapfile), '{} is not a valid file'.format(trimapfile)
        else:
            self.image = [os.path.join(self.args.data, imgfile) for imgfile in os.listdir(self.args.data)]

    def center_crop(self, image, type='image'):
        interpolation = cv2.INTER_CUBIC if type == 'image' else cv2.INTER_NEAREST
        h, w, c = image.shape
        scale = patch_size*1.0 / min(h, w)
        resize_shape = (int(np.round(h*scale)), int(np.round(w*scale)))
        image = cv2.resize(image, resize_shape, interpolation=interpolation)
        h, w, c = image.shape
        y_blanking = int((h-self.patchsize)/2)
        x_blanking = int((w-self.patchsize)/2)
        image = image[y_blanking:y_blanking+self.patchsize, x_blanking: x_blanking+self.patchsize, ...]
        return image

    def __getitem__(self, index):
        image = cv2.imread(self.image[index], -1)
        if self.args.subnet == 'm_net':
            trimap = cv2.imread(self.image[index].replace('image', 'trimap', count=1).strip()[:-3]+'png', -1)
            trimap[trimap==128] = 1
            trimap[trimap==255] = 2
            assert len(np.unique(trimap)) == 3, 'Groundtruth trimap should only contain 3 values in pixel value'
            assert trimap.shape == image.shape, 'Trimap should has same shape as input image'
            trimap = self.center_crop(trimap, type='trimap')
            trimap = dataset.np2Tensor(trimap)
        else:
            trimap = None
        image = self.center_crop(image)

        image = (image.astype(np.float32) - (114., 121., 134.)) / 255.
        image = dataset.np2Tensor(image)
        return {'image': image, 'trimap': trimap, 'filename': self.image[index].split('/')[-1]}


def main(args):
    model = load_model(args)
    test_data = Dataset(args)
    for i, sample in enumerate(iter(test_data)):
        if i > 10:
            break
        if args.subnet == 'm_net':
            trimap = sample['trimap']
            print('sample trimap.shape: ', trimap.shape, sample['image'].shape)
            trimap_softmax = torch.zeros([trimap.shape[0], 3, trimap.shape[-2], trimap.shape[-1]], dtype=torch.float32)
            trimap_softmax.scatter_(1, trimap.long().data.cpu(), 1)
            m_net_input = torch.cat((sample['image'], trimap_softmax), 1)
            alpha_r = model.m_net(m_net_input)
            result = trimap_softmax[:, 2, ...] + trimap_softmax[:, 1, ...]*alpha_r
            cv2.imwrite(os.path.join(args.save_dir, sample['filename']), result)

if __name__ == '__main__':
    main(args)