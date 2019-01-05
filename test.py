import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from models import DnCNN, DnCNNSpectral
from utils import *
from scipy.linalg import toeplitz

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN_Test")
parser.add_argument("--num_of_layers", type=int, default=2, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
parser.add_argument("--test_data", type=str, default='Set68', help='test on Set12 or Set68')
parser.add_argument("--test_noiseL", type=float, default=25, help='noise level used on test set')
opt = parser.parse_args()

def dft_conv(imgR,imgIm,kernelR,kernelIm):

    # Fast complex multiplication
    ac = torch.mul(kernelR, imgR)
    bd = torch.mul(kernelIm, imgIm)
    
    ab_cd = torch.mul(torch.add(kernelR, kernelIm), torch.add(imgR, imgIm))
    # print(ab_cd.sum(1)[0,0,:,:])
    imgsR = ac - bd
    imgsIm = ab_cd - ac - bd

    # Sum over in channels
    imgsR = imgsR.sum(1)
    imgsIm = imgsIm.sum(1)

    return imgsR,imgsIm

def prepForTorch_FromNumpy(img):

    # Add batch dim, channels, and last dim to concat real and imag
    img = np.expand_dims(img, 0)
    img = np.vstack((img, np.imag(img)))
    img = np.transpose(img, (1, 2, 0))

    # Add dimensions
    img = np.expand_dims(img, 0)
    img = np.expand_dims(img, 0)
    img = np.expand_dims(img, 0)

    return img

class FFT_Layer(nn.Module):

    def __init__(self, inputChannels, outputChannels, imgSize):
        super(FFT_Layer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(1, inputChannels, outputChannels, imgSize, imgSize, 2))
        self.inputChannels = inputChannels
        self.outputChannels = outputChannels
        self.imgSize = imgSize
        nn.init.xavier_normal_(self.weight)

    def forward(self, imgs):
        # assuming imgs is already complex
        imgs = torch.fft(imgs,2, normalized=True)
        imgs = torch.unsqueeze(imgs, 2)
        # Extract the real and imaginary parts
        imgsR = imgs[:, :, :, :, :, 0]
        imgsIm = imgs[:, :, :, :, :, 1]

        # Extract the real and imaginary parts
        filtR = self.weight[:, :, :, :, :, 0]
        filtIm = self.weight[:, :, :, :, :, 1]

        # Do element wise complex multiplication
        imgsR, imgsIm = dft_conv(imgsR,imgsIm,filtR,filtIm)

        # Add dim to concat over
        imgsR = imgsR.unsqueeze(4)
        imgsIm = imgsIm.unsqueeze(4)

        # Concat the real and imaginary again then IFFT
        imgs = torch.cat((imgsR,imgsIm),-1)
        imgs = torch.ifft(imgs,2, normalized=True)

        return imgs

class FFTNet(nn.Module):

    def __init__(self):
        super(FFTNet, self).__init__()
        self.fft1 = FFT_Layer(1, 40, 40)
        self.fft2 = FFT_Layer(40, 40, 40)
        self.fft3 = FFT_Layer(40, 40, 40)
        self.fft4 = FFT_Layer(40, 40, 40)
        self.fft5 = FFT_Layer(40, 40, 40)
        self.fft6 = FFT_Layer(40, 1, 40)


    def forward(self, x):
        zeros = torch.zeros_like(x)
        x = torch.stack((x, zeros), dim=-1)
        x = F.relu(self.fft1(x))
        x = F.relu(self.fft2(x))
        x = F.relu(self.fft3(x))
        x = F.relu(self.fft4(x))
        x = F.relu(self.fft5(x))
        x = self.fft6(x)
        return x[...,0]

def normalize(data):
    return data/255.

def main():
    # Build model
    print('Loading model ...\n')
    net = FFTNet()
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net.pth')))
    model.eval()

    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            print(layer.weight.size())

    with torch.no_grad():
        L1_total = 1
        Linf_total = 1
        sigma_total = 1
        for i, layer in enumerate(model.modules()):
            if isinstance(layer, nn.Conv2d):
                Linf = L_inf_Norm(layer.weight)
                Linf_total *= Linf
                L1 = L_1_Norm(layer.weight)
                L1_total *= L1
                #sigma = spectralNorm(layer.weight, padding=layer.padding, n_power_iterations=5)
                W = layer.weight.data.cpu().numpy()
                W = np.transpose(W, axes=(2 ,3, 0, 1))
                sigma = SVD_Conv_Tensor_NP(W, [1000, 1000]).max()
                sigma_total *= sigma
                print("Layer {}: ".format(i), L1)
                #W = layer.weight.data
                #W.div_(torch.Tensor([max(Linf.item() / L, 1)]).expand_as(W).cuda())
            '''
            if isinstance(layer, nn.BatchNorm2d):
                var = layer.__getattr__("running_var")
                gamma = layer.weight
                norm = (gamma / var).abs().max()
                print("Layer {}: ".format(i), norm.item())
                L1_total *= norm
                Linf_total *= norm
                sigma_total *=norm
            '''
                

        print("L1 norm of network: {}".format(L1_total))
        print("L_inf norm of network: {}".format(Linf_total))
        print("Spectral norm of network: {}".format(sigma_total))

    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join('data', opt.test_data, '*.png'))
    files_source.sort()
    # process data
    psnr_test = 0
    for f in files_source:
        # image
        Img = cv2.imread(f)
        Img = normalize(np.float32(Img[:,:,0]))
        Img = np.expand_dims(Img, 0)
        Img = np.expand_dims(Img, 1)
        ISource = torch.Tensor(Img)
        # noise
        noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL/255.)
        # noisy image
        INoisy = ISource + noise
        ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())
        with torch.no_grad(): # this can save much memory
            #Out = torch.clamp(INoisy-model(INoisy), 0., 1.)
            Out = torch.clamp(model(INoisy), 0., 1.)
        ## if you are using older version of PyTorch, torch.no_grad() may not be supported
        # ISource, INoisy = Variable(ISource.cuda(),volatile=True), Variable(INoisy.cuda(),volatile=True)
        # Out = torch.clamp(INoisy-model(INoisy), 0., 1.)
        psnr = batch_PSNR(Out, ISource, 1.)
        psnr_test += psnr
        print("%s PSNR %f" % (f, psnr))
    psnr_test /= len(files_source)
    print("\nPSNR on test data %f" % psnr_test)

if __name__ == "__main__":
    main()
