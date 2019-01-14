import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


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
        #imgs[...,1] *= 0

        return imgs

class FFTNet(nn.Module):

    def __init__(self):
        super(FFTNet, self).__init__()
        self.fft1 = FFT_Layer(1, 40, 40)
        self.fft2 = FFT_Layer(40, 40, 40)
        self.fft3 = FFT_Layer(40, 40, 40)
        self.fft4 = FFT_Layer(40, 40, 40)
        self.fft5 = FFT_Layer(40, 1, 40)


    def forward(self, x):
        zeros = torch.zeros_like(x)
        x = torch.stack((x, zeros), dim=-1)
        x = F.relu(self.fft1(x))
        x = F.relu(self.fft2(x))
        x = F.relu(self.fft3(x))
        x = F.relu(self.fft4(x))
        x = self.fft5(x)
        return x[...,0]