import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from scipy import signal
import scipy
import time

#######################################################

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

class FFT_Layer(torch.nn.Module):

    def __init__(self, inputChannels, outputChannels, imgSize, cuda=False):
        super(FFT_Layer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(1, inputChannels, outputChannels, imgSize, imgSize, 2))
        self.inputChannels = inputChannels
        self.outputChannels = outputChannels
        self.imgSize = imgSize
        nn.init.xavier_normal_(self.weight)

    def forward(self, imgs):
        # assuming imgs is already complex
        imgs = torch.fft(imgs,2, normalized=True)

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

class FFT_Conv_Layer(torch.nn.Module):

    def __init__(self,filts,imgSize,filtSize=3,cuda=False):

        super(FFT_Conv_Layer, self).__init__()

        if cuda:
            self.filts = torch.from_numpy(filts).type(torch.float32).cuda()
        else:
            self.filts = torch.from_numpy(filts).type(torch.float32)

        self.imgSize = imgSize
        self.filtSize = filtSize

    def forward(self,imgs):

        # Pad and transform the image
        # Pad arg = (last dim pad left side, last dim pad right side, 2nd last dim left side, etc..)
        imgs = F.pad(imgs, (0, 0, 0, self.filtSize - 1, 0,self.filtSize - 1))

        imgs = torch.fft(imgs,2)

        # Extract the real and imaginary parts
        imgsR = imgs[:, :, :, :, :, 0]
        imgsIm = imgs[:, :, :, :, :, 1]

        # Pad and transform the filters
        filts = F.pad(self.filts, (0, 0, 0, self.imgSize - 1, 0, self.imgSize - 1))

        filts = torch.fft(filts, 2)

        # Extract the real and imaginary parts
        filtR = filts[:, :, :, :, :, 0]
        filtIm = filts[:, :, :, :, :, 1]

        # Do element wise complex multiplication
        imgsR, imgsIm = dft_conv(imgsR,imgsIm,filtR,filtIm)

        # Add dim to concat over
        imgsR = imgsR.unsqueeze(4)
        imgsIm = imgsIm.unsqueeze(4)

        # Concat the real and imaginary again then IFFT
        imgs = torch.cat((imgsR,imgsIm),-1)
        imgs = torch.ifft(imgs,2)

        # Filter and imgs were real so imag should be ~0
        imgs = imgs[:,:,1:-1,1:-1,0]

        return imgs


def initialTest():
    imgSize = 5
    inCs = 1
    outCs = 1

    testImg = np.array([[1.0,2,3,4,5],[4,5,6,7,8],[7,8,9,10,11],[11,12,13,14,15],[16,17,18,19,20]])
    testFilt = np.array([[1,2,5],[3.0,4,2],[7,8,9]])

    # Numpy test
    npConv = scipy.signal.convolve2d(testImg,testFilt,mode='same')

    # Make arrays into proper torch size (BS,InC,OutC,ImgH,ImgW,2 -> Real | Complex)
    img = prepForTorch_FromNumpy(testImg)
    filt = prepForTorch_FromNumpy(testFilt)

    img = torch.from_numpy(img).type(torch.float32)

    fftConv = FFT_Conv_Layer(filt,imgSize)
    fftOut = fftConv(img)

    # Only need real part for conv2d
    img = img[:,:,0,:,:,0]
    testFilt = testFilt[::-1, ::-1]   # flip kernel because pytorch's conv2d function actually does cross-correlation
    filt = prepForTorch_FromNumpy(testFilt)
    filt = filt[:,:,0,:,:,0]
    filt = torch.from_numpy(filt).type(torch.float32)

    # Padding pads on both sides symmetrically
    # Doesn't match scipy, this does auto correlation NOT convolution
    funOut = F.conv2d(img, filt,bias=None,padding=1,stride=(1,1))

    print(npConv)
    print(fftOut)
    print(funOut)

def largerTestCPU():

    filtSize = 3
    inCs = 3
    outCs = 32
    batchSize = 100
    imgSize = 16
    imagDim = 2

    imgs = torch.randn(batchSize,inCs,1,imgSize, imgSize,imagDim)
    filts = np.random.normal(size=(1,inCs,outCs,filtSize,filtSize,imagDim))

    fftConv = FFT_Conv_Layer(filts, imgSize)

    st = time.time()
    for i in range(50):
        fftOut = fftConv(imgs)
    et = time.time()
    print("FFT Conv CPU Time: {}".format(et - st))

    filts = torch.from_numpy(filts).type(torch.float32)
    filts = torch.transpose(filts,1,2)

    imgs = imgs.squeeze(2)
    filts = filts.squeeze(0)
    imgs = imgs[:,:,:,:,0]
    filts = filts[:,:,:,:,0]

    st = time.time()
    for i in range(50):
        funOut = F.conv2d(imgs, filts, bias=None, padding=1)
    et = time.time()
    print("Functional Conv CPU Time: {}".format(et - st))

def largerTestGPU():

    filtSize = 3
    inCs = 16
    outCs = 32
    batchSize = 64
    imgSize = 64
    imagDim = 2
    numIters = 50

    imgs = torch.randn(batchSize,inCs,1,imgSize, imgSize,imagDim).cuda()
    filts = np.random.normal(size=(1,inCs,outCs,filtSize,filtSize,imagDim))

    fftConv = FFT_Conv_Layer(filts, imgSize,cuda=True)

    # GPU warm up time
    for i in range(2):
        fftOut = fftConv(imgs)

    # Element wise
    torch.cuda.synchronize()
    st = time.time()
    for i in range(numIters):
        fftOut = fftConv(imgs)
    torch.cuda.synchronize()
    et = time.time()
    print("FFT Conv Ele GPU Time: {}".format(et - st))

    filts = torch.from_numpy(filts).type(torch.float32).cuda()
    filts = torch.transpose(filts,1,2)

    imgs = imgs.squeeze(2)
    filts = filts.squeeze(0)
    imgs = imgs[:,:,:,:,0]
    filts = filts[:,:,:,:,0]

    # Functional Conv
    torch.cuda.synchronize()
    st = time.time()
    for i in range(numIters):
        funOut = F.conv2d(imgs, filts, bias=None, padding=1)
    torch.cuda.synchronize()
    et = time.time()
    print("Functional Conv GPU Time: {}".format(et - st))
    

#initialTest()
largerTestCPU()
largerTestGPU()