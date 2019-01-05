import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from skimage.measure.simple_metrics import compare_psnr
from torch.nn.functional import normalize
from torch.nn.parameter import Parameter

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant(m.bias.data, 0.0)

def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])

def data_augmentation(image, mode):
    out = np.transpose(image, (1,2,0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2,0,1))

def spectralNorm(convKernel, padding=1, n_power_iterations=1):
    return math.sqrt(L_inf_Norm(convKernel)*L_1_Norm(convKernel))
    '''
    Computes the spectral norm of conv2d layer using the power method
    '''
    inputDepth = convKernel.size(1)
    x = torch.randn(1, inputDepth, 40, 40).cuda()
    for _ in range(n_power_iterations):
        x_ = F.conv2d(x, convKernel, padding=padding)
        x = F.conv_transpose2d(x_, convKernel, padding=padding)
        x_shape = x.size()
        x = F.normalize(x.view(x.numel()), dim=0).view(x_shape) # x to vector => L2 normalize => back to tensor
    
    Wx = F.conv2d(x, convKernel)
    Wx = Wx.view(Wx.numel())
    #Wx = F.normalize(Wx, dim=0)
    x = x.view(x.numel())
    #x = F.normalize(x, dim=0)
    sigma = Wx.norm()/ x.norm()

    return sigma.item()


def L_inf_Norm(convKernel):
    '''
    L_inf norm of conv2d layer
    '''
    W = convKernel.view(convKernel.size(0), -1)
    norm = torch.abs(W).sum(1).max().item()
    return norm


def L_1_Norm(convKernel):
    '''
    L_1 norm of conv2d layer
    '''
    W = convKernel.view(convKernel.size(1), -1)
    norm = torch.abs(W).sum(1).max().item()
    return norm

def SVD_Conv_Tensor_NP(filter, inp_size):
  # compute the singular values using FFT
  # first compute the transforms for each pair of input and output channels
  transform_coeff = np.fft.fft2(filter, inp_size, axes=[0, 1])

  # now, for each transform coefficient, compute the singular values of the
  # matrix obtained by selecting that coefficient for
  # input-channel/output-channel pairs
  return np.linalg.svd(transform_coeff, compute_uv=False)

def Clip_SpectralNorm(filter, inp_shape, clip_to):
    transform_coeff = np.fft.fft2(filter, inp_shape, axes=[0,1])

    U, D, V = np.linalg.svd(transform_coeff, compute_uv=True, full_matrices=False)
    D_clipped = np.minimum(D, clip_to)
    if filter.shape[2] > filter.shape[3]:
        clipped_transform_coeff = np.matmul(U, D_clipped[..., None]*V)
    else:
        clipped_transform_coeff = np.matmul(U * D_clipped[..., None], V)
    clipped_filter = np.fft.ifft2(clipped_transform_coeff, axes=[0,1]).real
    args = [range(d) for d in filter.shape]
    return clipped_filter[np.ix_(*args)]