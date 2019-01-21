import torch
import torch.nn as nn
import torch.nn.functional as F
from models import FFTNet, FFTNet2, CNN
from models import FFT_Layer, FFT_Per_Channel_Layer
import numpy as np
import os
import matplotlib.pyplot as plt
from math import sqrt
from csvd import clipSingularValues

def main():
    # load model
    net = CNN()
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids)
    model.load_state_dict(torch.load(os.path.join('logs', 'net.pth')))
    model.eval()

    
    with torch.no_grad():
        for layerNum, layer in enumerate(model.modules()):
            if isinstance(layer, FFT_Layer):
                W = layer.weight.data
                
                print(f'Layer {layerNum}:')
                print(f'Before norm:')
                W = clipSingularValues(W, 1)
                

    weights = None

    num_images = 0
    for layer in model.modules():
        if isinstance(layer, FFT_Per_Channel_Layer):
            num_images += 1
        if isinstance(layer, nn.Conv2d):
            num_images +=1

    # visualize first layer
    fig = plt.figure()
    n = 0
    for layer in model.modules():
        if isinstance(layer,FFT_Per_Channel_Layer):
            weights = layer.weight.data.cpu().numpy()
            weights = 1j*weights[..., 1] + weights[..., 0]
            weights = np.squeeze(weights)
            weights = np.fft.fftshift(weights)
            fft2 = abs(weights)
            print(fft2.max())
            a = fig.add_subplot(sqrt(num_images), np.ceil(num_images/sqrt(num_images)), n+1)
            plt.gray()
            plt.imshow(fft2)
            plt.axis('off')
            n += 1
        if isinstance(layer, nn.Conv2d):
            weights = layer.weight.data.cpu()
            weights = weights.view(3,3)
            weights = torch.transpose(weights, 0, 1)
            weights = F.pad(weights, (0, 80 - 1, 0, 80 - 1))
            zeros = torch.zeros_like(weights)
            weights = torch.stack((weights, zeros), dim=-1)
            weights = torch.fft(weights, 2).numpy()
            weights = 1j*weights[..., 1] + weights[..., 0]
            weights = np.squeeze(weights)
            weights = np.fft.fftshift(weights)
            fft2 = abs(weights)
            print(fft2.max())
            a = fig.add_subplot(sqrt(num_images), np.ceil(num_images/sqrt(num_images)), n+1)
            plt.gray()
            plt.imshow(fft2)
            plt.axis('off')
            n += 1
    plt.show()
    
    '''
    layer = net.fft1
    weights = layer.weight.data.cpu().numpy()
    
    if weights is None:
        exit(-1)
    # convert to complex type
    weights = 1j*weights[...,1] + weights[...,0]
    weights = np.squeeze(weights)
    weights = np.fft.fftshift(weights, axes=(1,2))

    fft2 = abs(weights)

    num_images = fft2.shape[0]

    
    for n in range(0, num_images):
        image = fft2[n]
        a = fig.add_subplot(sqrt(num_images), np.ceil(num_images/sqrt(num_images)), n+1)
        plt.gray()
        plt.imshow(np.log2(image+1))
        plt.axis('off')
    plt.show()
    '''


if __name__ == '__main__':
    main()