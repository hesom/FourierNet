import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from dataset import prepare_data, Dataset
from utils import *
from csvd import clipSingularValues

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=64, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=3, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=40, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--mode", type=str, default="S", help='with known noise level (S) or blind training (B)')
parser.add_argument("--noiseL", type=float, default=25, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')
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
        #imgs[...,1] *= 0

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

def main():
    # Load dataset
    device = torch.device('cuda')
    print('Loading dataset ...\n')
    dataset_train = Dataset(train=True)
    dataset_val = Dataset(train=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    # Build model
    net = FFTNet()
    #net.apply(weights_init_kaiming)
    criterion = nn.MSELoss(reduction='sum')
    # Move to GPU
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()

    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            print(layer.weight.size())

    criterion.cuda()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)
    # training
    writer = SummaryWriter(opt.outf)
    step = 0
    noiseL_B=[0,55] # ingnored when opt.mode=='S'
    
    for epoch in range(opt.epochs):
        '''
        if epoch < opt.milestone:
            current_lr = opt.lr
        else:
            current_lr = opt.lr / 10.
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        '''
        scheduler.step()
        #print('learning rate %f' % current_lr)

        # train
        for i, data in enumerate(loader_train, 0):
            # training step
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            img_train = data
            if opt.mode == 'S':
                noise = torch.FloatTensor(img_train.size()).normal_(mean=0, std=opt.noiseL/255.)
            if opt.mode == 'B':
                noise = torch.zeros(img_train.size())
                stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise.size()[0])
                for n in range(noise.size()[0]):
                    sizeN = noise[0,:,:,:].size()
                    noise[n,:,:,:] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n]/255.)
            imgn_train = img_train + noise
            img_train, imgn_train = Variable(img_train.cuda()), Variable(imgn_train.cuda())
            noise = Variable(noise.cuda())
            out_train = model(imgn_train)
            loss = criterion(out_train, img_train) / (img_train.size()[0]*2)
            loss.backward()
            optimizer.step()

            lipschitzNormalisation = True
            L = 1   # Lipschitz constant per layer
            if lipschitzNormalisation:
                with torch.no_grad():
                    for layerNum, layer in enumerate(model.modules()):
                        if isinstance(layer, nn.Conv2d):
                            #norm = spectralNorm(layer.weight, layer.padding, n_power_iterations=10)
                            #norm = L_1_Norm(layer.weight)
                            #print(norm)

                            #W = layer.weight.data.view(layer.weight.data.size(1), -1)
                            #sums = torch.abs(W).sum(1)
                            #sums = torch.clamp(sums, min=1)
                            #W = W.transpose(0,1)
                            #W /= sums


                            W = layer.weight.data.cpu().numpy()
                            W = np.transpose(W, (2, 3, 0, 1))
                            W = Clip_SpectralNorm(W, [40, 40], L)
                            W = np.transpose(W, (2, 3, 0, 1))
                            layer.weight.data = torch.from_numpy(W).type(torch.FloatTensor).cuda()
                            
                            #W.div_(torch.Tensor([max(norm / L, 1)]).expand_as(W).cuda())
                        if isinstance(layer, FFT_Layer):
                            if False:
                                W = layer.weight.data
                                
                                print(f'Layer {layerNum}:')
                                print(f'Before norm:')
                                W = clipSingularValues(W, 1)

                            #norms = torch.norm(W, dim=5, keepdim=True)
                            #norms = torch.clamp(norms, min=1)
                            #print(norms.max().item())
                            #print(norms.min().item())
                            #W /= norms

                        if isinstance(layer, nn.BatchNorm2d):
                            var = layer.__getattr__("running_var")
                            gamma = layer.weight
                            norm = (gamma / var).abs().max()
                            gamma.data.div_(torch.Tensor([max(norm / L, 1)]).expand_as(gamma.data).cuda())
            # results
            model.eval()
            #out_train = torch.clamp(imgn_train-model(imgn_train), 0., 1.)
            out_train = torch.clamp(model(imgn_train), 0., 1.)
            psnr_train = batch_PSNR(out_train, img_train, 1.)
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                (epoch+1, i+1, len(loader_train), loss.item(), psnr_train))
            # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]
            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
            step += 1
        ## the end of each epoch
        model.eval()
        # validate
        with torch.no_grad():
                for layerNum, layer in enumerate(model.modules()):
                    if isinstance(layer, FFT_Layer):
                        W = layer.weight.data
                        
                        print(f'Layer {layerNum}:')
                        print(f'Before norm:')
                        W = clipSingularValues(W, 1)
        if epoch % 5 == 0:
            psnr_val = 0
            for k in range(len(dataset_val)):
                img_val = torch.unsqueeze(dataset_val[k], 0)
                noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=opt.val_noiseL/255.)
                imgn_val = img_val + noise
                img_val, imgn_val = Variable(img_val.cuda()), Variable(imgn_val.cuda())
                #out_val = torch.clamp(imgn_val-model(imgn_val), 0., 1.)
                out_val = torch.clamp(model(imgn_val), 0., 1.)
                psnr_val += batch_PSNR(out_val, img_val, 1.)
            psnr_val /= len(dataset_val)
            print("\n[epoch %d] PSNR_val: %.4f" % (epoch+1, psnr_val))
            writer.add_scalar('PSNR on validation data', psnr_val, epoch)
            # log the images
            #out_train = torch.clamp(imgn_train-model(imgn_train), 0., 1.)
            out_train = torch.clamp(model(imgn_train), 0., 1.)
            Img = utils.make_grid(img_train.data, nrow=8, normalize=True, scale_each=True)
            Imgn = utils.make_grid(imgn_train.data, nrow=8, normalize=True, scale_each=True)
            Irecon = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
            writer.add_image('clean image', Img, epoch)
            writer.add_image('noisy image', Imgn, epoch)
            writer.add_image('reconstructed image', Irecon, epoch)
        # save model
        torch.save(model.state_dict(), os.path.join(opt.outf, 'net.pth'))

if __name__ == "__main__":
    if opt.preprocess:
        if opt.mode == 'S':
            prepare_data(data_path='data', patch_size=40, stride=10, aug_times=1)
        if opt.mode == 'B':
            prepare_data(data_path='data', patch_size=50, stride=10, aug_times=2)
    main()
