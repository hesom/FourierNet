import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision import datasets, transforms

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

        # Filter and imgs were real so imag should be ~0
        #imgs = imgs[:,:,1:-1,1:-1,0]

        return imgs

class FFTNet(nn.Module):

    def __init__(self):
        super(FFTNet, self).__init__()
        self.fft1 = FFT_Layer(1, 40, 28)
        self.fft2 = FFT_Layer(40, 40, 28)
        self.fc1 = nn.Linear(40*28*28*2, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.fft1(x))
        x = F.relu(self.fft2(x))
        # flatten array
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        zeros = torch.zeros_like(data)
        data = torch.stack((data, zeros), dim=-1)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            zeros = torch.zeros_like(data)
            data = torch.stack((data, zeros), dim=-1)
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers':1, 'pin_memory':True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/MNIST', train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=64, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/MNIST', train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=64, shuffle=True, **kwargs)
    
    model = FFTNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, 100):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
    torch.save(model.state_dict(),'fftnet.pt')
