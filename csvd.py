import torch
import cupy as cp

from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack

def csvd(input, some=True, compute_uv=True):
    dx = to_dlpack(input)
    cx = cp.fromDlpack(dx)

    # pytorch complex arrays have seperate real and imaginary parts
    cx = 1j*cx[...,1] + cx[...,0]
    u,s,v = cp.linalg.svd(cx, full_matrices=some, compute_uv=compute_uv)

    u = from_dlpack(cp.stack((u.real, u.imag), axis=-1).toDlpack())
    s = from_dlpack(cp.stack((s.real, s.imag), axis=-1).toDlpack())
    v = from_dlpack(cp.stack((v.real, v.imag), axis=-1).toDlpack())

    return u,s,v

if __name__ == '__main__':
    x = torch.tensor([[8.79,  6.11, -9.15,  9.57, -3.49,  9.84],
                      [9.93,  6.91, -7.93,  1.64,  4.02,  0.15],
                      [9.83,  5.04,  4.86,  8.83,  9.80, -8.99],
                      [5.45, -0.27,  4.85,  0.74, 10.00, -6.02],
                      [3.16,  7.98,  3.01,  5.80,  4.27, -5.31]]).t().cuda()
    zeros = torch.zeros_like(x)
    x = torch.stack((x, zeros), dim=-1)
    u, s, v = csvd(x)
    print(u[...,0])
    print(u.shape)