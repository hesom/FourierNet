import torch
import cupy as cp

from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack

# for some reason the first cublas call always fails, so run a dummy cublas call on module load
try:
    a = cp.random.rand(1,1)
    b = cp.random.rand(1,10)
    cp.dot(a,b)
except Exception:
    pass
finally:
    del a
    del b

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

def clipSingularValues(input, clip_to):
    dx = to_dlpack(input)
    cx = cp.fromDlpack(dx)
    cx = 1j*cx[...,1] + cx[...,0]
    cx = cp.transpose(cx, (0,3,4,1,2))
    shape = cx.shape
    cx = cx.reshape(-1, shape[-2], shape[-1])
    maxSingularValue = 0
    for i in range(0, cx.shape[0]):
        u,s,v = cp.linalg.svd(cx[i], compute_uv=True, full_matrices=False)
        m = s.max()
        if m > maxSingularValue:
            maxSingularValue = m
        s = cp.minimum(s, clip_to)
        cx[i] = cp.dot(u*s, v)
    cx = cx.reshape(shape)
    cx = cp.transpose(cx,(0,3,4,1,2))
    print("Max singular value: ", maxSingularValue)
    return from_dlpack(cp.stack((cx.real, cx.imag), axis=-1).toDlpack())


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