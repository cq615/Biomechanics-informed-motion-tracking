import torch
import numpy as np


def compute_gradient(x):
    # compute gradients of deformation fields x =[u, v]
    # x: deformation field with 2 channels as x- and y- dimensional displacements
    # du/dx = (u(x+1)-u(x-1)/2
    bsize, csize, height, width = x.size()
    xw = torch.cat((torch.zeros(bsize, csize, height, 1).cuda(), x, torch.zeros(bsize, csize, height, 1).cuda()), 3)
    d_x = (torch.index_select(xw, 3, torch.arange(2, width+2).cuda()) - torch.index_select(xw, 3, torch.arange(width).cuda()))/2  #[du/dx, dv/dx]
    xh = torch.cat((torch.zeros(bsize, csize, 1, width).cuda(), x, torch.zeros(bsize, csize, 1, width).cuda()), 2)
    d_y = (torch.index_select(xh, 2, torch.arange(2, height+2).cuda()) - torch.index_select(xh, 2, torch.arange(height).cuda()))/2  #[du/dy, dv/dy]
    d_xy = torch.cat((d_x, d_y), 1)
    d_xy = torch.index_select(d_xy, 1, torch.tensor([0, 2, 1, 3]).cuda()) #[du/dx, du/dy, dv/dx, dv/dy]
    return d_xy


def centre_crop(img, size, centre):
    img_new = np.zeros((img.shape[0],size,size))
    h1 = np.amin([size//2, centre[0]])
    h2 = np.amin([size//2, img.shape[1]-centre[0]])
    w1 = np.amin([size//2, centre[1]])
    w2 = np.amin([size//2, img.shape[2]-centre[1]])
    img_new[:,size//2-h1:size//2+h2,size//2-w1:size//2+w2] = img[:,centre[0]-h1:centre[0]+h2,centre[1]-w1:centre[1]+w2]
    return img_new
