import math
import random

import cv2
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from utils.utils import bilinear_sampler, coords_grid
import torch.nn as nn
try:
    import alt_cuda_corr
except:
    pass

import torch


def random_num(size,end):
    range_ls=[i for i in range(end)]
    num_ls=[]
    for i in range(size):
        num=random.choice(range_ls)
        range_ls.remove(num)
        num_ls.append(num)
    return num_ls

def blur(feat,ksize,sigma):
    kernel = gaussian_kernel_2d(ksize,sigma)
    channels = feat.size()[1]
    kernel = torch.FloatTensor(kernel).expand(channels,channels, ksize, ksize)
    weight = nn.Parameter(data=kernel, requires_grad=False)
    weight = weight.cuda()
    return F.conv2d(feat, weight, padding=3)

def gaussian_kernel_2d(ksize, sigma):
    return cv2.getGaussianKernel(ksize, sigma) * cv2.getGaussianKernel(ksize, sigma).T

class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4,ksize=7,sigma=3,k=3):

        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # fmap1
        doG1_1 = blur(fmap1, ksize, sigma)
        doG1_2 = blur(fmap1, ksize, k * sigma)
        doG1_3 = blur(fmap1, ksize, k * k * sigma)

        # DoG fmap1
        diff1_corr12 = doG1_1 - doG1_2
        diff1_corr23 = doG1_2 - doG1_3

        # fmap2
        doG2_1 = blur(fmap2, ksize, sigma)
        doG2_2 = blur(fmap2, ksize, k * sigma)
        doG2_3 = blur(fmap2, ksize, k * k * sigma)

        # DoG fmap2
        diff2_corr12 = doG2_1 - doG2_2
        diff2_corr23 = doG2_2 - doG2_3


        corr1 = torch.sigmoid(torch.sigmoid(diff1_corr12 - diff1_corr23) * fmap1) *fmap1
        corr2 = torch.sigmoid(torch.sigmoid(diff2_corr12 - diff2_corr23) * fmap2) *fmap2

        #  corr1 ([4, 256, 46, 62])
        corr = CorrBlock.corr(corr1, corr2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch*h1*w1, dim, h2, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels-1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1, device=coords.device)
            dy = torch.linspace(-r, r, 2*r+1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)

            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl
            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht*wd)

        corr = torch.matmul(fmap1.transpose(1,2), fmap2) / math.sqrt(batch)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr  / torch.sqrt(torch.tensor(dim).float())