
import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.losses import CE_loss
import numpy as np

from torchvision.utils import save_image

def make_one_hot(labels, classes):
    one_hot = torch.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    return target.detach()


def DiceCELoss(output, target, ignore_index=255, smooth=1,ratio=0.75):
    if ignore_index not in range(target.min(), target.max()):
        if (target == ignore_index).sum() > 0:
            target[target == ignore_index] = target.min()
    target_onehot = make_one_hot(target.unsqueeze(dim=1), classes=output.size()[1])
    output_softmax = F.softmax(output, dim=1)
    output_flat = output_softmax.contiguous().view(-1)
    target_flat = target_onehot.contiguous().view(-1)
    intersection = (output_flat * target_flat).sum()
    loss = 1 - ((2. * intersection + smooth) /
                (output_flat.sum() + target_flat.sum() + smooth))

    CE = CE_loss(output, target.detach(), ignore_index=ignore_index,temperature=1.0)
    return ratio*loss + (1-ratio)*CE
        


class CriterionKD(nn.Module):
    '''
    knowledge distillation loss
    '''

    def __init__(self, upsample=False, temperature=0.1):
        super(CriterionKD, self).__init__()
        self.upsample = upsample
        self.temperature = temperature
        self.criterion_kd = torch.nn.KLDivLoss()

    def forward(self, pred, soft, mask):
        soft.detach()
        h, w = soft.size(2), soft.size(3)
        if self.upsample:
            scale_pred = F.upsample(input=pred, size=(h * 8, w * 8), mode='bilinear', align_corners=True)
            scale_soft = F.upsample(input=soft, size=(h * 8, w * 8), mode='bilinear', align_corners=True)
        else:
            scale_pred = pred
            scale_soft = soft

        b = soft.shape[0]
        loss = 0
        m_num = 0
        for ib in range(b):
            t_loss = self.criterion_kd(F.log_softmax(scale_pred[ib:ib+1,:,:,:] / self.temperature, dim=1), F.softmax(scale_soft[ib:ib+1,:,:,:]  / self.temperature, dim=1))
            loss += mask[ib]*t_loss
            m_num += mask[ib]
        if m_num>0:
            loss = loss/m_num

        return loss


def FocalLoss2d(input, target, gamma=1, pos_mask=None, ignore_index=255, keep_dim=False, weight_factor=None, num_classes=21, scale_factor=2):

    if pos_mask==None:
        pos_mask = torch.ones_like(target).cuda()
    # compute the negative likelyhood
    logpt = -F.cross_entropy(input, target, ignore_index=ignore_index, reduction='none')*pos_mask
    pt = torch.exp(logpt)

    # compute the loss
    if weight_factor == None:
        loss = -((1-pt)**gamma) * logpt
    else:
        loss = torch.zeros_like(target).cuda()
        for i_c in range(num_classes):
            c_mask = (target==i_c)
            gamma_c = scale_factor*(1 - weight_factor[i_c])
            #loss_c = -((gamma + gamma_c)/gamma)*((1-pt)**(gamma + gamma_c)) * logpt * c_mask
            loss_c = -((1-pt)**gamma_c) * logpt * c_mask

            loss = loss + loss_c

    if keep_dim:
        return loss
    else:    
        return loss.mean()


def gaussian_kernel(size=5, device=torch.device('cpu'), channels=3, sigma=1, dtype=torch.float):
    # Create Gaussian Kernel. In Numpy
    interval  = (2*sigma +1)/(size)
    ax = np.linspace(-(size - 1)/ 2., (size-1)/2., size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx)+ np.square(yy)) / np.square(sigma))
    kernel /= np.sum(kernel)
    # Change kernel to PyTorch. reshapes to (channels, 1, size, size)
    kernel_tensor = torch.as_tensor(kernel, dtype=dtype)
    kernel_tensor = kernel_tensor.repeat(channels, 1 , 1, 1)
    kernel_tensor.to(device)
    return kernel_tensor

def gaussian_conv2d(x, g_kernel, dtype=torch.float):
    #Assumes input of x is of shape: (minibatch, depth, height, width)
    #Infer depth automatically based on the shape
    channels = g_kernel.shape[0]
    padding = g_kernel.shape[-1] // 2 # Kernel size needs to be odd number
    if len(x.shape) != 4:
        raise IndexError('Expected input tensor to be of shape: (batch, depth, height, width) but got: ' + str(x.shape))
    y = F.conv2d(x, weight=g_kernel, stride=1, padding=padding, groups=channels)
    return y

def downsample(x):
    # Downsamples along  image (H,W). Takes every 2 pixels. output (H, W) = input (H/2, W/2)
    return x[:, :, ::2, ::2]

def create_laplacian_pyramid(x, kernel, levels):
    upsample = torch.nn.Upsample(scale_factor=2) # Default mode is nearest: [[1 2],[3 4]] -> [[1 1 2 2],[3 3 4 4]]
    pyramids = []
    current_x = x
    for level in range(0, levels):
        gauss_filtered_x = gaussian_conv2d(current_x, kernel)
        down = downsample(gauss_filtered_x)
        # Original Algorithm does indeed: L_i  = G_i  - expand(G_i+1), with L_i as current laplacian layer, and G_i as current gaussian filtered image, and G_i+1 the next.
        # Some implementations skip expand(G_i+1) and use gaussian_conv(G_i). We decided to use expand, as is the original algorithm
        laplacian = current_x - upsample(down) 
        laplacian = torch.abs(laplacian)
        laplacian = torch.mean(laplacian,1).unsqueeze(1)
        
        pyramids.append(laplacian)
        current_x = down 
    pyramids.append(current_x)
    return pyramids

class LaplacianPyramidLoss(torch.nn.Module):
    def __init__(self, max_levels=3, channels=3, kernel_size=5, sigma=1, dtype=torch.float, mode="CE"):
        super(LaplacianPyramidLoss, self).__init__()
        self.max_levels = max_levels
        self.kernel = gaussian_kernel(size=kernel_size, channels=channels, sigma=sigma, dtype=dtype)
        self.mode = mode
    
    def forward(self, x, target, device):

        self.kernel = self.kernel.to(device)

        input_pyramid = create_laplacian_pyramid(x, self.kernel, self.max_levels)
        target_pyramid = create_laplacian_pyramid(target, self.kernel, self.max_levels)

        feat = input_pyramid[0]
        #feat_sig = torch.abs(torch.sigmoid(feat)-0.5)
        feat_sum = 100*feat
        save_image(feat_sum[0:1,:,:].detach().cpu(), './temp/{}_Edge_s.png'.format(self.mode))

        feat = target_pyramid[0]
        feat_sum = 100*feat
        save_image(feat_sum[0:1,:,:].detach().cpu(), './temp/{}_Edge_w.png'.format(self.mode))

        return sum(torch.nn.functional.l1_loss(x,y) for x, y in zip(input_pyramid, target_pyramid))

class LaplacianPyramid(torch.nn.Module):
    def __init__(self, max_levels=3, channels=3, kernel_size=5, sigma=1, dtype=torch.float, mode="CE"):
        super(LaplacianPyramid, self).__init__()
        self.max_levels = max_levels
        self.kernel = gaussian_kernel(size=kernel_size, channels=channels, sigma=sigma, dtype=dtype)
        self.mode = mode
    
    def forward(self, x, device):
        
        b,c,h,w = x.shape
        self.kernel = self.kernel.to(device)

        input_pyramid = create_laplacian_pyramid(x, self.kernel, self.max_levels)

        total_lap_feat = 0
        for feat_p in input_pyramid[:-1]:
            feat_p = F.interpolate(feat_p, size=x.size()[2:], mode='bilinear', align_corners=True)
            total_lap_feat += feat_p
        
        total_lap_feat /= len(input_pyramid[:-1])
        #total_lap_feat = total_lap_feat / m.view(-1,1,1,1)
        #feat = input_pyramid[0]
        total_lap_feat_sig = torch.abs(torch.sigmoid(total_lap_feat)-0.5)
        total_lap_feat_sig = 100*total_lap_feat_sig

        total_lap_feat_sig = total_lap_feat_sig.clamp(max=1)
        
        save_image(total_lap_feat_sig[0:1,:,:].detach().cpu(), './temp/{}_Edge_w.png'.format(self.mode))

        return 2*(1-torch.sigmoid(0.5*total_lap_feat_sig)).view(b,h,w)
