import math, time
from itertools import chain
import torch
import torch.nn.functional as F
from torch import nn
from base import BaseModel
from utils.losses import *
from models.encoder import Encoder
from models.modeling.deeplab import DeepLab as DeepLab_v3p
from models.modeling.deeplab import DeepLab_ours as DeepLab_v3p_ours
from models.modeling.deeplab_SubCls import Deeplab_SubCls as Deeplab_SubCls
import numpy as np
import random

from torchvision.utils import save_image
from utils.criterion import FocalLoss2d, LaplacianPyramidLoss, LaplacianPyramid
from utils.helpers import colorize_mask
from utils import pallete
import torchvision.transforms.functional as TF
from PIL import Image
from utils.metrics import eval_metrics


class Test(BaseModel):
    def __init__(self, num_classes, conf, sup_loss=None, ignore_index=None, testing=False, pretrained=True, cont_vis=False):

        super(Test, self).__init__()
        assert int(conf['supervised']) + int(conf['semi']) == 1, 'one mode only'
        if conf['supervised']:
            self.mode = 'supervised'
        elif conf['semi']:
            self.mode = 'semi'
        else:
            raise ValueError('No such mode choice {}'.format(self.mode))
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.sup_loss_w = conf['supervised_w']
        self.sup_loss = sup_loss
        self.downsample = conf['downsample']
        self.backbone = conf['backbone']
        self.layers = conf['layers']
        self.out_dim = conf['out_dim']
        self.cont_vis = cont_vis
        assert self.layers in [50, 101]

        if self.backbone == 'deeplab_v3+':
            self.encoder = DeepLab_v3p(backbone='resnet{}'.format(self.layers))
            self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
            for m in self.classifier.modules():
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.SyncBatchNorm):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
        elif self.backbone == 'psp':
            self.encoder = Encoder(pretrained=pretrained)
            self.classifier = nn.Conv2d(self.out_dim, num_classes, kernel_size=1, stride=1)
        else:
            raise ValueError("No such backbone {}".format(self.backbone))

        if self.mode == 'semi':
            self.epoch_start_unsup = conf['epoch_start_unsup']
            self.step_save = conf['step_save']
            self.pos_thresh_value = conf['pos_thresh_value']
            self.stride = conf['stride']
        
        
        


    

    def forward(self, x_l=None, target_l=None, x_ul=None, target_ul=None, curr_iter=None, epoch=None, gpu=None,
                gt_l=None, ul1=None, br1=None, \
                ul2=None, br2=None, flip=None, visualize=False, use_classifier=False):
        if not self.training:
            if use_classifier:
                return self.classifier(x_l)

            with torch.no_grad():
                feat = self.encoder(x_l)
                enc = self.classifier(feat)
                output_l = F.interpolate(enc, size=x_l.size()[2:], mode='bilinear', align_corners=True)
            
            return output_l
                
        else:
            raise ValueError("No such mode {}".format(self.mode))

    def concat_all_gather(self, tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        with torch.no_grad():
            tensors_gather = [torch.ones_like(tensor)
                              for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

            output = torch.cat(tensors_gather, dim=0)
        return output

    def get_backbone_params(self):
        return self.encoder.get_backbone_params()

    def get_other_params(self):
        return chain(self.encoder.get_module_params(), self.classifier.parameters())

class Save_Features(BaseModel):
    def __init__(self, num_classes, conf, sup_loss=None, ignore_index=None, testing=False, pretrained=True):

        super(Save_Features, self).__init__()
        assert int(conf['supervised']) + int(conf['semi']) == 1, 'one mode only'
        if conf['supervised']:
            self.mode = 'supervised'
        elif conf['semi']:
            self.mode = 'semi'
        else:
            raise ValueError('No such mode choice {}'.format(self.mode))
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.sup_loss_w = conf['supervised_w']
        self.sup_loss = sup_loss
        self.downsample = conf['downsample']
        self.backbone = conf['backbone']
        self.layers = conf['layers']
        self.out_dim = conf['out_dim']
        assert self.layers in [50, 101]

        if self.backbone == 'deeplab_v3+':
            self.encoder = DeepLab_v3p(backbone='resnet{}'.format(self.layers))
            self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
            for m in self.classifier.modules():
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.SyncBatchNorm):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
        elif self.backbone == 'psp':
            self.encoder = Encoder(pretrained=pretrained)
            self.classifier = nn.Conv2d(self.out_dim, num_classes, kernel_size=1, stride=1)
        else:
            raise ValueError("No such backbone {}".format(self.backbone))

        if self.mode == 'semi':
            self.epoch_start_unsup = conf['epoch_start_unsup']
            self.step_save = conf['step_save']
            self.pos_thresh_value = conf['pos_thresh_value']
            self.stride = conf['stride']

    def forward(self, x_l=None, target_l=None, x_ul=None, target_ul=None, curr_iter=None, epoch=None, gpu=None,
                gt_l=None, ul1=None, br1=None, \
                ul2=None, br2=None, flip=None):
        if not self.training:
            with torch.no_grad():
                feat = self.encoder(x_l)
                enc = self.classifier(feat)
                output_l = F.interpolate(enc, size=x_l.size()[2:], mode='bilinear', align_corners=True)
            # return output_l
            return feat, output_l
        else:
            raise ValueError("No such mode {}".format(self.mode))

    def concat_all_gather(self, tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        with torch.no_grad():
            tensors_gather = [torch.ones_like(tensor)
                              for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

            output = torch.cat(tensors_gather, dim=0)
        return output

    def get_backbone_params(self):
        return self.encoder.get_backbone_params()

    def get_other_params(self):
        return chain(self.encoder.get_module_params(), self.classifier.parameters())

class Baseline(BaseModel):
    def __init__(self, num_classes, conf, sup_loss=None, ignore_index=None, testing=False, pretrained=True):

        super(Baseline, self).__init__()
        assert int(conf['supervised']) + int(conf['semi']) == 1, 'one mode only'
        if conf['supervised']:
            self.mode = 'supervised'
        elif conf['semi']:
            self.mode = 'semi'
        else:
            raise ValueError('No such mode choice {}'.format(self.mode))

        self.ignore_index = ignore_index

        self.num_classes = num_classes
        self.sup_loss_w = conf['supervised_w']
        self.sup_loss = sup_loss
        self.downsample = conf['downsample']
        self.backbone = conf['backbone']
        self.layers = conf['layers']
        self.out_dim = conf['out_dim']

        assert self.layers in [50, 101]

        if self.backbone == 'deeplab_v3+':
            self.encoder = DeepLab_v3p(backbone='resnet{}'.format(self.layers))
            self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
            for m in self.classifier.modules():
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.SyncBatchNorm):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
        elif self.backbone == 'psp':
            self.encoder = Encoder(pretrained=pretrained)
            self.classifier = nn.Conv2d(self.out_dim, num_classes, kernel_size=1, stride=1)
        else:
            raise ValueError("No such backbone {}".format(self.backbone))

    def forward(self, x_l=None, target_l=None, x_ul=None, target_ul=None, curr_iter=None, epoch=None, gpu=None,
                gt_l=None, ul1=None, br1=None, \
                ul2=None, br2=None, flip=None, visualize=False):
        if not self.training:
            enc = self.encoder(x_l)
            enc_f = enc
            enc = self.classifier(enc)
            if visualize:
                return F.interpolate(enc, size=x_l.size()[2:], mode='bilinear', align_corners=True), enc_f
            else:
                return F.interpolate(enc, size=x_l.size()[2:], mode='bilinear', align_corners=True)

        if self.mode == 'supervised':
            feat = self.encoder(x_l)
            enc = self.classifier(feat)
            output_l = F.interpolate(enc, size=x_l.size()[2:], mode='bilinear', align_corners=True)

            loss_sup = self.sup_loss(output_l, target_l, ignore_index=self.ignore_index,
                                     temperature=1.0) * self.sup_loss_w

            curr_losses = {'Ls': loss_sup}
            outputs = {'sup_pred': output_l}
            total_loss = loss_sup

            return total_loss, curr_losses, outputs
        else:
            raise ValueError("No such mode {}".format(self.mode))

    def concat_all_gather(self, tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        with torch.no_grad():
            tensors_gather = [torch.ones_like(tensor)
                              for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

            output = torch.cat(tensors_gather, dim=0)
        return output

    def get_backbone_params(self):
        return self.encoder.get_backbone_params()

    def get_other_params(self):
        return chain(self.encoder.get_module_params(), self.classifier.parameters())



class USRN(BaseModel):
    def __init__(self, num_classes, conf, sup_loss=None, ignore_index=None, testing=False, pretrained=True):
        super(USRN, self).__init__()
        assert int(conf['supervised']) + int(conf['semi']) == 1, 'one mode only'
        if conf['supervised']:
            self.mode = 'supervised'
        elif conf['semi']:
            self.mode = 'semi'
        else:
            raise ValueError('No such mode choice {}'.format(self.mode))
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.sup_loss_w = conf['supervised_w']
        self.sup_loss = sup_loss
        self.downsample = conf['downsample']
        self.backbone = conf['backbone']
        self.layers = conf['layers']
        self.out_dim = conf['out_dim']
        assert self.layers in [50, 101]

        self.loss_weight_unsup = conf['loss_weight_unsup']
        ### VOC Dataset
        if conf['n_labeled_examples'] == 662:
            self.split_list = [132, 2, 1, 1, 1, 2, 3, 4, 7, 2, 1, 2, 6, 2, 2, 15, 1, 1, 2, 2, 1]
        elif conf['n_labeled_examples'] == 331:
            self.split_list = [121, 2, 1, 1, 1, 1, 3, 3, 6, 3, 1, 2, 6, 2, 2, 15, 1, 1, 2, 2, 1]
        elif conf['n_labeled_examples'] == 165:
            self.split_list = [136, 2, 2, 1, 1, 1, 2, 4, 8, 3, 1, 2, 7, 2, 2, 18, 1, 1, 1, 3, 3]
        ### Cityscapes Dataset
        elif conf['n_labeled_examples'] == 372:
            self.split_list = [42, 7, 26, 1, 2, 2, 1, 1, 19, 2, 5, 2, 1, 8, 1, 1, 1, 1, 1]
        elif conf['n_labeled_examples'] == 186:
            self.split_list = [45, 7, 28, 1, 2, 2, 1, 1, 20, 2, 5, 2, 1, 8, 1, 1, 1, 1, 1]
        elif conf['n_labeled_examples'] == 93:
            self.split_list = [38, 6, 22, 1, 2, 2, 1, 1, 17, 2, 5, 1, 1, 7, 1, 1, 1, 1, 1]
        self.num_classes_subcls = sum(self.split_list)

        if self.backbone == 'deeplab_v3+':
            self.encoder = Deeplab_SubCls(backbone='resnet{}'.format(self.layers))
            self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
            for m in self.classifier.modules():
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.SyncBatchNorm):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            self.classifier_SubCls = nn.Sequential(nn.Dropout(0.1), nn.Conv2d(256, self.num_classes_subcls, kernel_size=1, stride=1))
            for m in self.classifier_SubCls.modules():
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.SyncBatchNorm):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
        elif self.backbone == 'psp':
            self.encoder = Encoder(pretrained=pretrained)
            self.classifier = nn.Conv2d(self.out_dim, num_classes, kernel_size=1, stride=1)
        else:
            raise ValueError("No such backbone {}".format(self.backbone))
        if self.mode == 'semi':
            self.epoch_start_unsup = conf['epoch_start_unsup']
            self.step_save = conf['step_save']
            self.pos_thresh_value = conf['pos_thresh_value']
            self.stride = conf['stride']
    
    def generate_cutout_mask(self, img_size, ratio=2):
        cutout_area = img_size[0] * img_size[1] / ratio

        w = np.random.randint(img_size[1] / ratio + 1, img_size[1])
        h = np.round(cutout_area / w)

        x_start = np.random.randint(0, img_size[1] - w + 1)
        y_start = np.random.randint(0, img_size[0] - h + 1)

        x_end = int(x_start + w)
        y_end = int(y_start + h)

        mask = torch.ones(img_size)
        mask[y_start:y_end, x_start:x_end] = 0
        return mask.long()

    def generate_unsup_data(self, data, target, target_F, logits, logits_F, mode="cutout", target_ul=None):
        batch_size, _, im_h, im_w = data.shape
        _, t_h, t_w = target.shape
        device = data.device
        target = F.interpolate(target.unsqueeze(1).float(), size=[im_h, im_w], mode='nearest')
        logits = F.interpolate(logits.unsqueeze(1), size=[im_h, im_w], mode='bilinear', align_corners=True)

        target_F = F.interpolate(target_F.unsqueeze(1).float(), size=[im_h, im_w], mode='nearest')
        logits_F = F.interpolate(logits_F.unsqueeze(1), size=[im_h, im_w], mode='bilinear', align_corners=True)

        target = target.view(batch_size, im_h, im_w).long()
        logits = logits.view(batch_size, im_h, im_w)
        target_F = target_F.view(batch_size, im_h, im_w).long()
        logits_F = logits_F.view(batch_size, im_h, im_w)

        if target_ul != None:
            target_ul = F.interpolate(target_ul.unsqueeze(1).float(), size=[im_h, im_w], mode='nearest')
            target_ul = target_ul.view(batch_size, im_h, im_w).long()



        new_data = []
        new_target = []
        new_logits = []
        new_target_F = []
        new_logits_F = []
        new_target_ul = []
        vis_data = data.clone().detach()
        for i in range(batch_size):
            mix_mask = self.generate_cutout_mask([im_h, im_w],cos_mask=None)
                
            
            mix_mask = mix_mask.to(device)
            
            
            if torch.rand(1) > 0.75:
                new_data.append(
                    (
                        data[i] * mix_mask + data[(i + 1) % batch_size] * (1 - mix_mask)
                    ).unsqueeze(0)
                )
                new_target.append(
                    (
                        target[i] * mix_mask + target[(i + 1) % batch_size] * (1 - mix_mask)
                    ).unsqueeze(0)
                )
                new_logits.append(
                    (
                        logits[i] * mix_mask + logits[(i + 1) % batch_size] * (1 - mix_mask)
                    ).unsqueeze(0)
                )
                new_target_F.append(
                    (
                        target_F[i] * mix_mask + target_F[(i + 1) % batch_size] * (1 - mix_mask)
                    ).unsqueeze(0)
                )
                new_logits_F.append(
                    (
                        logits_F[i] * mix_mask + logits_F[(i + 1) % batch_size] * (1 - mix_mask)
                    ).unsqueeze(0)
                )

                if target_ul != None:
                    new_target_ul.append(
                    (
                        target_ul[i] * mix_mask + target_ul[(i + 1) % batch_size] * (1 - mix_mask)
                    ).unsqueeze(0)
                )

            else:
                new_data.append(data[i].unsqueeze(0))
                new_target.append(target[i].unsqueeze(0))
                new_logits.append(logits[i].unsqueeze(0))
                new_target_F.append(target_F[i].unsqueeze(0))
                new_logits_F.append(logits_F[i].unsqueeze(0))
                new_target_ul.append(target_ul[i].unsqueeze(0))

        new_data, new_target, new_logits, new_target_F, new_logits_F = (
            torch.cat(new_data),
            torch.cat(new_target),
            torch.cat(new_logits),
            torch.cat(new_target_F),
            torch.cat(new_logits_F),
        )

        

        new_target = F.interpolate(new_target.unsqueeze(1).float(), size=[t_h, t_w], mode='nearest')
        new_logits = F.interpolate(new_logits.unsqueeze(1), size=[t_h, t_w], mode='bilinear', align_corners=True)

        new_target_F = F.interpolate(new_target_F.unsqueeze(1).float(), size=[t_h, t_w], mode='nearest')
        new_logits_F = F.interpolate(new_logits_F.unsqueeze(1), size=[t_h, t_w], mode='bilinear', align_corners=True)

        new_target = new_target.view(batch_size, t_h, t_w)
        new_logits = new_logits.view(batch_size, t_h, t_w)
        new_target_F = new_target_F.view(batch_size, t_h, t_w)
        new_logits_F = new_logits_F.view(batch_size, t_h, t_w)

        if target_ul != None:
            new_target_ul = torch.cat(new_target_ul)
            new_target_ul = F.interpolate(new_target_ul.unsqueeze(1).float(), size=[t_h, t_w], mode='nearest')
            new_target_ul = new_target_ul.view(batch_size, t_h, t_w)
            return new_data, new_target.long(), new_target_F.long(), new_logits, new_logits_F, new_target_ul

        return new_data, new_target.long(), new_target_F.long(), new_logits, new_logits_F

    def forward(self, x_l=None, target_l=None, target_l_subcls=None, x_ul=None, target_ul=None,
                curr_iter=None, epoch=None, gpu=None, gt_l=None, ul1=None, br1=None, ul2=None, br2=None, flip=None):
        if not self.training:
            enc, _ = self.encoder(x_l)
            enc = self.classifier(enc)
            return F.interpolate(enc, size=x_l.size()[2:], mode='bilinear', align_corners=True)

        if self.mode == 'supervised':
            feat, feat_SubCls = self.encoder(x_l)
            enc = self.classifier(feat)
            output_l = F.interpolate(enc, size=x_l.size()[2:], mode='bilinear', align_corners=True)
            loss_sup = self.sup_loss(output_l, target_l, ignore_index=self.ignore_index,
                                     temperature=1.0) * self.sup_loss_w
            curr_losses = {'Ls': loss_sup}
            outputs = {'sup_pred': output_l}
            total_loss = loss_sup

            enc_SubCls = self.classifier_SubCls(feat_SubCls)
            output_l_SubCls = F.interpolate(enc_SubCls, size=x_l.size()[2:], mode='bilinear', align_corners=True)
            loss_sup_SubCls = self.sup_loss(output_l_SubCls, target_l_subcls, ignore_index=self.ignore_index, temperature=1.0) * self.sup_loss_w
            curr_losses['Ls_sub'] = loss_sup_SubCls
            total_loss = total_loss + loss_sup_SubCls * self.loss_weight_subcls

            return total_loss, curr_losses, outputs

        elif self.mode == 'semi':
            # feat = self.encoder(x_l)
            feat, feat_SubCls = self.encoder(x_l)
            enc = self.classifier(feat)
            output_l = F.interpolate(enc, size=x_l.size()[2:], mode='bilinear', align_corners=True)
            loss_sup = self.sup_loss(output_l, target_l, ignore_index=self.ignore_index,
                                     temperature=1.0) * self.sup_loss_w
            curr_losses = {'Ls': loss_sup}
            outputs = {'sup_pred': output_l}
            total_loss = loss_sup

            enc_SubCls = self.classifier_SubCls(feat_SubCls)
            output_l_SubCls = F.interpolate(enc_SubCls, size=x_l.size()[2:], mode='bilinear', align_corners=True)
            loss_sup_SubCls = self.sup_loss(output_l_SubCls, target_l_subcls, ignore_index=self.ignore_index, temperature=1.0) * self.sup_loss_w
            curr_losses['Ls_sub'] = loss_sup_SubCls
            total_loss = total_loss + loss_sup_SubCls * self.loss_weight_subcls

            if epoch < self.epoch_start_unsup:
                return total_loss, curr_losses, outputs
            
            # x_ul: [batch_size, 2, 3, H, W]
            x_w = x_ul[:, 0, :, :, :]  # Weak Aug
            x_s = x_ul[:, 1, :, :, :]  # Strong Aug
            
            feat_w, feat_SubCls_w = self.encoder(x_w)
            if self.downsample:
                feat_w = F.avg_pool2d(feat_w, kernel_size=2, stride=2)
            logits_w = self.classifier(feat_w)
            seg_w = F.softmax(logits_w, 1)
            seg_w_ent = torch.sum(self.prob_2_entropy(seg_w.detach()),1)

            if self.downsample:
                feat_SubCls_w = F.avg_pool2d(feat_SubCls_w, kernel_size=2, stride=2)
            logits_SubCls_w = self.classifier_SubCls(feat_SubCls_w)
            seg_w_SubCls = F.softmax(logits_SubCls_w, 1)
            pseudo_logits_SubCls_w = seg_w_SubCls.max(1)[0].detach()
            pseudo_label_SubCls_w = seg_w_SubCls.max(1)[1].detach()

            seg_w_SubCls_ent = torch.sum(self.prob_2_entropy(seg_w_SubCls.detach()),1)
            SubCls_reg_label_one_hot_ent_reg = SubCls_reg_label_one_hot.clone()
            SubCls_reg_label_one_hot_ent_reg[(seg_w_SubCls_ent>seg_w_ent).unsqueeze(1).repeat(1,seg_w.shape[1],1,1)] = 1
            seg_w_reg = seg_w * SubCls_reg_label_one_hot_ent_reg
            pseudo_logits_w_reg = seg_w_reg.max(1)[0].detach()
            pseudo_label_w_reg = seg_w_reg.max(1)[1].detach()

            x_s_cutmix, pseudo_label_w_reg, pseudo_label_SubCls_w, pseudo_logits_w_reg, pseudo_logits_SubCls_w, target_ul_aug = self.generate_unsup_data(x_s, pseudo_label_w_reg, pseudo_label_SubCls_w, pseudo_logits_w_reg, pseudo_logits_SubCls_w, mode="cutmix", target_ul=target_ul)
            save_image(x_s_cutmix[0,:,:,:].cpu().detach(), './temp/cut_mix_img.png')
            #save_image(1-((cos_masks[0,:,:])).cpu().detach(), './temp/cos_vis.png')
            feat_s,feat_SubCls_s = self.encoder(x_s_cutmix)
            if self.downsample:
                feat_s = F.avg_pool2d(feat_s, kernel_size=2, stride=2)
            logits_s = self.classifier(feat_s)

            if self.downsample:
                feat_SubCls_s = F.avg_pool2d(feat_SubCls_s, kernel_size=2, stride=2)
            logits_SubCls_s = self.classifier_SubCls(feat_SubCls_s)
            
            pos_mask_SubCls = pseudo_logits_SubCls_w > self.pos_thresh_value
            loss_unsup_SubCls = (F.cross_entropy(logits_SubCls_s, pseudo_label_SubCls_w, reduction='none') * pos_mask_SubCls).mean()
            curr_losses['Lu_sub'] = loss_unsup_SubCls
            total_loss = total_loss + loss_unsup_SubCls * self.loss_weight_unsup * self.loss_weight_subcls

            SubCls_reg_label = self.SubCls_to_ParentCls(pseudo_label_SubCls_w)
            
            SubCls_reg_label_one_hot = F.one_hot(SubCls_reg_label, num_classes=self.num_classes).permute(0,3,1,2)
            # seg_w_reg = seg_w * SubCls_reg_label_one_hot
            
            
            pos_mask_reg = pseudo_logits_w_reg > self.pos_thresh_value
            loss_unsup_reg = (F.cross_entropy(logits_s, pseudo_label_w_reg, reduction='none') * pos_mask_reg).mean()
            curr_losses['Lu_reg'] = loss_unsup_reg
            total_loss = total_loss + loss_unsup_reg * self.loss_weight_unsup
            return total_loss, curr_losses, outputs

        else:
            raise ValueError("No such mode {}".format(self.mode))

    def prob_2_entropy(self, prob):
        n, c, h, w = prob.size()
        return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)


    def SubCls_to_ParentCls(self, label_SubCls):
        label_SubCls_to_ParentCls = label_SubCls.clone()
        subclasses = np.cumsum(np.asarray(self.split_list))
        subclasses = np.insert(subclasses, 0, 0)
        parentclasses = np.uint8(np.linspace(1,len(self.split_list),len(self.split_list))-1)
        for subcls_lower, subcls_upper, parcls in zip(np.flip(subclasses[:-1]), np.flip(subclasses[1:]), np.flip(parentclasses)):
            label_SubCls_to_ParentCls[(label_SubCls>=subcls_lower)*(label_SubCls<subcls_upper)] = parcls
        return label_SubCls_to_ParentCls.cuda().long()

    def concat_all_gather(self, tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        with torch.no_grad():
            tensors_gather = [torch.ones_like(tensor)
                              for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

            output = torch.cat(tensors_gather, dim=0)
        return output

    def get_backbone_params(self):
        return self.encoder.get_backbone_params()

    def get_other_params(self):
        return chain(self.encoder.get_module_params(), self.classifier.parameters(),
                     self.classifier_SubCls.parameters())

class Ours(BaseModel):
    def __init__(self, num_classes, conf, sup_loss=None, ignore_index=None, testing=False, pretrained=True):
        super(Ours, self).__init__()
        assert int(conf['supervised']) + int(conf['semi']) == 1, 'one mode only'
        if conf['supervised']:
            self.mode = 'supervised'
        elif conf['semi']:
            self.mode = 'semi'
        else:
            raise ValueError('No such mode choice {}'.format(self.mode))
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.sup_loss_w = conf['supervised_w']
        self.sup_loss = sup_loss
        self.downsample = conf['downsample']
        self.backbone = conf['backbone']
        self.layers = conf['layers']
        self.out_dim = conf['out_dim']
        assert self.layers in [50, 101]

        self.loss_weight_unsup = conf['loss_weight_unsup']
        ### VOC Dataset
        if conf['n_labeled_examples'] == 662:
            self.split_list = [132, 2, 1, 1, 1, 2, 3, 4, 7, 2, 1, 2, 6, 2, 2, 15, 1, 1, 2, 2, 1]
        elif conf['n_labeled_examples'] == 331:
            self.split_list = [121, 2, 1, 1, 1, 1, 3, 3, 6, 3, 1, 2, 6, 2, 2, 15, 1, 1, 2, 2, 1]
        elif conf['n_labeled_examples'] == 165:
            self.split_list = [136, 2, 2, 1, 1, 1, 2, 4, 8, 3, 1, 2, 7, 2, 2, 18, 1, 1, 1, 3, 3]
        ### Cityscapes Dataset
        elif conf['n_labeled_examples'] == 372:
            self.split_list = [42, 7, 26, 1, 2, 2, 1, 1, 19, 2, 5, 2, 1, 8, 1, 1, 1, 1, 1]
        elif conf['n_labeled_examples'] == 186:
            self.split_list = [45, 7, 28, 1, 2, 2, 1, 1, 20, 2, 5, 2, 1, 8, 1, 1, 1, 1, 1]
        elif conf['n_labeled_examples'] == 93:
            self.split_list = [38, 6, 22, 1, 2, 2, 1, 1, 17, 2, 5, 1, 1, 7, 1, 1, 1, 1, 1]
        self.num_classes_subcls = sum(self.split_list)

        if self.backbone == 'deeplab_v3+':
            self.encoder = DeepLab_v3p_ours(backbone='resnet{}'.format(self.layers))
            #self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
            self.classifier_CE = nn.Sequential(nn.Dropout(0.1), nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
            self.classifier_Focal = nn.Sequential(nn.Dropout(0.1), nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
            
            """
            for m in self.classifier.modules():
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.SyncBatchNorm):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            """
            for m in self.classifier_CE.modules():
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.SyncBatchNorm):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            
            for m in self.classifier_Focal.modules():
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.SyncBatchNorm):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            """
            self.classifier_SubCls = nn.Sequential(nn.Dropout(0.1), nn.Conv2d(256, self.num_classes_subcls, kernel_size=1, stride=1))
            for m in self.classifier_SubCls.modules():
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.SyncBatchNorm):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            """
        elif self.backbone == 'psp':
            self.encoder = Encoder(pretrained=pretrained)
            self.classifier = nn.Conv2d(self.out_dim, num_classes, kernel_size=1, stride=1)
        else:
            raise ValueError("No such backbone {}".format(self.backbone))
        if self.mode == 'semi':
            self.epoch_start_unsup = conf['epoch_start_unsup']
            self.step_save = conf['step_save']
            self.pos_thresh_value = conf['pos_thresh_value']
            self.stride = conf['stride']

        self.memory_bank = []
        self.Q_size = []
        for c in range(self.num_classes):
            self.memory_bank.append(torch.zeros(0,256).cuda())
            self.Q_size.append(0)
        
        self.eps = 1e-5
        self.proj_final_dim = 256
        
        self.criterion_Focal = FocalLoss2d

        self.palette = pallete.get_voc_pallete(self.num_classes)
        self.use_cutmix = conf['use_cutmix']
        self.ent_weight = torch.zeros(self.num_classes).cuda()

    def copy_paste(self, img, label, paste_img, paste_label):
        b,_,_,h,w = img.shape
        paste_label_flat = torch.flatten(paste_label,start_dim=1)
        paste_label_cond_idx = ((paste_label_flat>0) * (paste_label_flat<255)).int()

        paste_label_idx = torch.nonzero(paste_label_cond_idx)
        paste_label_cond_idx_flat = torch.flatten(paste_label_cond_idx,start_dim=1)
        paste_label_cond_num = torch.sum(paste_label_cond_idx_flat,dim=1)

        total_n = 0
        
        image_result = []
        label_result = []
        use_cp = []
        masking_result = []
        for ib in range(b):
            if random.random() < 0.5:
                image_result.append(img[ib:ib+1,:,:,:,:])
                label_result.append(label[ib:ib+1,:,:])
                masking_result.append(torch.zeros_like(label[ib:ib+1,:,:]).cuda())
                use_cp.append(0)
                continue

            if paste_label_cond_num[ib]==0:
                image_result.append(img[ib:ib+1,:,:,:,:])
                label_result.append(label[ib:ib+1,:,:])
                masking_result.append(torch.zeros_like(label[ib:ib+1,:,:]).cuda())
                use_cp.append(0)
                continue
            if ib>0:
                idx_cond = total_n + random.randint(0,paste_label_cond_num[ib]-1)
                total_n += paste_label_cond_num[ib]
            else:
                idx_cond = random.randint(0,paste_label_cond_num[ib]-1)
                total_n = paste_label_cond_num[0]
                
            p_idx = paste_label_idx[idx_cond]
            sampling_cls = paste_label_flat[p_idx[0],p_idx[1]]
            mask = (paste_label[ib:ib+1,:,:] == sampling_cls).unsqueeze(0)
            p_img = paste_img[ib:ib+1,:,:,:]
            p_label = paste_label[ib:ib+1,:,:]
            ratio = random.random()
            ratio = 0.5 + 0.4*ratio
            hn, wn = int(h*ratio), int(w*ratio)
            mask = F.interpolate(mask.float(), size=(hn,wn), mode='bilinear', align_corners=True)
            p_img = F.interpolate(p_img, size=(hn,wn), mode='bilinear', align_corners=True)
            mask = (mask>0.5)

            p_img = p_img*mask.detach()
            p_label = p_img*mask.detach()

            x1 = random.randint(0, h-hn-1)
            y1 = random.randint(0, w-wn-1)

            m_img = torch.zeros_like(img[ib,:,:,:,:]).cuda()
            pm_img = torch.zeros_like(img[ib,:,:,:,:]).cuda()

            c_img = img[ib,:,:,:,:].clone()
            m_img[:,:,x1:x1+hn,y1:y1+wn] = mask.expand(2,3,-1,-1)
            pm_img[:,:,x1:x1+hn,y1:y1+wn] = p_img
            r_img = c_img*(1-m_img.int()) + pm_img
            image_result.append(r_img.unsqueeze(0))

            m_label = torch.zeros_like(label[ib:ib+1,:,:]).cuda()
            pm_label = torch.zeros_like(label[ib:ib+1,:,:]).cuda()

            m_label[:,x1:x1+hn,y1:y1+wn] = mask
            c_label = sampling_cls * m_label
            label_result.append(c_label.long())
            masking_result.append(m_label)
            use_cp.append(sampling_cls)

        image_result = torch.cat(image_result,dim=0).cuda()
        label_result = torch.cat(label_result,dim=0).cuda()
        masking_result = torch.cat(masking_result,dim=0).cuda()

        return image_result, label_result, masking_result, torch.Tensor(use_cp).cuda()
        
    def generate_cutout_mask(self, img_size, cos_mask=None, ratio=2):
        if cos_mask == None:
            cutout_area = img_size[0] * img_size[1] / ratio

            w = np.random.randint(img_size[1] / ratio + 1, img_size[1])
            h = np.round(cutout_area / w)

            x_start = np.random.randint(0, img_size[1] - w + 1)
            y_start = np.random.randint(0, img_size[0] - h + 1)

            x_end = int(x_start + w)
            y_end = int(y_start + h)

            mask = torch.ones(img_size)
            mask[y_start:y_end, x_start:x_end] = 0
            return mask.long()
        else:
            cutout_area = img_size[0] * img_size[1] / ratio

            m_score = 0
            min_s = 1e30
            for ir in range(10):
                w = np.random.randint(img_size[1] / ratio + 1, img_size[1])
                h = np.round(cutout_area / w)
                x_start = np.random.randint(0, img_size[1] - w + 1)
                y_start = np.random.randint(0, img_size[0] - h + 1)

                x_end = int(x_start + w)
                y_end = int(y_start + h)
                i_score = torch.mean(cos_mask[y_start:y_end, x_start:x_end])
                
                if i_score >= m_score:
                    m_score = i_score
                    mask = torch.ones(img_size)
                    mask[y_start:y_end, x_start:x_end] = 0
                   
            return mask.long()

    def generate_class_mask(self, pseudo_labels, use_bg=True):
        labels = torch.unique(pseudo_labels)  # all unique labels
        if use_bg==False:
            labels = labels[labels!=self.ignore_index]
            labels = labels[labels!=0]

        labels_select = labels[torch.randperm(len(labels))][
            : 1 + len(labels) // 2
        ]  # randomly select half of labels

        mask = (pseudo_labels.unsqueeze(-1) == labels_select).any(-1)
        return mask.float()
        
    def generate_unsup_data(self, data, target, target_F, logits, logits_F, cos_masks=None, mode="cutout", target_ul=None):
        batch_size, _, im_h, im_w = data.shape
        _, t_h, t_w = target.shape
        device = data.device
        target = F.interpolate(target.unsqueeze(1).float(), size=[im_h, im_w], mode='nearest')
        logits = F.interpolate(logits.unsqueeze(1), size=[im_h, im_w], mode='bilinear', align_corners=True)

        target_F = F.interpolate(target_F.unsqueeze(1).float(), size=[im_h, im_w], mode='nearest')
        logits_F = F.interpolate(logits_F.unsqueeze(1), size=[im_h, im_w], mode='bilinear', align_corners=True)

        
        if cos_masks != None:
            cos_masks = F.interpolate(cos_masks.unsqueeze(1), size=[im_h, im_w], mode='bilinear', align_corners=True)
            cos_masks = cos_masks.view(-1,im_h,im_w)
        target = target.view(batch_size, im_h, im_w).long()
        logits = logits.view(batch_size, im_h, im_w)
        target_F = target_F.view(batch_size, im_h, im_w).long()
        logits_F = logits_F.view(batch_size, im_h, im_w)

        if target_ul != None:
            target_ul = F.interpolate(target_ul.unsqueeze(1).float(), size=[im_h, im_w], mode='nearest')
            target_ul = target_ul.view(batch_size, im_h, im_w).long()



        new_data = []
        new_target = []
        new_logits = []
        new_target_F = []
        new_logits_F = []
        new_target_ul = []
        vis_data = data.clone().detach()
        for i in range(batch_size):
            if cos_masks != None:
                mix_mask = self.generate_cutout_mask([im_h, im_w],cos_masks[(i + 1) % batch_size,:,:])
            else:
                mix_mask = self.generate_cutout_mask([im_h, im_w],cos_mask=None)
            
            mix_mask = mix_mask.to(device)
            
            
            if torch.rand(1) > 0.75:
                new_data.append(
                    (
                        data[i] * mix_mask + data[(i + 1) % batch_size] * (1 - mix_mask)
                    ).unsqueeze(0)
                )
                new_target.append(
                    (
                        target[i] * mix_mask + target[(i + 1) % batch_size] * (1 - mix_mask)
                    ).unsqueeze(0)
                )
                new_logits.append(
                    (
                        logits[i] * mix_mask + logits[(i + 1) % batch_size] * (1 - mix_mask)
                    ).unsqueeze(0)
                )
                new_target_F.append(
                    (
                        target_F[i] * mix_mask + target_F[(i + 1) % batch_size] * (1 - mix_mask)
                    ).unsqueeze(0)
                )
                new_logits_F.append(
                    (
                        logits_F[i] * mix_mask + logits_F[(i + 1) % batch_size] * (1 - mix_mask)
                    ).unsqueeze(0)
                )

                if target_ul != None:
                    new_target_ul.append(
                    (
                        target_ul[i] * mix_mask + target_ul[(i + 1) % batch_size] * (1 - mix_mask)
                    ).unsqueeze(0)
                )

            else:
                new_data.append(data[i].unsqueeze(0))
                new_target.append(target[i].unsqueeze(0))
                new_logits.append(logits[i].unsqueeze(0))
                new_target_F.append(target_F[i].unsqueeze(0))
                new_logits_F.append(logits_F[i].unsqueeze(0))
                new_target_ul.append(target_ul[i].unsqueeze(0))

        new_data, new_target, new_logits, new_target_F, new_logits_F = (
            torch.cat(new_data),
            torch.cat(new_target),
            torch.cat(new_logits),
            torch.cat(new_target_F),
            torch.cat(new_logits_F),
        )

        

        new_target = F.interpolate(new_target.unsqueeze(1).float(), size=[t_h, t_w], mode='nearest')
        new_logits = F.interpolate(new_logits.unsqueeze(1), size=[t_h, t_w], mode='bilinear', align_corners=True)

        new_target_F = F.interpolate(new_target_F.unsqueeze(1).float(), size=[t_h, t_w], mode='nearest')
        new_logits_F = F.interpolate(new_logits_F.unsqueeze(1), size=[t_h, t_w], mode='bilinear', align_corners=True)

        new_target = new_target.view(batch_size, t_h, t_w)
        new_logits = new_logits.view(batch_size, t_h, t_w)
        new_target_F = new_target_F.view(batch_size, t_h, t_w)
        new_logits_F = new_logits_F.view(batch_size, t_h, t_w)

        if target_ul != None:
            new_target_ul = torch.cat(new_target_ul)
            new_target_ul = F.interpolate(new_target_ul.unsqueeze(1).float(), size=[t_h, t_w], mode='nearest')
            new_target_ul = new_target_ul.view(batch_size, t_h, t_w)
            return new_data, new_target.long(), new_target_F.long(), new_logits, new_logits_F, new_target_ul

        return new_data, new_target.long(), new_target_F.long(), new_logits, new_logits_F
    
    

    def forward(self, x_l=None, target_l=None, x_ul=None, target_ul=None,
                curr_iter=None, epoch=None, gpu=None, gt_l=None, ul1=None, br1=None, ul2=None, br2=None, flip=None,img_id=None,batch_idx=None, total_epoch=None):
        if not self.training:
            enc1, enc2 = self.encoder(x_l)
            logits_w = self.classifier_CE(enc1)
            logits_w_Focal = self.classifier_Focal(enc2)

            output_ce = F.interpolate(logits_w, size=x_l.size()[2:], mode='bilinear', align_corners=True)
            output_fl = F.interpolate(logits_w_Focal, size=x_l.size()[2:], mode='bilinear', align_corners=True)
    
            
            return output_ce, output_fl

        if self.mode == 'supervised':
            feat = self.encoder(x_l)
            enc = self.classifier(feat)
            output_l = F.interpolate(enc, size=x_l.size()[2:], mode='bilinear', align_corners=True)
            loss_sup = self.sup_loss(output_l, target_l, ignore_index=self.ignore_index,
                                     temperature=1.0) * self.sup_loss_w
            curr_losses = {'Ls': loss_sup}
            outputs = {'sup_pred': output_l}
            total_loss = loss_sup

            #C_sim part#
            _,_,h,w = feat.shape
            target_small = F.interpolate(target_l, scale_factor=(h,w), mode='nearest')
            idx_rand_h = torch.randint(0, h, (4,))
            idx_rand_w = torch.randint(0, w, (4,))

            c_sampled = torch.cat([target_small[:, ih, iw].unsqueeze(0) for ih, iw in zip(idx_rand_h, idx_rand_w)])
            
            curr_losses['Ls_sub'] = 0
            total_loss = total_loss + loss_sup_SubCls * self.loss_weight_subcls

            return total_loss, curr_losses, outputs

        elif self.mode == 'semi':
            #x_ul_new, target_ul_new, _, use_cp = self.copy_paste(x_ul.detach(), target_ul.detach(), x_l.detach(), target_l.detach())
            feat1, feat2 = self.encoder(x_l)
            enc_CE = self.classifier_CE(feat1)
            output_CE_l = F.interpolate(enc_CE, size=x_l.size()[2:], mode='bilinear', align_corners=True)
            loss_sup_CE = self.sup_loss(output_CE_l, target_l.detach(), ignore_index=self.ignore_index,
                                     temperature=1.0) * self.sup_loss_w
            curr_losses = {'Ls': loss_sup_CE}
            outputs = {'sup_pred': output_CE_l}
            total_loss = loss_sup_CE
            
            enc_Focal = self.classifier_Focal(feat2)
            output_Focal_l = F.interpolate(enc_Focal, size=x_l.size()[2:], mode='bilinear', align_corners=True)
            #loss_sup_Focal = self.criterion_Focal(output_Focal_l, target_l.clone().detach(),pos_mask=(target_l!=255)) * self.sup_loss_w
            
            outputs['sup_pred_Focal'] = output_Focal_l
            #total_loss = loss_sup_Focal

            #total_loss = total_loss + loss_sup_Focal


            pred_CE = np.asarray(enc_CE.max(1)[1][0,:,:].detach().cpu(), np.uint8)
            pred_col_CE = colorize_mask(pred_CE, self.palette)
            pred_col_CE.save('./temp/pred_CE.png')

            pred_Focal = np.asarray(enc_Focal.max(1)[1][0,:,:].squeeze(0).detach().cpu(), np.uint8)
            pred_col_Focal = colorize_mask(pred_Focal, self.palette)
            pred_col_Focal.save('./temp/pred_Focal.png')
            #####pixel_sim supervised part#####
            
            if epoch < self.epoch_start_unsup:
                return total_loss, curr_losses, outputs
            
            # x_ul: [batch_size, 2, 3, H, W]
            x_w = x_ul[:, 0, :, :, :]  # Weak Aug
            x_s = x_ul[:, 1, :, :, :]  # Strong Aug

            feat_w1,feat_w2 = self.encoder(x_w)
            if self.downsample:
                feat_w1 = F.avg_pool2d(feat_w1, kernel_size=2, stride=2)
                feat_w2 = F.avg_pool2d(feat_w2, kernel_size=2, stride=2)
            logits_w = self.classifier_CE(feat_w1)
            logits_w_Focal = self.classifier_Focal(feat_w2)



            seg_w = F.softmax(logits_w, 1)
            seg_w_F = F.softmax(logits_w_Focal, 1)

            pseudo_logits_w = seg_w.max(1)[0].detach()
            pseudo_label_w = seg_w.max(1)[1].detach()

            pseudo_logits_w_F = seg_w_F.max(1)[0].detach()
            pseudo_label_w_F = seg_w_F.max(1)[1].detach()
            
            if self.use_cutmix:
                #cos_masks = F.cosine_similarity(seg_w, seg_w_F,dim=1).detach()
                #cos_masks = cos_masks**4
                x_s_cutmix, pseudo_label_w_aug, pseudo_label_w_F_aug, pseudo_logits_w_aug, pseudo_logits_w_F_aug, target_ul_aug = self.generate_unsup_data(x_s, pseudo_label_w, pseudo_label_w_F, pseudo_logits_w, pseudo_logits_w_F, None, mode="cutmix", target_ul=target_ul)
                #save_image(x_s_cutmix[0,:,:,:].cpu().detach(), './temp/cut_mix_img.png')
                #save_image(1-((cos_masks[0,:,:])).cpu().detach(), './temp/cos_vis.png')
                feat_s1,feat_s2 = self.encoder(x_s_cutmix)

                if self.downsample:
                    feat_s1 = F.avg_pool2d(feat_s1, kernel_size=2, stride=2)
                    feat_s2 = F.avg_pool2d(feat_s2, kernel_size=2, stride=2)
                logits_s = self.classifier_CE(feat_s1)
                logits_s_Focal = self.classifier_Focal(feat_s2)

                seg_s = F.softmax(logits_s, 1)
                pseudo_logits_s = seg_s.max(1)[0].detach()
                pseudo_label_s = seg_s.max(1)[1].detach()

                CE_mask = (pseudo_label_w_aug == pseudo_label_w_F_aug)*(pseudo_logits_w_aug > self.pos_thresh_value)
                #FL_mask = (pseudo_label_w_aug == pseudo_label_w_F_aug)*(pseudo_logits_w_F_aug > self.pos_thresh_value)
                easy_mask = (pseudo_logits_s > self.pos_thresh_value-0.1)*(pseudo_label_s==pseudo_label_w_F_aug)
                FL_mask = (easy_mask*(pseudo_label_w_aug == pseudo_label_w_F_aug)*(pseudo_logits_w_F_aug > self.pos_thresh_value) \
                            + torch.logical_not(easy_mask)*(pseudo_label_w_aug == pseudo_label_w_F_aug)*(mix_logit > self.pos_thresh_value)) > 0

                New_pseudo_label = pseudo_label_w_aug*CE_mask + (1-CE_mask.int())*self.ignore_index
                New_pseudo_label_F = pseudo_label_w_F_aug*FL_mask + (1-FL_mask.int())*self.ignore_index
                New_pseudo_label = New_pseudo_label.long()
                New_pseudo_label_F = New_pseudo_label_F.long()

                outputs['ps_ce'] = New_pseudo_label.detach()
                outputs['ps_fl'] = New_pseudo_label_F.detach()
                outputs['target_ul'] = target_ul_aug.detach()
            else:
                
                feat_s1,feat_s2 = self.encoder(x_s)

                if self.downsample:
                    feat_s1 = F.avg_pool2d(feat_s1, kernel_size=2, stride=2)
                    feat_s2 = F.avg_pool2d(feat_s2, kernel_size=2, stride=2)
                logits_s = self.classifier_CE(feat_s1)
                logits_s_Focal = self.classifier_Focal(feat_s2)

                seg_s = F.softmax(logits_s, 1)
                seg_s_F = F.softmax(logits_s_Focal, 1)

                ent_w = self.prob_2_entropy(seg_w).detach()
                ent_w_F = self.prob_2_entropy(seg_w_F).detach()

                ent_s = self.prob_2_entropy(seg_s).detach()
                ent_s_F = self.prob_2_entropy(seg_s_F).detach()

                outputs['ent_w'] = ent_w.detach()
                outputs['ent_w_F'] = ent_w_F.detach()
                outputs['ent_s'] = ent_s.detach()
                outputs['ent_s_F'] = ent_s_F.detach()


                #save_image(ent_w_F[0,:,:].cpu().detach(), './temp/ent_w_F.png')

                CE_mask = (pseudo_label_w == pseudo_label_w_F)*(pseudo_logits_w > 0.75)
                FL_mask = (pseudo_label_w == pseudo_label_w_F)*(pseudo_logits_w_F > 0.75)
                
                

                New_pseudo_label = pseudo_label_w*CE_mask + (1-CE_mask.int())*self.ignore_index
                New_pseudo_label_F = pseudo_label_w_F*FL_mask + (1-FL_mask.int())*self.ignore_index
                New_pseudo_label = New_pseudo_label.long()
                New_pseudo_label_F = New_pseudo_label_F.long()

                outputs['ps_ce'] = New_pseudo_label.detach()
                outputs['ps_fl'] = New_pseudo_label_F.detach()


                target_ul_s = F.interpolate(target_ul.unsqueeze(1).float(), size=New_pseudo_label_F.shape[-2:], mode='nearest')
                outputs['target_ul'] = target_ul_s.detach()
                
            outputs['ce_mask'] = CE_mask
            outputs['fl_mask'] = FL_mask

            """
            if self.downsample:
                feat_s1 = F.avg_pool2d(feat_s1, kernel_size=2, stride=2)
                feat_s2 = F.avg_pool2d(feat_s2, kernel_size=2, stride=2)
            logits_s = self.classifier_CE(feat_s1)
            logits_s_Focal = self.classifier_Focal(feat_s2)
            """


            ###pseudo pos mask Generate####
            
            

            

            ####################################
            #save_image(1-((cos_masks_local[0,:,:])**10).cpu().detach(), './temp/cos_vis_local.png')
         
            pred_pseudo_CE = np.asarray(pseudo_label_w[0,:,:].detach().cpu(), np.uint8)
            pred_col_pseudo_CE = colorize_mask(pred_pseudo_CE, self.palette)
            pred_col_pseudo_CE.save('./temp/pred_pseudo_CE.png')
            
            pred_pseudo_FL = np.asarray(pseudo_label_w_F[0,:,:].squeeze(0).detach().cpu(), np.uint8)
            pred_col_pseudo_FL = colorize_mask(pred_pseudo_FL, self.palette)
            pred_col_pseudo_FL.save('./temp/pred_pseudo_FL.png')

            pred_pseudo_New = np.asarray(New_pseudo_label[0,:,:].squeeze(0).detach().cpu(), np.uint8)
            pred_col_pseudo_New = colorize_mask(pred_pseudo_New, self.palette)
            pred_col_pseudo_New.save('./temp/pred_pseudo_New.png')
           
            pred_pseudo_New_F = np.asarray(New_pseudo_label_F[0,:,:].squeeze(0).detach().cpu(), np.uint8)
            pred_col_pseudo_New_F = colorize_mask(pred_pseudo_New_F, self.palette)
            pred_col_pseudo_New_F.save('./temp/pred_pseudo_New_F.png')

            loss_sup_Focal = self.criterion_Focal(output_Focal_l, target_l.clone().detach(),pos_mask=(target_l!=255), num_classes=self.num_classes, scale_factor=2) * self.sup_loss_w
            total_loss = total_loss + loss_sup_Focal
            
            #loss_sup_Focal = loss_sup_Focal.detach()
            curr_losses['Ls_Focal'] = loss_sup_Focal

            loss_focal = self.criterion_Focal(logits_s_Focal, New_pseudo_label.detach(),pos_mask=CE_mask, num_classes=self.num_classes, scale_factor=2, keep_dim=True)
            loss_cross = (F.cross_entropy(logits_s, New_pseudo_label_F.detach().clone(), reduction='none',ignore_index=self.ignore_index) * FL_mask.detach())
            
            pseudo_logits_w_F = pseudo_logits_w_F.detach()
            loss_unsup_CF = loss_focal
            loss_unsup_CF = loss_unsup_CF.sum() / torch.sum(CE_mask)

            weight_loss = pseudo_logits_w_F**4
            loss_unsup_FC = loss_cross * weight_loss
            loss_unsup_FC = loss_unsup_FC.sum()  / torch.sum(weight_loss*FL_mask)
            loss_unsup = 0.5*(loss_unsup_CF + loss_unsup_FC)

            curr_losses['Lu_reg'] = loss_unsup

            
            total_loss = total_loss + loss_unsup * self.loss_weight_unsup

            outputs['ce_loss'] = pseudo_logits_w_F * (F.cross_entropy(logits_s, New_pseudo_label_F.detach().clone(), reduction='none',ignore_index=self.ignore_index) * FL_mask.detach()).detach()
            outputs['fl_loss'] = pseudo_logits_w_F * self.criterion_Focal(logits_s_Focal, New_pseudo_label.detach(),pos_mask=CE_mask,keep_dim=True, num_classes=self.num_classes, scale_factor=2).detach()
            
            outputs['seg_ce_w'] = seg_w.detach()
            outputs['seg_fl_w'] = seg_w_F.detach()

            
            outputs['seg_ce_s'] = F.softmax(logits_s,1).detach()
            outputs['seg_fl_s'] = F.softmax(logits_s_Focal,1).detach()

            outputs['ent_weight'] = self.ent_weight.detach()

            return total_loss, curr_losses, outputs

        else:
            raise ValueError("No such mode {}".format(self.mode))

    def loss_vis(self, prob, target):

        n,H,W = target.size()
        prob_l = F.interpolate(prob, size=(H,W), mode='bilinear', align_corners=True)
        pseudo_label = prob_l.max(1)[1].detach()
        
        pos_mask = (target!=self.ignore_index) * (pseudo_label!=target)
        loss_2d = F.cross_entropy(prob_l, target.long().detach(), reduction='none',ignore_index=self.ignore_index) * pos_mask
        return loss_2d

    def prob_2_entropy(self, prob):
        n, c, h, w = prob.size()
        return -torch.sum(torch.mul(prob, torch.log2(prob + 1e-30)),dim=1) / np.log2(c)

    def prob_2_entropy_v2(self, prob1, prob2):
        n, c, h, w = prob1.size()
        ent1 = -torch.sum(torch.mul(prob1, torch.log2(prob2 + 1e-30)),dim=1) / np.log2(c)
        ent2 = -torch.sum(torch.mul(prob2, torch.log2(prob1 + 1e-30)),dim=1) / np.log2(c)
        
        return 0.5*(ent1+ent2)


    def SubCls_to_ParentCls(self, label_SubCls):
        label_SubCls_to_ParentCls = label_SubCls.clone()
        subclasses = np.cumsum(np.asarray(self.split_list))
        subclasses = np.insert(subclasses, 0, 0)
        parentclasses = np.uint8(np.linspace(1,len(self.split_list),len(self.split_list))-1)
        for subcls_lower, subcls_upper, parcls in zip(np.flip(subclasses[:-1]), np.flip(subclasses[1:]), np.flip(parentclasses)):
            label_SubCls_to_ParentCls[(label_SubCls>=subcls_lower)*(label_SubCls<subcls_upper)] = parcls
        return label_SubCls_to_ParentCls.cuda().long()

    def concat_all_gather(self, tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        with torch.no_grad():
            tensors_gather = [torch.ones_like(tensor)
                              for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

            output = torch.cat(tensors_gather, dim=0)
        return output

    def get_backbone_params(self):
        return self.encoder.get_backbone_params()

    def get_other_params(self):
        return chain(self.encoder.get_module_params(), self.classifier_CE.parameters(), self.classifier_Focal.parameters())