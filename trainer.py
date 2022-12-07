import torch
import time, random, cv2, sys
from math import ceil
import numpy as np
from itertools import cycle
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchvision import transforms
from base import BaseTrainer
from utils.helpers import colorize_mask
from utils.metrics import eval_metrics, AverageMeter
from tqdm import tqdm
from PIL import Image
import pandas as pd
import seaborn as sns
from utils.helpers import DeNormalize
import torch.distributed as dist
import os
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import PIL
import torchvision.transforms as T
from utils.criterion import FocalLoss2d
from torchmetrics import ConfusionMatrix

class Test(BaseTrainer):
    def __init__(self, model, resume, config, iter_per_epoch, val_loader=None, train_logger=None, gpu=None, test=False, visualize=False, cont_vis=False):
        super(Test, self).__init__(model, resume, config, iter_per_epoch, train_logger, gpu=gpu, test=test, visualize=visualize)

        self.val_loader = val_loader

        self.ignore_index = self.val_loader.dataset.ignore_index
        self.wrt_mode, self.wrt_step = 'train_', 0
        self.log_step = config['trainer'].get('log_per_iter', int(np.sqrt(self.val_loader.batch_size)))
        if config['trainer']['log_per_iter']:
            self.log_step = int(self.log_step / self.val_loader.batch_size) + 1

        self.num_classes = self.val_loader.dataset.num_classes
        self.mode = self.model.module.mode
        self.test = test
        self.save_dir = config['trainer']['save_dir'] + config['experim_name']
        self.visualize = visualize
        self.cont_vis = cont_vis
        # TRANSORMS FOR VISUALIZATION
        self.restore_transform = transforms.Compose([
            DeNormalize(self.val_loader.MEAN, self.val_loader.STD),
            transforms.ToPILImage()])
        self.viz_transform = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor()])

        self.start_time = time.time()

        self.n_labeled_examples = config['n_labeled_examples']

        self.dataset = config['dataset']





    def _train_epoch(self, epoch):
        print(epoch)

    def _valid_epoch(self, epoch):
        if self.val_loader is None:
            if self.gpu == 0:
                self.logger.warning('Not data loader was passed for the validation step, No validation is performed !')
            return {}

        if self.gpu == 0:
            self.logger.info('\n###### EVALUATION ######')

        self.model.eval()
        self.wrt_mode = 'val'

        total_loss_val = AverageMeter()
        total_inter, total_union = 0, 0
        total_correct, total_label = 0, 0

        save_folder = os.path.join(self.save_dir, 'preds')
        os.makedirs(save_folder, exist_ok=True)

        if self.dataset == 'cityscapes':
            from utils import pallete
            palette = pallete.citypallete
        else:
            from utils import pallete
            palette = pallete.get_voc_pallete(self.num_classes)


        tbar = tqdm(self.val_loader, ncols=160)
        if self.cont_vis:
            with torch.no_grad():
                
                total_sig = torch.zeros(self.num_classes).cuda()
                total_num = torch.zeros(self.num_classes).cuda()
                for batch_idx, (data, target, image_id) in enumerate(tbar):
                    target, data = target.cuda(non_blocking=True), data.cuda(non_blocking=True)
                    H, W = target.size(1), target.size(2)
                    up_sizes = (ceil(H / 8) * 8, ceil(W / 8) * 8)
                    pad_h, pad_w = up_sizes[0] - data.size(2), up_sizes[1] - data.size(3)
                    data = F.pad(data, pad=(0, pad_w, 0, pad_h), mode='reflect')

                    output_ce, output_fl = self.model(data)
                    output_ce = output_ce[:, :, :H, :W]
                    output_fl = output_fl[:, :, :H, :W]
                    b,_,h,w = output_fl.shape

                    output_ce = F.interpolate(output_ce, size=(H,W), mode='bilinear', align_corners=True)
                    output_fl = F.interpolate(output_fl, size=(H,W), mode='bilinear', align_corners=True)

                    seg_ce = F.softmax(output_ce, 1)
                    seg_fl = F.softmax(output_fl, 1)
                    
                    pseudo_label_ce = seg_ce.max(1)[1].detach()
                    pseudo_label_fl = seg_fl.max(1)[1].detach()

                    pseudo_logit_ce = seg_ce.max(1)[0].detach()
                    pseudo_logit_fl = seg_fl.max(1)[0].detach()

                    pseudo_label_ce[pseudo_label_ce!=pseudo_label_fl] = self.ignore_index
                    pseudo_label_fl[pseudo_label_ce!=pseudo_label_fl] = self.ignore_index

                    sig_logit = torch.abs(pseudo_logit_ce-pseudo_logit_fl)

                    for c in range(self.num_classes):
                        total_sig[c] += torch.sum(sig_logit[pseudo_label_ce==c])
                        total_num[c] += torch.sum(pseudo_label_ce==c)
                    
                    correct, labeled, inter, union = eval_metrics(output_ce, target, self.num_classes, self.ignore_index)

                    total_inter, total_union = total_inter + inter, total_union + union
                    total_correct, total_label = total_correct + correct, total_label + labeled
            IoU = 1.0 * total_inter / (np.spacing(1) + total_union)

            total_score = total_sig/total_num
            for c in range(self.num_classes):
                print("{}th class: {}".format(c, total_score[c]))
            
            s_max = torch.max(total_score)
            for c in range(self.num_classes):
                print("{}th class: {}".format(c, 1 - total_score[c]/s_max))
            result_score = 1 - total_score/s_max
            
            _, indices_s = torch.sort(result_score)
            _, indices_i = torch.sort(torch.from_numpy(IoU))
            print(indices_s, indices_i)
            x_values = list(range(self.num_classes))
            plt.plot(x_values, result_score.detach().cpu().tolist())
            plt.savefig('score.png')
            plt.cla()
            plt.plot(x_values, IoU)
            plt.savefig('IoU.png')
        else:                
            with torch.no_grad():
                for batch_idx, (data, target, image_id) in enumerate(tbar):
                    target, data = target.cuda(non_blocking=True), data.cuda(non_blocking=True)
                    H, W = target.size(1), target.size(2)
                    up_sizes = (ceil(H / 8) * 8, ceil(W / 8) * 8)
                    pad_h, pad_w = up_sizes[0] - data.size(2), up_sizes[1] - data.size(3)
                    data = F.pad(data, pad=(0, pad_w, 0, pad_h), mode='reflect')

                    
                    output,_ = self.model(data)
                    output = output[:, :, :H, :W]
                    output_softmax = F.softmax(output,dim=1)
                    pred_logit = output.max(1)[0]
                    pred_logit_softmax = output_softmax.max(1)[0]
                    pred_label = output_softmax.max(1)[1]
                    
                    """
                    for c in range(self.num_classes):
                        if torch.sum(pred_label==c) <5 and torch.sum(target==c) <5:
                            continue
                        right_mask = (pred_label==c) * ((pred_label==target))
                        wrong_mask = (pred_label==c) * ((pred_label!=target))
                        pred_logit
                        logit_dist_r = pred_logit[right_mask]
                        logit_dist_w = pred_logit[wrong_mask]
                        logit_dist_r_soft = pred_logit_softmax[right_mask]
                        logit_dist_w_soft = pred_logit_softmax[wrong_mask]

                        hist_r = torch.histc(logit_dist_r, bins=40, min=-20, max=20)
                        hist_w = torch.histc(logit_dist_w, bins=40, min=-20, max=20)
                        hist_r_soft = torch.histc(logit_dist_r_soft, bins=20, min=0, max=1)
                        hist_w_soft = torch.histc(logit_dist_w_soft, bins=20, min=0, max=1)

                        plt.clf()
                        x_axis = list(range(40))
                        x_axis = torch.Tensor(x_axis)
                        plt.bar(x_axis, hist_r.detach().cpu(), align='center', color=['forestgreen'],alpha=0.5,linewidth=0.05)
                        plt.bar(x_axis, hist_w.detach().cpu(), align='center', color=['red'],alpha=0.5,linewidth=0.05)
                        plt.savefig('./temp/histogram/{}_{}.png'.format(image_id[0],c), dpi=300)

                        plt.clf()
                        x_axis = list(range(20))
                        x_axis = torch.Tensor(x_axis)
                        plt.bar(x_axis, hist_r_soft.detach().cpu(), align='center', color=['forestgreen'],alpha=0.5,linewidth=0.05)
                        plt.bar(x_axis, hist_w_soft.detach().cpu(), align='center', color=['red'],alpha=0.5,linewidth=0.05)
                        plt.savefig('./temp/histogram/{}_{}_soft.png'.format(image_id[0],c), dpi=300)
                    pred = np.asarray(output.max(1)[1].squeeze(0).detach().cpu(), np.uint8)
                    pred_col = colorize_mask(pred, palette)
                    pred_col.save(os.path.join(save_folder, image_id[0] + '.png'))
                    """
                    # LOSS
                    loss = F.cross_entropy(output, target, ignore_index=self.ignore_index)
                    total_loss_val.update(loss.item())

                    correct, labeled, inter, union = eval_metrics(output, target, self.num_classes, self.ignore_index)

                    total_inter, total_union = total_inter + inter, total_union + union
                    total_correct, total_label = total_correct + correct, total_label + labeled
                    IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
                    mIoU = IoU.mean()

                    seg_metrics = {"Mean_IoU": np.round(100*mIoU,2), "Class_IoU": dict(zip(range(self.num_classes), np.round(100*IoU,2)))}
                    if self.gpu == 0:
                        tbar.set_description('EVAL ({}) | Loss: {:.3f}, mIoU: {:.2f} |'.format(epoch, total_loss_val.average, 100*mIoU))
                

                

                if self.gpu == 0:
                    # METRICS TO TENSORBOARD
                    self.wrt_step = (epoch) * len(self.val_loader)
                    self.writer.add_scalar(f'{self.wrt_mode}/loss', total_loss_val.average, self.wrt_step)
                    for k, v in list(seg_metrics.items())[:-1]: self.writer.add_scalar(f'{self.wrt_mode}/{k}', v, self.wrt_step)

                log = {
                    'val_loss': total_loss_val.average,
                    **seg_metrics
                }

        return log

    def SubCls_to_ParentCls(self, label_SubCls):
        label_SubCls_to_ParentCls = label_SubCls.copy()
        subclasses = np.cumsum(np.asarray(self.split_list))
        subclasses = np.insert(subclasses, 0, 0)
        parentclasses = np.uint8(np.linspace(1,len(self.split_list),len(self.split_list))-1)
        for subcls_lower, subcls_upper, parcls in zip(np.flip(subclasses[:-1]), np.flip(subclasses[1:]), np.flip(parentclasses)):
            label_SubCls_to_ParentCls[(label_SubCls>=subcls_lower)*(label_SubCls<subcls_upper)] = parcls
        return label_SubCls_to_ParentCls

    def _reset_metrics(self):
        self.loss_sup = AverageMeter()
        self.loss_unsup = AverageMeter()
        self.loss_weakly = AverageMeter()
        self.pair_wise = AverageMeter()
        self.total_inter_l, self.total_union_l = 0, 0
        self.total_correct_l, self.total_label_l = 0, 0
        self.total_inter_ul, self.total_union_ul = 0, 0
        self.total_correct_ul, self.total_label_ul = 0, 0
        self.mIoU_l, self.mIoU_ul = 0, 0
        self.pixel_acc_l, self.pixel_acc_ul = 0, 0
        self.class_iou_l, self.class_iou_ul = {}, {}
        self.ClsReg_F1_l, self.ClsReg_F1_ul = 0, 0

    def _update_losses(self, cur_losses):
        for key in cur_losses:
            loss = cur_losses[key]
            n = loss.numel()
            count = torch.tensor([n]).long().cuda()
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            mean = loss.sum() / n
            if self.gpu == 0:
                getattr(self, key).update(mean.item())

    def _compute_metrics(self, outputs, target_l, target_ul, epoch):
        seg_metrics_l = eval_metrics(outputs['sup_pred'], target_l, self.num_classes, self.ignore_index)

        if self.gpu == 0:
            self._update_seg_metrics(*seg_metrics_l, True)
            seg_metrics_l = self._get_seg_metrics(True)
            self.pixel_acc_l, self.mIoU_l, self.class_iou_l = seg_metrics_l.values()

        if 'unsup_pred' in outputs:
            seg_metrics_ul = eval_metrics(outputs['unsup_pred'], target_ul, self.num_classes, self.ignore_index)

            if self.gpu == 0:
                self._update_seg_metrics(*seg_metrics_ul, False)
                seg_metrics_ul = self._get_seg_metrics(False)
                self.pixel_acc_ul, self.mIoU_ul, self.class_iou_ul = seg_metrics_ul.values()

    def _update_seg_metrics(self, correct, labeled, inter, union, supervised=True):
        if supervised:
            self.total_correct_l += correct
            self.total_label_l += labeled
            self.total_inter_l += inter
            self.total_union_l += union
        else:
            self.total_correct_ul += correct
            self.total_label_ul += labeled
            self.total_inter_ul += inter
            self.total_union_ul += union

    def _get_seg_metrics(self, supervised=True):
        if supervised:
            pixAcc = 1.0 * self.total_correct_l / (np.spacing(1) + self.total_label_l)
            IoU = 1.0 * self.total_inter_l / (np.spacing(1) + self.total_union_l)
        else:
            pixAcc = 1.0 * self.total_correct_ul / (np.spacing(1) + self.total_label_ul)
            IoU = 1.0 * self.total_inter_ul / (np.spacing(1) + self.total_union_ul)
        mIoU = IoU.mean()
        return {
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Mean_IoU": np.round(mIoU, 4),
            "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 4)))
        }

    def _log_values(self, cur_losses):
        logs = {}
        if "loss_sup" in cur_losses.keys():
            logs['loss_sup'] = self.loss_sup.average
        if "loss_unsup" in cur_losses.keys():
            logs['loss_unsup'] = self.loss_unsup.average
        if "loss_weakly" in cur_losses.keys():
            logs['loss_weakly'] = self.loss_weakly.average
        if "pair_wise" in cur_losses.keys():
            logs['pair_wise'] = self.pair_wise.average

        logs['mIoU_labeled'] = self.mIoU_l
        logs['pixel_acc_labeled'] = self.pixel_acc_l
        if self.mode == 'semi':
            logs['mIoU_unlabeled'] = self.mIoU_ul
            logs['pixel_acc_unlabeled'] = self.pixel_acc_ul
        return logs

    def _write_scalars_tb(self, logs):
        for k, v in logs.items():
            if 'class_iou' not in k: self.writer.add_scalar(f'train/{k}', v, self.wrt_step)
        for i, opt_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(f'train/Learning_rate_{i}', opt_group['lr'], self.wrt_step)

    def _add_img_tb(self, val_visual, wrt_mode):
        val_img = []
        palette = self.val_loader.dataset.palette
        for imgs in val_visual:
            imgs = [self.restore_transform(i) if (isinstance(i, torch.Tensor) and len(i.shape) == 3)
                    else colorize_mask(i, palette) for i in imgs]
            imgs = [i.convert('RGB') for i in imgs]
            imgs = [self.viz_transform(i) for i in imgs]
            val_img.extend(imgs)
        val_img = torch.stack(val_img, 0)
        val_img = make_grid(val_img.cpu(), nrow=val_img.size(0) // len(val_visual), padding=5)
        self.writer.add_image(f'{wrt_mode}/inputs_targets_predictions', val_img, self.wrt_step)

    def _write_img_tb(self, input_l, target_l, input_ul, target_ul, outputs, epoch):
        outputs_l_np = outputs['sup_pred'].data.max(1)[1].cpu().numpy()
        targets_l_np = target_l.data.cpu().numpy()
        imgs = [[i.data.cpu(), j, k] for i, j, k in zip(input_l, outputs_l_np, targets_l_np)]
        self._add_img_tb(imgs, 'supervised')

class Save_Features(BaseTrainer):
    def __init__(self, model, resume, config, iter_per_epoch, val_loader=None, train_logger=None, gpu=None, test=False):
        super(Save_Features, self).__init__(model, resume, config, iter_per_epoch, train_logger, gpu=gpu, test=test)

        self.val_loader = val_loader

        self.ignore_index = self.val_loader.dataset.ignore_index
        self.wrt_mode, self.wrt_step = 'train_', 0
        self.log_step = config['trainer'].get('log_per_iter', int(np.sqrt(self.val_loader.batch_size)))
        if config['trainer']['log_per_iter']:
            self.log_step = int(self.log_step / self.val_loader.batch_size) + 1

        self.num_classes = self.val_loader.dataset.num_classes
        self.mode = self.model.module.mode
        self.test = test
        self.save_dir = config['trainer']['save_dir'] + config['experim_name']

        # TRANSORMS FOR VISUALIZATION
        self.restore_transform = transforms.Compose([
            DeNormalize(self.val_loader.MEAN, self.val_loader.STD),
            transforms.ToPILImage()])
        self.viz_transform = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor()])

        self.start_time = time.time()

        self.n_labeled_examples = config['n_labeled_examples']

        self.dataset = config['dataset']

    def _train_epoch(self, epoch):
        print(epoch)

    def _valid_epoch(self, epoch):
        if self.val_loader is None:
            if self.gpu == 0:
                self.logger.warning('Not data loader was passed for the validation step, No validation is performed !')
            return {}

        if self.gpu == 0:
            self.logger.info('\n###### EVALUATION ######')

        self.model.eval()
        self.wrt_mode = 'val'

        total_loss_val = AverageMeter()
        total_inter, total_union = 0, 0
        total_correct, total_label = 0, 0

        save_folder = os.path.join(self.save_dir, 'features')
        os.makedirs(save_folder, exist_ok=True)

        if self.dataset == 'cityscapes':
            from utils import pallete
            palette = pallete.citypallete
        else:
            from utils import pallete
            palette = pallete.get_voc_pallete(self.num_classes)


        tbar = tqdm(self.val_loader, ncols=160)
        with torch.no_grad():
            for batch_idx, (data, target, image_id) in enumerate(tbar):
                target, data = target.cuda(non_blocking=True), data.cuda(non_blocking=True)

                H, W = target.size(1), target.size(2)
                up_sizes = (ceil(H / 8) * 8, ceil(W / 8) * 8)
                pad_h, pad_w = up_sizes[0] - data.size(2), up_sizes[1] - data.size(3)
                data = F.pad(data, pad=(0, pad_w, 0, pad_h), mode='reflect')

                # output = self.model(data)
                feat, output = self.model(data)

                feat = F.interpolate(feat, scale_factor=0.5, mode='bilinear', align_corners=True)
                feat = feat.cpu().numpy()
                feat = feat.astype(np.float16)
                for j, id in enumerate(image_id):
                    np.save(os.path.join(save_folder, id + '.npy'), feat[j,:])

                output = output[:, :, :H, :W]
                # LOSS
                loss = F.cross_entropy(output, target, ignore_index=self.ignore_index)
                total_loss_val.update(loss.item())

                correct, labeled, inter, union = eval_metrics(output, target, self.num_classes, self.ignore_index)
                total_inter, total_union = total_inter + inter, total_union + union
                total_correct, total_label = total_correct + correct, total_label + labeled
                IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
                mIoU = IoU.mean()

                seg_metrics = {"Mean_IoU": np.round(100*mIoU,2), "Class_IoU": dict(zip(range(self.num_classes), np.round(100*IoU,2)))}
                if self.gpu == 0:
                    tbar.set_description('EVAL ({}) | Loss: {:.3f}, mIoU: {:.2f} |'.format(epoch, total_loss_val.average, 100*mIoU))

            if self.gpu == 0:
                # METRICS TO TENSORBOARD
                self.wrt_step = (epoch) * len(self.val_loader)
                self.writer.add_scalar(f'{self.wrt_mode}/loss', total_loss_val.average, self.wrt_step)
                for k, v in list(seg_metrics.items())[:-1]: self.writer.add_scalar(f'{self.wrt_mode}/{k}', v, self.wrt_step)

            log = {
                'val_loss': total_loss_val.average,
                **seg_metrics
            }

        return log

    def SubCls_to_ParentCls(self, label_SubCls):
        label_SubCls_to_ParentCls = label_SubCls.copy()
        subclasses = np.cumsum(np.asarray(self.split_list))
        subclasses = np.insert(subclasses, 0, 0)
        parentclasses = np.uint8(np.linspace(1,len(self.split_list),len(self.split_list))-1)
        for subcls_lower, subcls_upper, parcls in zip(np.flip(subclasses[:-1]), np.flip(subclasses[1:]), np.flip(parentclasses)):
            label_SubCls_to_ParentCls[(label_SubCls>=subcls_lower)*(label_SubCls<subcls_upper)] = parcls
        return label_SubCls_to_ParentCls

    def _reset_metrics(self):
        self.loss_sup = AverageMeter()
        self.loss_unsup = AverageMeter()
        self.loss_weakly = AverageMeter()
        self.pair_wise = AverageMeter()
        self.total_inter_l, self.total_union_l = 0, 0
        self.total_correct_l, self.total_label_l = 0, 0
        self.total_inter_ul, self.total_union_ul = 0, 0
        self.total_correct_ul, self.total_label_ul = 0, 0
        self.mIoU_l, self.mIoU_ul = 0, 0
        self.pixel_acc_l, self.pixel_acc_ul = 0, 0
        self.class_iou_l, self.class_iou_ul = {}, {}
        self.ClsReg_F1_l, self.ClsReg_F1_ul = 0, 0

    def _update_losses(self, cur_losses):
        for key in cur_losses:
            loss = cur_losses[key]
            n = loss.numel()
            count = torch.tensor([n]).long().cuda()
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            mean = loss.sum() / n
            if self.gpu == 0:
                getattr(self, key).update(mean.item())

    def _compute_metrics(self, outputs, target_l, target_ul, epoch):
        seg_metrics_l = eval_metrics(outputs['sup_pred'], target_l, self.num_classes, self.ignore_index)

        if self.gpu == 0:
            self._update_seg_metrics(*seg_metrics_l, True)
            seg_metrics_l = self._get_seg_metrics(True)
            self.pixel_acc_l, self.mIoU_l, self.class_iou_l = seg_metrics_l.values()

        if 'unsup_pred' in outputs:
            seg_metrics_ul = eval_metrics(outputs['unsup_pred'], target_ul, self.num_classes, self.ignore_index)

            if self.gpu == 0:
                self._update_seg_metrics(*seg_metrics_ul, False)
                seg_metrics_ul = self._get_seg_metrics(False)
                self.pixel_acc_ul, self.mIoU_ul, self.class_iou_ul = seg_metrics_ul.values()

    def _update_seg_metrics(self, correct, labeled, inter, union, supervised=True):
        if supervised:
            self.total_correct_l += correct
            self.total_label_l += labeled
            self.total_inter_l += inter
            self.total_union_l += union
        else:
            self.total_correct_ul += correct
            self.total_label_ul += labeled
            self.total_inter_ul += inter
            self.total_union_ul += union

    def _get_seg_metrics(self, supervised=True):
        if supervised:
            pixAcc = 1.0 * self.total_correct_l / (np.spacing(1) + self.total_label_l)
            IoU = 1.0 * self.total_inter_l / (np.spacing(1) + self.total_union_l)
        else:
            pixAcc = 1.0 * self.total_correct_ul / (np.spacing(1) + self.total_label_ul)
            IoU = 1.0 * self.total_inter_ul / (np.spacing(1) + self.total_union_ul)
        mIoU = IoU.mean()
        return {
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Mean_IoU": np.round(mIoU, 4),
            "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 4)))
        }

    def _log_values(self, cur_losses):
        logs = {}
        if "loss_sup" in cur_losses.keys():
            logs['loss_sup'] = self.loss_sup.average
        if "loss_unsup" in cur_losses.keys():
            logs['loss_unsup'] = self.loss_unsup.average
        if "loss_weakly" in cur_losses.keys():
            logs['loss_weakly'] = self.loss_weakly.average
        if "pair_wise" in cur_losses.keys():
            logs['pair_wise'] = self.pair_wise.average

        logs['mIoU_labeled'] = self.mIoU_l
        logs['pixel_acc_labeled'] = self.pixel_acc_l
        if self.mode == 'semi':
            logs['mIoU_unlabeled'] = self.mIoU_ul
            logs['pixel_acc_unlabeled'] = self.pixel_acc_ul
        return logs

    def _write_scalars_tb(self, logs):
        for k, v in logs.items():
            if 'class_iou' not in k: self.writer.add_scalar(f'train/{k}', v, self.wrt_step)
        for i, opt_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(f'train/Learning_rate_{i}', opt_group['lr'], self.wrt_step)

    def _add_img_tb(self, val_visual, wrt_mode):
        val_img = []
        palette = self.val_loader.dataset.palette
        for imgs in val_visual:
            imgs = [self.restore_transform(i) if (isinstance(i, torch.Tensor) and len(i.shape) == 3)
                    else colorize_mask(i, palette) for i in imgs]
            imgs = [i.convert('RGB') for i in imgs]
            imgs = [self.viz_transform(i) for i in imgs]
            val_img.extend(imgs)
        val_img = torch.stack(val_img, 0)
        val_img = make_grid(val_img.cpu(), nrow=val_img.size(0) // len(val_visual), padding=5)
        self.writer.add_image(f'{wrt_mode}/inputs_targets_predictions', val_img, self.wrt_step)

    def _write_img_tb(self, input_l, target_l, input_ul, target_ul, outputs, epoch):
        outputs_l_np = outputs['sup_pred'].data.max(1)[1].cpu().numpy()
        targets_l_np = target_l.data.cpu().numpy()
        imgs = [[i.data.cpu(), j, k] for i, j, k in zip(input_l, outputs_l_np, targets_l_np)]
        self._add_img_tb(imgs, 'supervised')

class Trainer_Baseline(BaseTrainer):
    def     __init__(self, model, resume, config, supervised_loader, unsupervised_loader, iter_per_epoch,
                 val_loader=None, train_logger=None, gpu=None, test=False):
        super(Trainer_Baseline, self).__init__(model, resume, config, iter_per_epoch, train_logger, gpu=gpu, test=test)

        self.supervised_loader = supervised_loader
        self.unsupervised_loader = unsupervised_loader
        self.val_loader = val_loader
        self.iter_per_epoch = iter_per_epoch

        self.ignore_index = self.val_loader.dataset.ignore_index
        self.wrt_mode, self.wrt_step = 'train_', 0
        self.log_step = config['trainer'].get('log_per_iter', int(np.sqrt(self.val_loader.batch_size)))
        if config['trainer']['log_per_iter']:
            self.log_step = int(self.log_step / self.val_loader.batch_size) + 1

        self.num_classes = self.val_loader.dataset.num_classes
        self.mode = self.model.module.mode
        self.test = test

        # TRANSORMS FOR VISUALIZATION
        self.restore_transform = transforms.Compose([
            DeNormalize(self.val_loader.MEAN, self.val_loader.STD),
            transforms.ToPILImage()])
        self.viz_transform = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor()])

        self.start_time = time.time()

        self.epoch_start_unsup = config['model']['epoch_start_unsup']

    def _train_epoch(self, epoch):
        if self.gpu == 0:
            self.logger.info('\n')

        self.model.train()

        self.supervised_loader.train_sampler.set_epoch(epoch)
        self.unsupervised_loader.train_sampler.set_epoch(epoch)

        if self.mode == 'supervised':
            dataloader = iter(self.supervised_loader)
            tbar = tqdm(range(len(self.supervised_loader)), ncols=160)
        else:
            dataloader = iter(zip(cycle(self.supervised_loader), cycle(self.unsupervised_loader)))
            tbar = tqdm(range(self.iter_per_epoch), ncols=160)

        self._reset_metrics()

        for batch_idx in tbar:

            if self.mode == 'supervised':
                # (input_l, target_l), (input_ul, target_ul) = next(dataloader), (None, None)
                (input_l, target_l, image_id), (input_ul, target_ul, flip) = next(dataloader), (None, None, None)
                if target_l.dim()==4: target_l = target_l.squeeze(1)
            else:
                # (input_l, target_l), (input_ul, target_ul, flip) = next(dataloader)
                (input_l, target_l, image_id), (input_ul, target_ul, flip) = next(dataloader)
                if target_l.dim()==4: target_l = target_l.squeeze(1)
                if target_ul.dim()==4: target_ul = target_ul.squeeze(1)

            if self.mode == 'supervised':
                input_l, target_l = input_l.cuda(non_blocking=True), target_l.cuda(non_blocking=True)
                self.optimizer.zero_grad()
                total_loss, cur_losses, outputs = self.model(x_l=input_l, target_l=target_l, x_ul=input_ul,
                                                             curr_iter=batch_idx, target_ul=target_ul, epoch=epoch - 1)
            else:
                input_l, target_l = input_l.cuda(non_blocking=True), target_l.cuda(non_blocking=True)
                input_ul, target_ul = input_ul.cuda(non_blocking=True), target_ul.cuda(non_blocking=True)
                self.optimizer.zero_grad()
                kargs = {'gpu': self.gpu, 'flip': flip}
                total_loss, cur_losses, outputs = self.model(x_l=input_l, target_l=target_l, x_ul=input_ul,
                                                             curr_iter=batch_idx, target_ul=target_ul, epoch=epoch - 1,
                                                             **kargs)

            total_loss.backward()
            self.optimizer.step()

            if self.gpu == 0:
                if batch_idx % 100 == 0:
                    if self.mode == 'supervised':
                        self.logger.info("epoch:{}, L={:.3f}, Ls={:.3f}".
                                         format(epoch, total_loss, cur_losses['Ls']))
                    else:
                        if epoch-1 < self.epoch_start_unsup:
                            self.logger.info("epoch:{}, L={:.3f}, Ls={:.3f}".
                                             format(epoch, total_loss, cur_losses['Ls']))
                        else:
                            self.logger.info("epoch:{}, L={:.3f}, Ls={:.3f}, Lu={:.3f}, mIoU_l={:.2f}, ul={:.2f}".
                                             format(epoch, total_loss, cur_losses['Ls'], cur_losses['Lu'],
                                                    100*self.mIoU_l, 100*self.mIoU_ul))

            if batch_idx == 0:
                for key in cur_losses:
                    if not hasattr(self, key):
                        setattr(self, key, AverageMeter())

            # self._update_losses has already implemented synchronized DDP
            self._update_losses(cur_losses)

            self._compute_metrics(outputs, target_l, target_ul, epoch - 1)

            if self.gpu == 0:
                logs = self._log_values(cur_losses)

                if batch_idx % self.log_step == 0:
                    self.wrt_step = (epoch - 1) * len(self.unsupervised_loader) + batch_idx
                    self._write_scalars_tb(logs)

                descrip = 'T ({}) | '.format(epoch)
                for key in cur_losses:
                    descrip += key + ' {:.2f} '.format(getattr(self, key).average)
                descrip += 'mIoU_l {:.2f} ul {:.2f} |'.format(self.mIoU_l, self.mIoU_ul)
                tbar.set_description(descrip)

            del input_l, target_l, input_ul, target_ul
            del total_loss, cur_losses, outputs

            self.lr_scheduler.step(epoch=epoch - 1)

        return logs if self.gpu == 0 else None

    def _valid_epoch(self, epoch):
        if self.val_loader is None:
            if self.gpu == 0:
                self.logger.warning('Not data loader was passed for the validation step, No validation is performed !')
            return {}

        if self.gpu == 0:
            self.logger.info('\n###### EVALUATION ######')

        self.model.eval()
        self.wrt_mode = 'val'

        total_loss_val = AverageMeter()
        total_inter, total_union = 0, 0
        total_correct, total_label = 0, 0

        tbar = tqdm(self.val_loader, ncols=160)
        with torch.no_grad():
            # for batch_idx, (data, target) in enumerate(tbar):
            for batch_idx, (data, target, image_id) in enumerate(tbar):
                target, data = target.cuda(non_blocking=True), data.cuda(non_blocking=True)

                H, W = target.size(1), target.size(2)
                up_sizes = (ceil(H / 8) * 8, ceil(W / 8) * 8)
                pad_h, pad_w = up_sizes[0] - data.size(2), up_sizes[1] - data.size(3)
                data = F.pad(data, pad=(0, pad_w, 0, pad_h), mode='reflect')

                output = self.model(data)

                output = output[:, :, :H, :W]
                # LOSS
                loss = F.cross_entropy(output, target, ignore_index=self.ignore_index)

                total_loss_val.update(loss.item())

                # eval_metrics has already implemented DDP synchronized
                correct, labeled, inter, union = eval_metrics(output, target, self.num_classes, self.ignore_index)

                total_inter, total_union = total_inter + inter, total_union + union
                total_correct, total_label = total_correct + correct, total_label + labeled

                # PRINT INFO
                pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
                IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
                mIoU = IoU.mean()
                seg_metrics = {"Mean_IoU": np.round(100*mIoU,2), "Class_IoU": dict(zip(range(self.num_classes), np.round(100*IoU,2)))}

                if self.gpu == 0:
                    tbar.set_description('EVAL ({}) | Loss: {:.3f}, Mean IoU: {:.2f} |'.format(epoch, total_loss_val.average,100*mIoU))

            if self.gpu == 0:
                # self._add_img_tb(val_visual, 'val')

                # METRICS TO TENSORBOARD
                self.wrt_step = (epoch) * len(self.val_loader)
                self.writer.add_scalar(f'{self.wrt_mode}/loss', total_loss_val.average, self.wrt_step)
                for k, v in list(seg_metrics.items())[:-1]:
                    self.writer.add_scalar(f'{self.wrt_mode}/{k}', v, self.wrt_step)

            log = {
                'val_loss': total_loss_val.average,
                **seg_metrics
            }

        return log

    def _reset_metrics(self):
        self.loss_sup = AverageMeter()
        self.loss_unsup = AverageMeter()
        self.loss_weakly = AverageMeter()
        self.pair_wise = AverageMeter()
        self.total_inter_l, self.total_union_l = 0, 0
        self.total_correct_l, self.total_label_l = 0, 0
        self.total_inter_ul, self.total_union_ul = 0, 0
        self.total_correct_ul, self.total_label_ul = 0, 0
        self.mIoU_l, self.mIoU_ul = 0, 0
        self.pixel_acc_l, self.pixel_acc_ul = 0, 0
        self.class_iou_l, self.class_iou_ul = {}, {}
        self.ClsReg_F1_l, self.ClsReg_F1_ul = 0, 0

    def _update_losses(self, cur_losses):
        for key in cur_losses:
            loss = cur_losses[key]
            n = loss.numel()
            count = torch.tensor([n]).long().cuda()
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            mean = loss.sum() / n
            if self.gpu == 0:
                getattr(self, key).update(mean.item())

    def _compute_metrics(self, outputs, target_l, target_ul, epoch):
        seg_metrics_l = eval_metrics(outputs['sup_pred'], target_l, self.num_classes, self.ignore_index)

        if self.gpu == 0:
            self._update_seg_metrics(*seg_metrics_l, True)
            seg_metrics_l = self._get_seg_metrics(True)
            self.pixel_acc_l, self.mIoU_l, self.class_iou_l = seg_metrics_l.values()

        if 'unsup_pred' in outputs:
            seg_metrics_ul = eval_metrics(outputs['unsup_pred'], target_ul, self.num_classes, self.ignore_index)

            if self.gpu == 0:
                self._update_seg_metrics(*seg_metrics_ul, False)
                seg_metrics_ul = self._get_seg_metrics(False)
                self.pixel_acc_ul, self.mIoU_ul, self.class_iou_ul = seg_metrics_ul.values()

    def _update_seg_metrics(self, correct, labeled, inter, union, supervised=True):
        if supervised:
            self.total_correct_l += correct
            self.total_label_l += labeled
            self.total_inter_l += inter
            self.total_union_l += union
        else:
            self.total_correct_ul += correct
            self.total_label_ul += labeled
            self.total_inter_ul += inter
            self.total_union_ul += union

    def _get_seg_metrics(self, supervised=True):
        if supervised:
            pixAcc = 1.0 * self.total_correct_l / (np.spacing(1) + self.total_label_l)
            IoU = 1.0 * self.total_inter_l / (np.spacing(1) + self.total_union_l)
        else:
            pixAcc = 1.0 * self.total_correct_ul / (np.spacing(1) + self.total_label_ul)
            IoU = 1.0 * self.total_inter_ul / (np.spacing(1) + self.total_union_ul)
        mIoU = IoU.mean()
        return {
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Mean_IoU": np.round(mIoU, 4),
            "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 4)))
        }

    def _log_values(self, cur_losses):
        logs = {}
        if "Ls" in cur_losses.keys():
            logs['Ls'] = self.Ls.average
        if "Lu" in cur_losses.keys():
            logs['Lu'] = self.Lu.average
        if "loss_weakly" in cur_losses.keys():
            logs['loss_weakly'] = self.loss_weakly.average
        if "pair_wise" in cur_losses.keys():
            logs['pair_wise'] = self.pair_wise.average

        logs['mIoU_l'] = self.mIoU_l
        if self.mode == 'semi':
            logs['mIoU_ul'] = self.mIoU_ul
        return logs

    def _write_scalars_tb(self, logs):
        for k, v in logs.items():
            if 'class_iou' not in k: self.writer.add_scalar(f'train/{k}', v, self.wrt_step)
        for i, opt_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(f'train/Learning_rate_{i}', opt_group['lr'], self.wrt_step)

    def _add_img_tb(self, val_visual, wrt_mode):
        val_img = []
        palette = self.val_loader.dataset.palette
        for imgs in val_visual:
            imgs = [self.restore_transform(i) if (isinstance(i, torch.Tensor) and len(i.shape) == 3)
                    else colorize_mask(i, palette) for i in imgs]
            imgs = [i.convert('RGB') for i in imgs]
            imgs = [self.viz_transform(i) for i in imgs]
            val_img.extend(imgs)
        val_img = torch.stack(val_img, 0)
        val_img = make_grid(val_img.cpu(), nrow=val_img.size(0) // len(val_visual), padding=5)
        self.writer.add_image(f'{wrt_mode}/inputs_targets_predictions', val_img, self.wrt_step)

    def _write_img_tb(self, input_l, target_l, input_ul, target_ul, outputs, epoch):
        outputs_l_np = outputs['sup_pred'].data.max(1)[1].cpu().numpy()
        targets_l_np = target_l.data.cpu().numpy()
        imgs = [[i.data.cpu(), j, k] for i, j, k in zip(input_l, outputs_l_np, targets_l_np)]
        self._add_img_tb(imgs, 'supervised')

class Trainer_Ours(BaseTrainer):
    def __init__(self, model, resume, config, supervised_loader, unsupervised_loader, iter_per_epoch,
                 val_loader=None, train_logger=None, gpu=None, test=False):
        super(Trainer_Ours, self).__init__(model, resume, config, iter_per_epoch, train_logger, gpu=gpu, test=test)

        self.supervised_loader = supervised_loader
        self.unsupervised_loader = unsupervised_loader
        self.val_loader = val_loader
        self.iter_per_epoch = iter_per_epoch

        self.ignore_index = self.val_loader.dataset.ignore_index
        self.wrt_mode, self.wrt_step = 'train_', 0
        self.log_step = config['trainer'].get('log_per_iter', int(np.sqrt(self.val_loader.batch_size)))
        if config['trainer']['log_per_iter']:
            self.log_step = int(self.log_step / self.val_loader.batch_size) + 1

        self.num_classes = self.val_loader.dataset.num_classes
        self.mode = self.model.module.mode
        self.test = test

        # TRANSORMS FOR VISUALIZATION
        self.restore_transform = transforms.Compose([
            DeNormalize(self.val_loader.MEAN, self.val_loader.STD),
            transforms.ToPILImage()])
        self.viz_transform = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor()])

        self.start_time = time.time()

        self.epoch_start_unsup = config['model']['epoch_start_unsup']
        self.criterion_Focal = FocalLoss2d
        self.total_epoch = config['trainer']['epochs']

        

    def _train_epoch(self, epoch):
        if self.gpu == 0:
            self.logger.info('\n')

        self.model.train()

        self.supervised_loader.train_sampler.set_epoch(epoch)
        self.unsupervised_loader.train_sampler.set_epoch(epoch)

        if self.mode == 'supervised':
            dataloader = iter(self.supervised_loader)
            tbar = tqdm(range(len(self.supervised_loader)), ncols=160)
        else:
            dataloader = iter(zip(cycle(self.supervised_loader), cycle(self.unsupervised_loader)))
            tbar = tqdm(range(self.iter_per_epoch), ncols=160)

        self._reset_metrics()

        if self.gpu == 0:
            total_conf_bias_ce_T = torch.zeros(49).cuda()
            total_conf_bias_ce_F = torch.zeros(49).cuda()
            total_conf_bias_fl_T = torch.zeros(49).cuda()
            total_conf_bias_fl_F = torch.zeros(49).cuda()

            total_count_ce_T = torch.zeros(49).cuda()
            total_count_ce_F = torch.zeros(49).cuda()
            total_count_fl_T = torch.zeros(49).cuda()
            total_count_fl_F = torch.zeros(49).cuda()

            total_conf_bias_FG_ce_T = torch.zeros(49).cuda()
            total_conf_bias_FG_ce_F = torch.zeros(49).cuda()
            total_conf_bias_FG_fl_T = torch.zeros(49).cuda()
            total_conf_bias_FG_fl_F = torch.zeros(49).cuda()

            total_count_FG_ce_T = torch.zeros(49).cuda()
            total_count_FG_ce_F = torch.zeros(49).cuda()
            
            total_count_FG_fl_T = torch.zeros(49).cuda()
            total_count_FG_fl_F = torch.zeros(49).cuda()

            total_ece_T_ce = torch.zeros(49).cuda()
            total_ece_F_ce = torch.zeros(49).cuda()
            total_ece_T_fl = torch.zeros(49).cuda()
            total_ece_F_fl = torch.zeros(49).cuda()

            total_ece_T_FG_ce = torch.zeros(49).cuda()
            total_ece_F_FG_ce = torch.zeros(49).cuda()
            total_ece_T_FG_fl = torch.zeros(49).cuda()
            total_ece_F_FG_fl = torch.zeros(49).cuda()



            total_loss_ce = torch.zeros(2).cuda()
            total_loss_fl = torch.zeros(2).cuda()

            total_count = 0
            total_count_ce = 0
            total_count_fl = 0
            
            total_ce_cor, total_ce_wrg = 0, 0
            total_fl_cor, total_fl_wrg = 0, 0

        
        for batch_idx in tbar:
            
            if self.mode == 'supervised':
                # (input_l, target_l, image_id), (input_ul, target_ul) = next(dataloader), (None, None)
                (input_l, target_l, image_id), (input_ul, target_ul, flip) = next(dataloader), (None, None, None)
                if target_l.dim()==4: target_l = target_l.squeeze(1)
            else:
                # (input_l, target_l, image_id), (input_ul, target_ul, flip) = next(dataloader)
                (input_l, target_l, image_id), (input_ul, target_ul, flip) = next(dataloader)
                if target_l.dim()==4: target_l = target_l.squeeze(1)
                if target_ul.dim()==4: target_ul = target_ul.squeeze(1)

            if self.mode == 'supervised':
                input_l, target_l = input_l.cuda(non_blocking=True), target_l.cuda(non_blocking=True)
                self.optimizer.zero_grad()
                total_loss, cur_losses, outputs = self.model(x_l=input_l, target_l=target_l, x_ul=input_ul,
                                                                curr_iter=batch_idx, target_ul=target_ul, epoch=epoch - 1)
            else:
                input_l, target_l = input_l.cuda(non_blocking=True), target_l.cuda(non_blocking=True)
                input_ul, target_ul = input_ul.cuda(non_blocking=True), target_ul.cuda(non_blocking=True)
                self.optimizer.zero_grad()
                kargs = {'gpu': self.gpu, 'flip': flip}
                total_loss, cur_losses, outputs = self.model(x_l=input_l, target_l=target_l, x_ul=input_ul,
                                                                curr_iter=batch_idx, target_ul=target_ul, epoch=epoch - 1, total_epoch = self.total_epoch,
                                                                **kargs)
            total_loss.backward()
            self.optimizer.step()
                
            if batch_idx == 0:
                for key in cur_losses:
                    if not hasattr(self, key):
                        setattr(self, key, AverageMeter())

            self._update_losses(cur_losses)
            self._compute_metrics(outputs, target_l, target_ul, epoch - 1)

            if self.gpu == 0:
                if batch_idx % 20 == 0:
                    if self.mode == 'supervised':
                        self.logger.info("epoch:{}, L={:.3f}, Ls={:.3f}".
                                         format(epoch, total_loss, cur_losses['Ls']))
                    else:
                        if epoch -1 < self.epoch_start_unsup:
                            self.logger.info("epoch:{}, L={:.3f}, Ls={:.3f}".
                                             format(epoch, total_loss, cur_losses['Ls']))
                        else:
                            self.logger.info("epoch:{}, L={:.3f}, Ls={:.3f}, Lu_reg={:.3f}".
                                             format(epoch, total_loss, cur_losses['Ls'],
                                                    cur_losses['Lu_reg'], ))

            

            if self.gpu == 0:
                logs = self._log_values(cur_losses)

                
                
                ce_mask = outputs['ce_mask']
                fl_mask = outputs['fl_mask']

                b,h,w = ce_mask.shape

                total_count += (b*h*w)
                total_count_ce += torch.sum(ce_mask)
                total_count_fl += torch.sum(fl_mask)
                
                seg_ce_w = outputs['seg_ce_w']
                seg_fl_w = outputs['seg_fl_w']

                seg_ce_s = outputs['seg_ce_s']
                seg_fl_s = outputs['seg_fl_s']
                
                ps_ce_w = outputs['ps_ce']
                ps_fl_w = outputs['ps_fl']
                prob_ce_w = seg_ce_w.max(1)[0]
                prob_fl_w = seg_fl_w.max(1)[0]

                ps_ce_w_full = seg_ce_w.max(1)[1]
                ps_fl_w_full = seg_fl_w.max(1)[1]
                
                ignore_index = self.ignore_index

                hs,ws = ps_ce_w.shape[-2:]
                target_ul_s = outputs['target_ul']
                target_ul_s = target_ul_s.view(-1,hs,ws).long()
                ce_cor = (ps_fl_w==target_ul_s) * (ps_fl_w!=ignore_index)
                ce_wrg = (ps_fl_w!=target_ul_s) * (ps_fl_w!=ignore_index) * (target_ul_s!=ignore_index)


                conf_range = torch.Tensor(list(range(50))).cuda()
                conf_range = conf_range/50


                
                ce_loss = outputs['ce_loss']
                fl_loss = outputs['fl_loss']

                ce_loss_cor = torch.sum(ce_loss*ce_cor)
                ce_loss_wrg = torch.sum(ce_loss*ce_wrg)

                total_ce_cor += ce_loss_cor
                total_ce_wrg += ce_loss_wrg

                
                pos_ece_T_ce = (ps_ce_w_full==target_ul_s) * (target_ul_s!=255)
                pos_ece_F_ce = (ps_ce_w_full!=target_ul_s) * (target_ul_s!=255)

                pos_ece_T_fl = (ps_fl_w_full==target_ul_s) * (target_ul_s!=255)
                pos_ece_F_fl = (ps_fl_w_full!=target_ul_s) * (target_ul_s!=255)

                pos_ece_T_FG_ce = (ps_ce_w_full==target_ul_s) * (ps_ce_w_full!=0) * (target_ul_s!=255)
                pos_ece_F_FG_ce = (ps_ce_w_full!=target_ul_s) * (ps_ce_w_full!=0) * (target_ul_s!=255)

                pos_ece_T_fl = (ps_fl_w_full==target_ul_s) * (target_ul_s!=255)
                pos_ece_F_fl = (ps_fl_w_full!=target_ul_s) * (target_ul_s!=255)

                pos_ece_T_FG_fl = (ps_fl_w_full==target_ul_s) * (ps_fl_w_full!=0) * (target_ul_s!=255)
                pos_ece_F_FG_fl = (ps_fl_w_full!=target_ul_s) * (ps_fl_w_full!=0) * (target_ul_s!=255)



                pos_mask_h_T_ce = (ps_fl_w==target_ul_s) * (ps_fl_w!=255)
                pos_mask_h_F_ce = (ps_fl_w!=target_ul_s) * (target_ul_s!=255) * (ps_fl_w!=255)
                
                pos_mask_h_FG_T_ce = (ps_fl_w==target_ul_s) * (target_ul_s!=0) * (ps_fl_w!=255)
                pos_mask_h_FG_F_ce = (ps_fl_w!=target_ul_s) * (target_ul_s!=255) * (target_ul_s!=0) * (ps_fl_w!=255)

                pos_mask_h_T_fl = (ps_ce_w==target_ul_s) * (ps_ce_w!=255)
                pos_mask_h_F_fl = (ps_ce_w!=target_ul_s) * (target_ul_s!=255) * (ps_ce_w!=255)

                pos_mask_h_FG_T_fl = (ps_ce_w==target_ul_s) * (target_ul_s!=0) * (ps_ce_w!=255)
                pos_mask_h_FG_F_fl = (ps_ce_w!=target_ul_s) * (target_ul_s!=255) * (target_ul_s!=0) * (ps_ce_w!=255)


                logit_ps_ce = seg_ce_w.max(1)[0]
                logit_ps_fl = seg_fl_w.max(1)[0]

                

                logit_ece_T_ce = logit_ps_ce*pos_ece_T_ce
                logit_ece_F_ce = logit_ps_ce*pos_ece_F_ce

                logit_ece_T_fl = logit_ps_fl*pos_ece_T_fl
                logit_ece_F_fl = logit_ps_fl*pos_ece_F_fl

                logit_ece_T_FG_ce = logit_ps_ce*pos_ece_T_FG_ce
                logit_ece_F_FG_ce = logit_ps_ce*pos_ece_F_FG_ce

                logit_ece_T_FG_fl = logit_ps_fl*pos_ece_T_FG_fl
                logit_ece_F_FG_fl = logit_ps_fl*pos_ece_F_FG_fl

                logit_ps_T_ce = logit_ps_ce*pos_mask_h_T_fl
                logit_ps_F_ce = logit_ps_ce*pos_mask_h_F_fl

                logit_ps_T_fl = logit_ps_fl*pos_mask_h_T_fl
                logit_ps_F_fl = logit_ps_fl*pos_mask_h_F_fl

                logit_ps_FG_T_ce = logit_ps_ce*pos_mask_h_FG_T_ce
                logit_ps_FG_F_ce = logit_ps_ce*pos_mask_h_FG_F_ce

                logit_ps_FG_T_fl = logit_ps_fl*pos_mask_h_FG_T_fl
                logit_ps_FG_F_fl = logit_ps_fl*pos_mask_h_FG_F_fl

                logit_ps_T_ce_count = logit_ps_ce*pos_mask_h_T_fl
                logit_ps_F_ce_count = logit_ps_ce*pos_mask_h_F_fl

                logit_ps_T_fl_count = logit_ps_fl*pos_mask_h_T_ce
                logit_ps_F_fl_count = logit_ps_fl*pos_mask_h_F_ce

                
                for i in range(49):
                    r_min = conf_range[i]
                    r_max = conf_range[i+1]
                    
                    ece_ce_T = (logit_ece_T_ce >= r_min) * (logit_ece_T_ce < r_max)
                    ece_ce_F = (logit_ece_F_ce >= r_min) * (logit_ece_F_ce < r_max)

                    ece_fl_T = (logit_ece_T_fl >= r_min) * (logit_ece_T_fl < r_max)
                    ece_fl_F = (logit_ece_F_fl >= r_min) * (logit_ece_F_fl < r_max)

                    
                    ece_ce_T_FG = (logit_ece_T_FG_ce >= r_min) * (logit_ece_T_FG_ce < r_max)
                    ece_ce_F_FG = (logit_ece_F_FG_ce >= r_min) * (logit_ece_F_FG_ce < r_max)

                    ece_fl_T_FG = (logit_ece_T_FG_fl >= r_min) * (logit_ece_T_FG_fl < r_max)
                    ece_fl_F_FG = (logit_ece_F_FG_fl >= r_min) * (logit_ece_F_FG_fl < r_max)

                    total_ece_T_ce[i] += torch.sum(ece_ce_T)
                    total_ece_F_ce[i] += torch.sum(ece_ce_F)

                    total_ece_T_fl[i] += torch.sum(ece_fl_T)
                    total_ece_F_fl[i] += torch.sum(ece_fl_F)

                    total_ece_T_FG_ce[i] += torch.sum(ece_ce_T_FG)
                    total_ece_F_FG_ce[i] += torch.sum(ece_ce_F_FG)

                    total_ece_T_FG_fl[i] += torch.sum(ece_fl_T_FG)
                    total_ece_F_FG_fl[i] += torch.sum(ece_fl_F_FG)

                    use_m_ce_T = (logit_ps_T_ce >= r_min) * (logit_ps_T_ce < r_max)
                    use_m_ce_F = (logit_ps_F_ce >= r_min) * (logit_ps_F_ce < r_max)

                    use_m_fl_T = (logit_ps_T_fl >= r_min) * (logit_ps_T_fl < r_max)
                    use_m_fl_F = (logit_ps_F_fl >= r_min) * (logit_ps_F_fl < r_max)

                    use_m_ce_T_count = (logit_ps_T_ce_count >= r_min) * (logit_ps_T_ce_count < r_max)
                    use_m_ce_F_count = (logit_ps_F_ce_count >= r_min) * (logit_ps_F_ce_count < r_max)

                    use_m_fl_T_count = (logit_ps_T_fl_count >= r_min) * (logit_ps_T_fl_count < r_max)
                    use_m_fl_F_count = (logit_ps_F_fl_count >= r_min) * (logit_ps_F_fl_count < r_max)

                    if r_min > 0.75:
                        #print("Hight",(torch.sum(ce_loss * use_m_ce_T) + torch.sum(ce_loss * use_m_ce_F)) )
                        total_loss_ce[1] = total_loss_ce[1] + (torch.sum(ce_loss * use_m_ce_T) + torch.sum(ce_loss * use_m_ce_F)) 
                        total_loss_fl[1] = total_loss_fl[1] + (torch.sum(fl_loss * use_m_fl_T) + torch.sum(fl_loss * use_m_fl_F)) 
                    else:
                        #print("Low",(torch.sum(ce_loss * use_m_ce_T) + torch.sum(ce_loss * use_m_ce_F)) )
                        total_loss_ce[0] = total_loss_ce[0] + (torch.sum(ce_loss * use_m_ce_T) + torch.sum(ce_loss * use_m_ce_F))
                        total_loss_fl[0] = total_loss_fl[0] + (torch.sum(fl_loss * use_m_fl_T) + torch.sum(fl_loss * use_m_fl_F))  
                    total_conf_bias_ce_T[i] += torch.sum(ce_loss * use_m_ce_T)
                    total_conf_bias_ce_F[i] += torch.sum(ce_loss * use_m_ce_F)

                    total_conf_bias_fl_T[i] += torch.sum(fl_loss * use_m_fl_T)
                    total_conf_bias_fl_F[i] += torch.sum(fl_loss * use_m_fl_F)

                    total_count_ce_T[i] += torch.sum(use_m_ce_T_count)
                    total_count_ce_F[i] += torch.sum(use_m_ce_F_count)

                    total_count_fl_T[i] += torch.sum(use_m_fl_T_count)
                    total_count_fl_F[i] += torch.sum(use_m_fl_F_count)

                    use_m_FG_ce_T = (logit_ps_FG_T_ce >= r_min) * (logit_ps_FG_T_ce < r_max)
                    use_m_FG_ce_F = (logit_ps_FG_F_ce >= r_min) * (logit_ps_FG_F_ce < r_max)

                    use_m_FG_fl_T = (logit_ps_FG_T_fl >= r_min) * (logit_ps_FG_T_fl < r_max)
                    use_m_FG_fl_F = (logit_ps_FG_F_fl >= r_min) * (logit_ps_FG_F_fl < r_max)

                    total_conf_bias_FG_ce_T[i] += torch.sum(ce_loss * use_m_FG_ce_T)
                    total_conf_bias_FG_ce_F[i] += torch.sum(ce_loss * use_m_FG_ce_F)

                    total_conf_bias_FG_fl_T[i] += torch.sum(fl_loss * use_m_FG_fl_T)
                    total_conf_bias_FG_fl_F[i] += torch.sum(fl_loss * use_m_FG_fl_F)

                    total_count_FG_ce_T[i] += torch.sum(use_m_FG_ce_T)
                    total_count_FG_ce_F[i] += torch.sum(use_m_FG_ce_F)

                    total_count_FG_fl_T[i] += torch.sum(use_m_FG_fl_T)
                    total_count_FG_fl_F[i] += torch.sum(use_m_FG_fl_F)

                if batch_idx % self.log_step == 0:
                    self.wrt_step = (epoch - 1) * len(self.unsupervised_loader) + batch_idx
                    self._write_scalars_tb(logs)
                    

                    


                descrip = 'T ({}) | '.format(epoch)
                for key in cur_losses:
                    descrip += key + ' {:.2f} '.format(getattr(self, key).average)
                descrip += 'mIoU_l {:.2f} ul {:.2f} |'.format(100*self.mIoU_l, 100*self.mIoU_ul)
                tbar.set_description(descrip)

            
            del input_l, target_l, input_ul, target_ul
            del total_loss, cur_losses, outputs

            self.lr_scheduler.step(epoch=epoch - 1)
        
        if self.gpu == 0:
            
            self.writer.add_scalar(f'train/ce_ratio', 100*total_count_ce/total_count, epoch)
            self.writer.add_scalar(f'train/fl_ratio', 100*total_count_fl/total_count, epoch)

            self.writer.add_scalar(f'train/loss_ce_cor_ratio', total_ce_cor / (1e-5+total_ce_cor+total_ce_wrg), epoch)

            self.writer.add_scalar(f'train/ce_high_ratio', 100*total_loss_ce[1]/(total_loss_ce[0]+total_loss_ce[1]), epoch)
            self.writer.add_scalar(f'train/fl_high_ratio', 100*total_loss_fl[1]/(total_loss_fl[0]+total_loss_fl[1]), epoch)
            

            conf_range = torch.Tensor(list(range(48)))
            conf_range = conf_range/48

            ece_acc_ce = total_ece_T_ce / (0.0001 + total_ece_T_ce + total_ece_F_ce)
            ece_acc_fl = total_ece_T_fl / (0.0001 + total_ece_T_fl + total_ece_F_fl)

            total_ece_T_FG_ce

            ece_acc_FG_ce = total_ece_T_FG_ce / (0.0001 + total_ece_T_FG_ce + total_ece_F_FG_ce)
            ece_acc_FG_fl = total_ece_T_FG_fl / (0.0001 + total_ece_T_FG_fl + total_ece_F_FG_fl)

            ece_acc_ce = ece_acc_ce.detach().cpu()
            ece_acc_fl = ece_acc_fl.detach().cpu()

            ece_acc_FG_ce = ece_acc_FG_ce.detach().cpu()
            ece_acc_FG_fl = ece_acc_FG_fl.detach().cpu()

            acc_line = list(range(48))
            acc_line = torch.FloatTensor(acc_line)
            acc_line = acc_line/48

            ece_ce = torch.sum(torch.abs(ece_acc_ce[1:] - acc_line))
            ece_fl = torch.sum(torch.abs(ece_acc_fl[1:] - acc_line))

            ece_FG_ce = torch.sum(torch.abs(ece_acc_FG_ce[1:] - acc_line))
            ece_FG_fl = torch.sum(torch.abs(ece_acc_FG_fl[1:] - acc_line))

            count_ce = (total_ece_T_ce[1:] + total_ece_F_ce[1:]) / torch.sum(total_ece_T_ce[1:] + total_ece_F_ce[1:])
            count_fl = (total_ece_T_fl[1:] + total_ece_F_fl[1:]) / torch.sum(total_ece_T_fl[1:] + total_ece_F_fl[1:])

            count_ce = count_ce / torch.max(count_ce)
            count_fl = count_fl / torch.max(count_fl)

            count_FG_ce = (total_ece_T_FG_ce[1:] + total_ece_F_FG_ce[1:]) / torch.sum(total_ece_T_FG_ce[1:] + total_ece_F_FG_ce[1:])
            count_FG_fl = (total_ece_T_FG_fl[1:] + total_ece_F_FG_fl[1:]) / torch.sum(total_ece_T_FG_fl[1:] + total_ece_F_FG_fl[1:])

            count_FG_ce = count_FG_ce / torch.max(count_FG_ce)
            count_FG_fl = count_FG_fl / torch.max(count_FG_fl)

            
            plt.cla()
            fig = plt.figure(figsize=(8,8)) ## Figure 생성 
            ax1 = fig.add_subplot()

            ax1.bar(conf_range.detach().cpu().numpy(), ece_acc_ce[1:].detach().cpu().numpy(),width=0.02, label='CE:{:.2f}'.format(ece_ce))
            ax1.bar(conf_range.detach().cpu().numpy(), ece_acc_fl[1:].detach().cpu().numpy(),width=0.02,alpha=0.5, label='FL:{:.2f}'.format(ece_fl))
            plt.legend()

            ax2 = ax1.twinx()

            ax2.plot(conf_range.detach().cpu().numpy(), acc_line.detach().cpu().numpy())
            ax2.plot(conf_range.detach().cpu().numpy(), count_ce.detach().cpu().numpy(),'b')
            ax2.plot(conf_range.detach().cpu().numpy(), count_fl.detach().cpu().numpy(),'r')
            plt.savefig('temp/ece/{}_total.png'.format(epoch), dpi=300)

            plt.cla()
            fig = plt.figure(figsize=(8,8)) ## Figure 생성 
            ax1 = fig.add_subplot()
            
            ax1.bar(conf_range.detach().cpu().numpy(), ece_acc_FG_ce[1:].detach().cpu().numpy(),width=0.02, label='CE:{:.2f}'.format(ece_FG_ce))
            ax1.bar(conf_range.detach().cpu().numpy(), ece_acc_FG_fl[1:].detach().cpu().numpy(),width=0.02,alpha=0.5, label='FL:{:.2f}'.format(ece_FG_fl))
            plt.legend()

            ax2 = ax1.twinx()

            ax2.plot(conf_range.detach().cpu().numpy(), acc_line.detach().cpu().numpy())
            ax2.plot(conf_range.detach().cpu().numpy(), count_FG_ce.detach().cpu().numpy(),'b')
            ax2.plot(conf_range.detach().cpu().numpy(), count_FG_fl.detach().cpu().numpy(),'r')
            plt.savefig('temp/ece/{}_FG.png'.format(epoch), dpi=300)
            

            plt.cla()
            plt.bar(conf_range.detach().cpu().numpy(), total_conf_bias_fl_T[1:].detach().cpu().numpy(),width=0.02)
            plt.bar(conf_range.detach().cpu().numpy(), total_conf_bias_fl_F[1:].detach().cpu().numpy(),width=0.02,alpha=0.5)
            plt.savefig('temp/loss_dist/{}_fl.png'.format(epoch), dpi=300)
            plt.cla()
            plt.bar(conf_range.detach().cpu().numpy(), total_count_ce_T[1:].detach().cpu().numpy(),width=0.02)
            plt.bar(conf_range.detach().cpu().numpy(), total_count_ce_F[1:].detach().cpu().numpy(),width=0.02,alpha=0.5)
            plt.savefig('temp/conf_dist/{}_ce.png'.format(epoch), dpi=300)

            plt.cla()
            plt.bar(conf_range.detach().cpu().numpy(), total_count_fl_T[1:].detach().cpu().numpy(),width=0.02)
            plt.bar(conf_range.detach().cpu().numpy(), total_count_fl_F[1:].detach().cpu().numpy(),width=0.02,alpha=0.5)
            plt.savefig('temp/conf_dist/{}_fl.png'.format(epoch), dpi=300)

            plt.cla()
            plt.bar(conf_range.detach().cpu().numpy(), total_conf_bias_FG_ce_T[1:].detach().cpu().numpy(),width=0.02)
            plt.bar(conf_range.detach().cpu().numpy(), total_conf_bias_FG_ce_F[1:].detach().cpu().numpy(),width=0.02,alpha=0.5)
            plt.savefig('temp/loss_dist/{}_ce_FG.png'.format(epoch), dpi=300)

            plt.cla()
            plt.bar(conf_range.detach().cpu().numpy(), total_conf_bias_FG_fl_T[1:].detach().cpu().numpy(),width=0.02)
            plt.bar(conf_range.detach().cpu().numpy(), total_conf_bias_FG_fl_F[1:].detach().cpu().numpy(),width=0.02,alpha=0.5)
            plt.savefig('temp/loss_dist/{}_fl_FG.png'.format(epoch), dpi=300)

            plt.cla()
            plt.bar(conf_range.detach().cpu().numpy(), total_count_FG_ce_T[1:].detach().cpu().numpy(),width=0.02)
            plt.bar(conf_range.detach().cpu().numpy(), total_count_FG_ce_F[1:].detach().cpu().numpy(),width=0.02,alpha=0.5)
            plt.savefig('temp/conf_dist/{}_ce_FG.png'.format(epoch), dpi=300)

            plt.cla()
            plt.bar(conf_range.detach().cpu().numpy(), total_count_FG_fl_T[1:].detach().cpu().numpy(),width=0.02)
            plt.bar(conf_range.detach().cpu().numpy(), total_count_FG_fl_F[1:].detach().cpu().numpy(),width=0.02,alpha=0.5)
            plt.savefig('temp/conf_dist/{}_fl_FG.png'.format(epoch), dpi=300)

        return logs if self.gpu == 0 else None

    def _valid_epoch(self, epoch):
        if self.val_loader is None:
            if self.gpu == 0:
                self.logger.warning('Not data loader was passed for the validation step, No validation is performed !')
            return {}

        if self.gpu == 0:
            self.logger.info('\n###### EVALUATION ######')

        self.model.eval()
        self.wrt_mode = 'val'

        total_loss_val_ce = AverageMeter()
        total_loss_val_fl = AverageMeter()
        total_inter_ce, total_union_ce = 0, 0
        total_inter_fl, total_union_fl = 0, 0

        total_ce_cor, total_ce_wrg = 0, 0
        total_fl_cor, total_fl_wrg = 0, 0

        total_ce_weight_cor, total_ce_weight_wrg = 0, 0


        total_inter_ce_pseudo, total_union_ce_pseudo = 0, 0
        total_inter_fl_pseudo, total_union_fl_pseudo = 0, 0

        total_inter_ce_pseudo_N, total_union_ce_pseudo_N = 0, 0
        total_inter_fl_pseudo_N, total_union_fl_pseudo_N = 0, 0

        total_correct_ce, total_label_ce = 0, 0
        total_correct_fl, total_label_fl = 0, 0

        total_correct_ce_pseudo, total_label_ce_pseudo = 0, 0
        total_correct_fl_pseudo, total_label_fl_pseudo = 0, 0

        total_correct_ce_pseudo_N, total_label_ce_pseudo_N = 0, 0
        total_correct_fl_pseudo_N, total_label_fl_pseudo_N = 0, 0

        reli_count_ce, unreli_count_ce = 0, 0
        reli_count_fl, unreli_count_fl = 0, 0


        conf_hist_ce_T = torch.Tensor([]).cuda()
        conf_hist_ce_F = torch.Tensor([]).cuda()

        conf_hist_fl_T = torch.Tensor([]).cuda()
        conf_hist_fl_F = torch.Tensor([]).cuda()

        conf_hist_FG_ce_T = torch.Tensor([]).cuda()
        conf_hist_FG_ce_F = torch.Tensor([]).cuda()

        conf_hist_FG_fl_T = torch.Tensor([]).cuda()
        conf_hist_FG_fl_F = torch.Tensor([]).cuda()

        tbar = tqdm(self.val_loader, ncols=160)

        with torch.no_grad():
            # for batch_idx, (data, target) in enumerate(tbar):
            for batch_idx, (data, target, image_id) in enumerate(tbar):
                target, data = target.cuda(non_blocking=True), data.cuda(non_blocking=True)
                H, W = target.size(1), target.size(2)
                up_sizes = (ceil(H / 8) * 8, ceil(W / 8) * 8)
                pad_h, pad_w = up_sizes[0] - data.size(2), up_sizes[1] - data.size(3)
                data = F.pad(data, pad=(0, pad_w, 0, pad_h), mode='reflect')
                target_t = F.pad(target.float().unsqueeze(1), pad=(0, pad_w, 0, pad_h), mode='reflect')
                b,_,h,w=target_t.shape
                target_t = target_t.long().view(b,h,w)
                output_ce, output_fl = self.model(data,target_l=target_t,batch_idx=batch_idx,img_id=image_id, epoch=epoch)
                output_ce = output_ce[:, :, :H, :W]
                output_fl = output_fl[:, :, :H, :W]
                # LOSS
                loss_ce = F.cross_entropy(output_ce, target, ignore_index=self.ignore_index)
                total_loss_val_ce.update(loss_ce.item())
                
                seg_ce = F.softmax(output_ce, 1)
                seg_fl = F.softmax(output_fl, 1)

                prob_ce = seg_ce.max(1)[0]
                ps_ce = seg_ce.max(1)[1]

                prob_fl = seg_fl.max(1)[0]
                ps_fl = seg_fl.max(1)[1]

                ignore_index = self.model.module.ignore_index
                ps_fl[prob_fl<self.model.module.pos_thresh_value] = ignore_index
                ps_ce[prob_ce<self.model.module.pos_thresh_value] = ignore_index

                ce_cor = (ps_ce==target) * (ps_ce!=ignore_index)
                ce_wrg = (ps_ce!=target) * (ps_ce!=ignore_index) * (target!=ignore_index)

                ce_loss = F.cross_entropy(output_ce.detach(), ps_ce.detach().clone(), reduction='none',ignore_index=ignore_index)
                
                ce_loss_cor = torch.sum(ce_loss*ce_cor)
                ce_loss_wrg = torch.sum(ce_loss*ce_wrg)
                

                total_ce_cor += ce_loss_cor
                total_ce_wrg += ce_loss_wrg

                
                loss_fl = self.criterion_Focal(output_fl, target, ignore_index=self.ignore_index)
                total_loss_val_fl.update(loss_fl.item())

                # eval_metrics has already implemented DDP synchronized
                correct_ce, labeled_ce, inter_ce, union_ce = eval_metrics(output_ce, target, self.num_classes, self.ignore_index)
                correct_fl, labeled_fl, inter_fl, union_fl = eval_metrics(output_fl, target, self.num_classes, self.ignore_index)

                total_inter_ce, total_union_ce = total_inter_ce + inter_ce, total_union_ce + union_ce
                total_correct_ce, total_label_ce = total_correct_ce + correct_ce, total_label_ce + labeled_ce

                total_inter_fl, total_union_fl = total_inter_fl + inter_fl, total_union_fl + union_fl
                total_correct_fl, total_label_fl = total_correct_fl + correct_fl, total_label_fl + labeled_fl

                IoU_ce = 1.0 * total_inter_ce / (np.spacing(1) + total_union_ce)
                IoU_fl = 1.0 * total_inter_fl / (np.spacing(1) + total_union_fl)

                mIoU_ce = IoU_ce.mean()
                mIoU_fl = IoU_fl.mean()

                seg_metrics_ce = {"Mean_IoU": np.round(100*mIoU_ce,2), "Class_IoU": dict(zip(range(self.num_classes), np.round(100*IoU_ce,2)))}
                seg_metrics_fl = {"Mean_IoU": np.round(100*mIoU_fl,2), "Class_IoU": dict(zip(range(self.num_classes), np.round(100*IoU_fl,2)))}

                ###### eval pseudo label#####

                output_ce_sf = F.softmax(output_ce,dim=1)
                output_fl_sf = F.softmax(output_fl,dim=1)

                pseudo_logits_w = output_ce_sf.max(1)[0].detach()
                pseudo_label_w = output_ce_sf.max(1)[1].detach()

                pseudo_logits_w_F = output_fl_sf.max(1)[0].detach()
                pseudo_label_w_F = output_fl_sf.max(1)[1].detach()

                pos_mask_ce = (pseudo_label_w == pseudo_label_w_F)*(pseudo_logits_w > self.model.module.pos_thresh_value)
                #pos_mask_ce = (pseudo_label_w == pseudo_label_w_F)*(pseudo_logits_w > pseudo_logits_w_F)
                
                #pos_mask_fl = pos_mask_ce.clone()
                pos_mask_fl = (pseudo_label_w == pseudo_label_w_F)*(pseudo_logits_w_F > self.model.module.pos_thresh_value)

                reli_count_ce += torch.sum(pos_mask_ce)
                unreli_count_ce += torch.sum(torch.logical_not(pos_mask_ce))

                reli_count_fl += torch.sum(pos_mask_fl)
                unreli_count_fl += torch.sum(torch.logical_not(pos_mask_fl))

                target_ce_N = target.clone()
                target_ce_N[pos_mask_ce] = self.ignore_index

                target_fl_N = target.clone()
                target_fl_N[pos_mask_fl] = self.ignore_index

                target_ce = target.clone()
                target_ce[torch.logical_not(pos_mask_ce)] = self.ignore_index

                target_fl = target.clone()
                target_fl[torch.logical_not(pos_mask_fl)] = self.ignore_index
                
                correct_ce_pseudo, labeled_ce_pseudo, inter_ce_pseudo, union_ce_pseudo = eval_metrics(output_ce, target_ce, self.num_classes, self.ignore_index)
                correct_fl_pseudo, labeled_fl_pseudo, inter_fl_pseudo, union_fl_pseudo = eval_metrics(output_fl, target_fl, self.num_classes, self.ignore_index)
                
                total_inter_ce_pseudo, total_union_ce_pseudo = total_inter_ce_pseudo + inter_ce_pseudo, total_union_ce_pseudo + union_ce_pseudo
                total_correct_ce_pseudo, total_label_ce_pseudo = total_correct_ce_pseudo + correct_ce_pseudo, total_label_ce_pseudo + labeled_ce_pseudo

                total_inter_fl_pseudo, total_union_fl_pseudo = total_inter_fl_pseudo + inter_fl_pseudo, total_union_fl_pseudo + union_fl_pseudo
                total_correct_fl_pseudo, total_label_fl_pseudo = total_correct_fl_pseudo + correct_fl_pseudo, total_label_fl_pseudo + labeled_fl_pseudo
                
                IoU_ce_ps = 1.0 * total_inter_ce_pseudo / (np.spacing(1) + total_union_ce_pseudo)
                IoU_fl_ps = 1.0 * total_inter_fl_pseudo / (np.spacing(1) + total_union_fl_pseudo)

                mIoU_ce_ps = IoU_ce_ps.mean()
                mIoU_fl_ps = IoU_fl_ps.mean()

                seg_metrics_ce_ps_thr = {"Mean_IoU": np.round(100*mIoU_ce_ps,2), "Class_IoU": dict(zip(range(self.num_classes), np.round(100*IoU_ce_ps,2)))}
                seg_metrics_fl_ps_thr = {"Mean_IoU": np.round(100*mIoU_fl_ps,2), "Class_IoU": dict(zip(range(self.num_classes), np.round(100*IoU_fl_ps,2)))}
                
                correct_ce_pseudo_N, labeled_ce_pseudo_N, inter_ce_pseudo_N, union_ce_pseudo_N = eval_metrics(output_ce, target_ce_N, self.num_classes, self.ignore_index)
                correct_fl_pseudo_N, labeled_fl_pseudo_N, inter_fl_pseudo_N, union_fl_pseudo_N = eval_metrics(output_fl, target_fl_N, self.num_classes, self.ignore_index)
                
                total_inter_ce_pseudo_N, total_union_ce_pseudo_N = total_inter_ce_pseudo_N + inter_ce_pseudo_N, total_union_ce_pseudo_N + union_ce_pseudo_N
                total_correct_ce_pseudo_N, total_label_ce_pseudo_N = total_correct_ce_pseudo_N + correct_ce_pseudo_N, total_label_ce_pseudo_N + labeled_ce_pseudo_N

                total_inter_fl_pseudo_N, total_union_fl_pseudo_N = total_inter_fl_pseudo_N + inter_fl_pseudo_N, total_union_fl_pseudo_N + union_fl_pseudo_N
                total_correct_fl_pseudo_N, total_label_fl_pseudo_N = total_correct_fl_pseudo_N + correct_fl_pseudo_N, total_label_fl_pseudo_N + labeled_fl_pseudo_N
                
                IoU_ce_ps_N = 1.0 * total_inter_ce_pseudo_N / (np.spacing(1) + total_union_ce_pseudo_N)
                IoU_fl_ps_N = 1.0 * total_inter_fl_pseudo_N / (np.spacing(1) + total_union_fl_pseudo_N)

                mIoU_ce_ps_N = IoU_ce_ps_N.mean()
                mIoU_fl_ps_N = IoU_fl_ps_N.mean()

                seg_metrics_ce_ps_thr_N = {"Mean_IoU": np.round(100*mIoU_ce_ps_N,2), "Class_IoU": dict(zip(range(self.num_classes), np.round(100*IoU_ce_ps_N,2)))}
                seg_metrics_fl_ps_thr_N = {"Mean_IoU": np.round(100*mIoU_fl_ps_N,2), "Class_IoU": dict(zip(range(self.num_classes), np.round(100*IoU_fl_ps_N,2)))}
                
                ####pseudo hist vis####


                pos_mask_h_T_ce = (output_ce.max(1)[1]==target)
                pos_mask_h_F_ce = (output_ce.max(1)[1]!=target) * (target!=255)

                pos_mask_h_FG_T_ce = (output_ce.max(1)[1]==target) * (target!=0)
                pos_mask_h_FG_F_ce = (output_ce.max(1)[1]!=target) * (target!=255) * (target!=0)

                pos_mask_h_T_fl = (output_fl.max(1)[1]==target)
                pos_mask_h_F_fl = (output_fl.max(1)[1]!=target) * (target!=255)

                pos_mask_h_FG_T_fl = (output_fl.max(1)[1]==target) * (target!=0)
                pos_mask_h_FG_F_fl = (output_fl.max(1)[1]!=target) * (target!=255) * (target!=0)

                logit_ps_ce = output_ce_sf.max(1)[0]
                logit_ps_fl = output_fl_sf.max(1)[0]

                logit_ps_T_ce = logit_ps_ce[pos_mask_h_T_ce]
                logit_ps_F_ce = logit_ps_ce[pos_mask_h_F_ce]

                logit_ps_T_fl = logit_ps_fl[pos_mask_h_T_fl]
                logit_ps_F_fl = logit_ps_fl[pos_mask_h_F_fl]

                logit_ps_FG_T_ce = logit_ps_ce[pos_mask_h_FG_T_ce]
                logit_ps_FG_F_ce = logit_ps_ce[pos_mask_h_FG_F_ce]

                logit_ps_FG_T_fl = logit_ps_fl[pos_mask_h_FG_T_fl]
                logit_ps_FG_F_fl = logit_ps_fl[pos_mask_h_FG_F_fl]

                conf_hist_ce_T = torch.cat([conf_hist_ce_T, logit_ps_T_ce])
                conf_hist_ce_F = torch.cat([conf_hist_ce_F, logit_ps_F_ce])

                conf_hist_fl_T = torch.cat([conf_hist_fl_T, logit_ps_T_fl])
                conf_hist_fl_F = torch.cat([conf_hist_fl_F, logit_ps_F_fl])

                conf_hist_FG_ce_T = torch.cat([conf_hist_FG_ce_T, logit_ps_FG_T_ce])
                conf_hist_FG_ce_F = torch.cat([conf_hist_FG_ce_F, logit_ps_FG_F_ce])

                conf_hist_FG_fl_T = torch.cat([conf_hist_FG_fl_T, logit_ps_FG_T_fl])
                conf_hist_FG_fl_F = torch.cat([conf_hist_FG_fl_F, logit_ps_FG_F_fl])
                if self.gpu == 0:
                    tbar.set_description('EVAL ({}) | Loss: {:.3f}, Mean IoU: {:.2f} |'.format(epoch, total_loss_val_ce.average,100*mIoU_ce))
            if self.gpu == 0:

                self.wrt_step = (epoch) * len(self.val_loader)
                self.writer.add_scalar(f'{self.wrt_mode}/loss_ce', total_loss_val_ce.average, self.wrt_step)
                self.writer.add_scalar(f'{self.wrt_mode}/loss_fl', total_loss_val_fl.average, self.wrt_step)

                self.writer.add_scalar(f'{self.wrt_mode}/loss_ce_cor_ratio', total_ce_cor / (total_ce_cor+total_ce_wrg), self.wrt_step)
                
                self.writer.add_scalar(f'{self.wrt_mode}/Ratio_relable_CE', 100*(reli_count_ce)/(reli_count_ce+unreli_count_ce), self.wrt_step)
                self.writer.add_scalar(f'{self.wrt_mode}/Ratio_relable_FL', 100*(reli_count_fl)/(reli_count_fl+unreli_count_fl), self.wrt_step)
                for k, v in list(seg_metrics_ce.items())[:-1]:
                    self.writer.add_scalar(f'{self.wrt_mode}/{k}_ce', v, self.wrt_step)

                for k, v in list(seg_metrics_fl.items())[:-1]:
                    self.writer.add_scalar(f'{self.wrt_mode}/{k}_fl', v, self.wrt_step)

                for k, v in list(seg_metrics_ce_ps_thr_N.items())[:-1]:
                    self.writer.add_scalar(f'{self.wrt_mode}/{k}_ce_ps', v, self.wrt_step)

                for k, v in list(seg_metrics_fl_ps_thr_N.items())[:-1]:
                    self.writer.add_scalar(f'{self.wrt_mode}/{k}_fl_ps', v, self.wrt_step)

                for k, v in list(seg_metrics_ce_ps_thr.items())[:-1]:
                    self.writer.add_scalar(f'{self.wrt_mode}/{k}_ce_ps_thr', v, self.wrt_step)

                for k, v in list(seg_metrics_fl_ps_thr.items())[:-1]:
                    self.writer.add_scalar(f'{self.wrt_mode}/{k}_fl_ps_thr', v, self.wrt_step)
                """
                self.writer.add_histogram(f'{self.wrt_mode}/pseudo_hist_ce', conf_hist_ce_T.detach().cpu(), self.wrt_step)
                self.writer.add_histogram(f'{self.wrt_mode}/pseudo_hist_ce', conf_hist_ce_F.detach().cpu(), self.wrt_step+0.25)

                self.writer.add_histogram(f'{self.wrt_mode}/pseudo_hist_fl', conf_hist_fl_T.detach().cpu(), self.wrt_step)
                self.writer.add_histogram(f'{self.wrt_mode}/pseudo_hist_fl', conf_hist_fl_F.detach().cpu(), self.wrt_step+0.25)

                self.writer.add_histogram(f'{self.wrt_mode}/pseudo_hist_ce_FG', conf_hist_FG_ce_T.detach().cpu(), self.wrt_step)
                self.writer.add_histogram(f'{self.wrt_mode}/pseudo_hist_ce_FG', conf_hist_FG_ce_F.detach().cpu(), self.wrt_step+0.25)

                self.writer.add_histogram(f'{self.wrt_mode}/pseudo_hist_fl_FG', conf_hist_FG_fl_T.detach().cpu(), self.wrt_step)
                self.writer.add_histogram(f'{self.wrt_mode}/pseudo_hist_fl_FG', conf_hist_FG_fl_F.detach().cpu(), self.wrt_step+0.25)
                """
            log = {
                'val_loss': np.round(total_loss_val_ce.average,3),
                **seg_metrics_ce
            }
        return log

    def _reset_metrics(self):
        self.Ls = AverageMeter()
        self.Lu_reg = AverageMeter()
        self.Lu_sub = AverageMeter()
        self.total_inter_l, self.total_union_l = 0, 0
        self.total_correct_l, self.total_label_l = 0, 0
        self.total_inter_ul, self.total_union_ul = 0, 0
        self.total_correct_ul, self.total_label_ul = 0, 0
        self.mIoU_l, self.mIoU_ul = 0, 0
        self.mIoU_ul_reg = 0
        self.pixel_acc_l, self.pixel_acc_ul = 0, 0
        self.class_iou_l, self.class_iou_ul = {}, {}

    def _update_losses(self, cur_losses):
        for key in cur_losses:
            loss = cur_losses[key]
            n = loss.numel()
            count = torch.tensor([n]).long().cuda()
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            mean = loss.sum() / n
            if self.gpu == 0:
                getattr(self, key).update(mean.item())

    def _compute_metrics(self, outputs, target_l, target_ul, epoch):
        seg_metrics_l = eval_metrics(outputs['sup_pred'], target_l, self.num_classes, self.ignore_index)

        if self.gpu == 0:
            self._update_seg_metrics(*seg_metrics_l, True)
            seg_metrics_l = self._get_seg_metrics(True)
            self.pixel_acc_l, self.mIoU_l, self.class_iou_l = seg_metrics_l.values()

    def _update_seg_metrics(self, correct, labeled, inter, union, supervised=True):
        if supervised:
            self.total_correct_l += correct
            self.total_label_l += labeled
            self.total_inter_l += inter
            self.total_union_l += union
        else:
            self.total_correct_ul += correct
            self.total_label_ul += labeled
            self.total_inter_ul += inter
            self.total_union_ul += union

    def _get_seg_metrics(self, supervised=True):
        if supervised:
            pixAcc = 1.0 * self.total_correct_l / (np.spacing(1) + self.total_label_l)
            IoU = 1.0 * self.total_inter_l / (np.spacing(1) + self.total_union_l)
        else:
            pixAcc = 1.0 * self.total_correct_ul / (np.spacing(1) + self.total_label_ul)
            IoU = 1.0 * self.total_inter_ul / (np.spacing(1) + self.total_union_ul)
        mIoU = IoU.mean()
        return {"Pixel_Accuracy": pixAcc, "Mean_IoU": mIoU, "Class_IoU": dict(zip(range(self.num_classes), IoU))}

    def _log_values(self, cur_losses):
        logs = {}
        if "Ls" in cur_losses.keys():
            logs['Ls'] = self.Ls.average
        if "Ls_sub" in cur_losses.keys():
            logs['Ls_sub'] = self.Ls_sub.average
        if "Lu" in cur_losses.keys():
            logs['Lu'] = self.Lu.average
        if "Lu_reg" in cur_losses.keys():
            logs['Lu_reg'] = self.Lu_reg.average
        if "Lu_sub" in cur_losses.keys():
            logs['Lu_sub'] = self.Lu_sub.average
        logs['mIoU_l'] = self.mIoU_l
        if self.mode == 'semi':
            logs['mIoU_ul'] = self.mIoU_ul
            logs['mIoU_ul_reg'] = self.mIoU_ul_reg
        return logs

    def _write_scalars_tb(self, logs):
        for k, v in logs.items():
            if 'class_iou' not in k: self.writer.add_scalar(f'train/{k}', v, self.wrt_step)
        for i, opt_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(f'train/Learning_rate_{i}', opt_group['lr'], self.wrt_step)

class Trainer_USRN(BaseTrainer):
    def __init__(self, model, resume, config, supervised_loader, unsupervised_loader, iter_per_epoch,
                 val_loader=None, train_logger=None, gpu=None, test=False):
        super(Trainer_USRN, self).__init__(model, resume, config, iter_per_epoch, train_logger, gpu=gpu, test=test)

        self.supervised_loader = supervised_loader
        self.unsupervised_loader = unsupervised_loader
        self.val_loader = val_loader
        self.iter_per_epoch = iter_per_epoch

        self.ignore_index = self.val_loader.dataset.ignore_index
        self.wrt_mode, self.wrt_step = 'train_', 0
        self.log_step = config['trainer'].get('log_per_iter', int(np.sqrt(self.val_loader.batch_size)))
        if config['trainer']['log_per_iter']:
            self.log_step = int(self.log_step / self.val_loader.batch_size) + 1

        self.num_classes = self.val_loader.dataset.num_classes
        self.mode = self.model.module.mode
        self.test = test

        # TRANSORMS FOR VISUALIZATION
        self.restore_transform = transforms.Compose([
            DeNormalize(self.val_loader.MEAN, self.val_loader.STD),
            transforms.ToPILImage()])
        self.viz_transform = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor()])

        self.start_time = time.time()

        self.epoch_start_unsup = config['model']['epoch_start_unsup']

    def _train_epoch(self, epoch):
        if self.gpu == 0:
            self.logger.info('\n')

        self.model.train()

        self.supervised_loader.train_sampler.set_epoch(epoch)
        self.unsupervised_loader.train_sampler.set_epoch(epoch)

        if self.mode == 'supervised':
            dataloader = iter(self.supervised_loader)
            tbar = tqdm(range(len(self.supervised_loader)), ncols=160)
        else:
            dataloader = iter(zip(cycle(self.supervised_loader), cycle(self.unsupervised_loader)))
            tbar = tqdm(range(self.iter_per_epoch), ncols=160)

        self._reset_metrics()

        for batch_idx in tbar:

            if self.mode == 'supervised':
                # (input_l, target_l, image_id), (input_ul, target_ul) = next(dataloader), (None, None)
                (input_l, target_l, target_l_subcls, image_id), (input_ul, target_ul, flip) = next(dataloader), (None, None, None)
                if target_l.dim()==4: target_l = target_l.squeeze(1)
            else:
                # (input_l, target_l, image_id), (input_ul, target_ul, flip) = next(dataloader)
                (input_l, target_l, target_l_subcls, image_id), (input_ul, target_ul, flip) = next(dataloader)
                if target_l.dim()==4: target_l = target_l.squeeze(1)
                if target_ul.dim()==4: target_ul = target_ul.squeeze(1)

            if self.mode == 'supervised':
                input_l, target_l = input_l.cuda(non_blocking=True), target_l.cuda(non_blocking=True)
                self.optimizer.zero_grad()
                total_loss, cur_losses, outputs = self.model(x_l=input_l, target_l=target_l, target_l_subcls=target_l_subcls, x_ul=input_ul,
                                                             curr_iter=batch_idx, target_ul=target_ul, epoch=epoch - 1)
            else:
                input_l, target_l = input_l.cuda(non_blocking=True), target_l.cuda(non_blocking=True)
                target_l_subcls = target_l_subcls.cuda(non_blocking=True)
                input_ul, target_ul = input_ul.cuda(non_blocking=True), target_ul.cuda(non_blocking=True)
                self.optimizer.zero_grad()
                kargs = {'gpu': self.gpu, 'flip': flip}
                total_loss, cur_losses, outputs = self.model(x_l=input_l, target_l=target_l, target_l_subcls=target_l_subcls, x_ul=input_ul,
                                                             curr_iter=batch_idx, target_ul=target_ul, epoch=epoch - 1,
                                                             **kargs)
            total_loss.backward()
            self.optimizer.step()

            if batch_idx == 0:
                for key in cur_losses:
                    if not hasattr(self, key):
                        setattr(self, key, AverageMeter())

            self._update_losses(cur_losses)
            self._compute_metrics(outputs, target_l, target_ul, epoch - 1)

            if self.gpu == 0:
                if batch_idx % 20 == 0:
                    if self.mode == 'supervised':
                        self.logger.info("epoch:{}, L={:.3f}, Ls={:.3f}, Ls_sub={:.3f}".
                                         format(epoch, total_loss, cur_losses['Ls'], cur_losses['Ls_sub']))
                    else:
                        if epoch -1 < self.epoch_start_unsup:
                            self.logger.info("epoch:{}, L={:.3f}, Ls={:.3f}, Ls_sub={:.3f}".
                                             format(epoch, total_loss, cur_losses['Ls'], cur_losses['Ls_sub']))
                        else:
                            self.logger.info("epoch:{}, L={:.3f}, Ls={:.3f}, Ls_sub={:.3f}, Lu_reg={:.3f}, Lu_sub={:.3f}".
                                             format(epoch, total_loss, cur_losses['Ls'], cur_losses['Ls_sub'],
                                                    cur_losses['Lu_reg'], cur_losses['Lu_sub'], ))

            if self.gpu == 0:
                logs = self._log_values(cur_losses)

                if batch_idx % self.log_step == 0:
                    self.wrt_step = (epoch - 1) * len(self.unsupervised_loader) + batch_idx
                    self._write_scalars_tb(logs)

                descrip = 'T ({}) | '.format(epoch)
                for key in cur_losses:
                    descrip += key + ' {:.2f} '.format(getattr(self, key).average)
                descrip += 'mIoU_l {:.2f} ul {:.2f} |'.format(100*self.mIoU_l, 100*self.mIoU_ul)
                tbar.set_description(descrip)

            del input_l, target_l, input_ul, target_ul
            del total_loss, cur_losses, outputs

            self.lr_scheduler.step(epoch=epoch - 1)

        return logs if self.gpu == 0 else None

    def _valid_epoch(self, epoch):
        if self.val_loader is None:
            if self.gpu == 0:
                self.logger.warning('Not data loader was passed for the validation step, No validation is performed !')
            return {}

        if self.gpu == 0:
            self.logger.info('\n###### EVALUATION ######')

        self.model.eval()
        self.wrt_mode = 'val'

        total_loss_val = AverageMeter()
        total_inter, total_union = 0, 0
        total_correct, total_label = 0, 0

        tbar = tqdm(self.val_loader, ncols=160)
        with torch.no_grad():
            # for batch_idx, (data, target) in enumerate(tbar):
            for batch_idx, (data, target, image_id) in enumerate(tbar):
                target, data = target.cuda(non_blocking=True), data.cuda(non_blocking=True)
                H, W = target.size(1), target.size(2)
                up_sizes = (ceil(H / 8) * 8, ceil(W / 8) * 8)
                pad_h, pad_w = up_sizes[0] - data.size(2), up_sizes[1] - data.size(3)
                data = F.pad(data, pad=(0, pad_w, 0, pad_h), mode='reflect')
                output = self.model(data)
                output = output[:, :, :H, :W]
                # LOSS
                loss = F.cross_entropy(output, target, ignore_index=self.ignore_index)
                total_loss_val.update(loss.item())

                # eval_metrics has already implemented DDP synchronized
                correct, labeled, inter, union = eval_metrics(output, target, self.num_classes, self.ignore_index)

                total_inter, total_union = total_inter + inter, total_union + union
                total_correct, total_label = total_correct + correct, total_label + labeled

                IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
                mIoU = IoU.mean()
                seg_metrics = {"Mean_IoU": np.round(100*mIoU,2), "Class_IoU": dict(zip(range(self.num_classes), np.round(100*IoU,2)))}
                if self.gpu == 0:
                    tbar.set_description('EVAL ({}) | Loss: {:.3f}, Mean IoU: {:.2f} |'.format(epoch, total_loss_val.average,100*mIoU))
            if self.gpu == 0:
                self.wrt_step = (epoch) * len(self.val_loader)
                self.writer.add_scalar(f'{self.wrt_mode}/loss', total_loss_val.average, self.wrt_step)
                for k, v in list(seg_metrics.items())[:-1]:
                    self.writer.add_scalar(f'{self.wrt_mode}/{k}', v, self.wrt_step)

            log = {
                'val_loss': np.round(total_loss_val.average,3),
                **seg_metrics
            }
        return log

    def _reset_metrics(self):
        self.Ls = AverageMeter()
        self.Ls_sub = AverageMeter()
        self.Lu_reg = AverageMeter()
        self.Lu_sub = AverageMeter()
        self.total_inter_l, self.total_union_l = 0, 0
        self.total_correct_l, self.total_label_l = 0, 0
        self.total_inter_ul, self.total_union_ul = 0, 0
        self.total_correct_ul, self.total_label_ul = 0, 0
        self.mIoU_l, self.mIoU_ul = 0, 0
        self.mIoU_ul_reg = 0
        self.pixel_acc_l, self.pixel_acc_ul = 0, 0
        self.class_iou_l, self.class_iou_ul = {}, {}

    def _update_losses(self, cur_losses):
        for key in cur_losses:
            loss = cur_losses[key]
            n = loss.numel()
            count = torch.tensor([n]).long().cuda()
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            mean = loss.sum() / n
            if self.gpu == 0:
                getattr(self, key).update(mean.item())

    def _compute_metrics(self, outputs, target_l, target_ul, epoch):
        seg_metrics_l = eval_metrics(outputs['sup_pred'], target_l, self.num_classes, self.ignore_index)

        if self.gpu == 0:
            self._update_seg_metrics(*seg_metrics_l, True)
            seg_metrics_l = self._get_seg_metrics(True)
            self.pixel_acc_l, self.mIoU_l, self.class_iou_l = seg_metrics_l.values()

    def _update_seg_metrics(self, correct, labeled, inter, union, supervised=True):
        if supervised:
            self.total_correct_l += correct
            self.total_label_l += labeled
            self.total_inter_l += inter
            self.total_union_l += union
        else:
            self.total_correct_ul += correct
            self.total_label_ul += labeled
            self.total_inter_ul += inter
            self.total_union_ul += union

    def _get_seg_metrics(self, supervised=True):
        if supervised:
            pixAcc = 1.0 * self.total_correct_l / (np.spacing(1) + self.total_label_l)
            IoU = 1.0 * self.total_inter_l / (np.spacing(1) + self.total_union_l)
        else:
            pixAcc = 1.0 * self.total_correct_ul / (np.spacing(1) + self.total_label_ul)
            IoU = 1.0 * self.total_inter_ul / (np.spacing(1) + self.total_union_ul)
        mIoU = IoU.mean()
        return {"Pixel_Accuracy": pixAcc, "Mean_IoU": mIoU, "Class_IoU": dict(zip(range(self.num_classes), IoU))}

    def _log_values(self, cur_losses):
        logs = {}
        if "Ls" in cur_losses.keys():
            logs['Ls'] = self.Ls.average
        if "Ls_sub" in cur_losses.keys():
            logs['Ls_sub'] = self.Ls_sub.average
        if "Lu" in cur_losses.keys():
            logs['Lu'] = self.Lu.average
        if "Lu_reg" in cur_losses.keys():
            logs['Lu_reg'] = self.Lu_reg.average
        if "Lu_sub" in cur_losses.keys():
            logs['Lu_sub'] = self.Lu_sub.average
        logs['mIoU_l'] = self.mIoU_l
        if self.mode == 'semi':
            logs['mIoU_ul'] = self.mIoU_ul
            logs['mIoU_ul_reg'] = self.mIoU_ul_reg
        return logs

    def _write_scalars_tb(self, logs):
        for k, v in logs.items():
            if 'class_iou' not in k: self.writer.add_scalar(f'train/{k}', v, self.wrt_step)
        for i, opt_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(f'train/Learning_rate_{i}', opt_group['lr'], self.wrt_step)