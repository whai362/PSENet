import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import cv2
import time
from ..loss import build_loss, ohem_batch, iou
from ..post_processing import pse


class PSENet_Head(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_dim,
                 num_classes,
                 loss_text,
                 loss_kernel):
        super(PSENet_Head, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(hidden_dim, num_classes, kernel_size=1, stride=1, padding=0)

        self.text_loss = build_loss(loss_text)
        self.kernel_loss = build_loss(loss_kernel)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, f):
        out = self.conv1(f)
        out = self.relu1(self.bn1(out))
        out = self.conv2(out)

        return out

    def get_results(self, out, img_meta, cfg):
        outputs = dict()

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            start = time.time()

        score = torch.sigmoid(out[:, 0, :, :])
        # out = (torch.sign(out - 1) + 1) / 2  # 0 1
        #
        # text_mask = out[:, 0, :, :]
        # kernels = out[:, 1:cfg.test_cfg.kernel_num, :, :] * text_mask

        kernels = out[:, :cfg.test_cfg.kernel_num, :, :] > 0
        text_mask = kernels[:, :1, :, :]
        kernels[:, 1:, :, :] = kernels[:, 1:, :, :] * text_mask

        score = score.data.cpu().numpy()[0].astype(np.float32)
        kernels = kernels.data.cpu().numpy()[0].astype(np.uint8)
        # kernel_1 = kernels[1]
        # kernel_2 = kernels[2]
        # kernel_3 = kernels[3]
        # kernel_4 = kernels[4]
        # kernel_5 = kernels[5]
        # kernel_6 = kernels[6]
        #
        # kernel_1 = kernel_1.reshape(736, 1120, 1)
        # kernel_2 = kernel_2.reshape(736, 1120, 1)
        # kernel_3 = kernel_3.reshape(736, 1120, 1)
        # kernel_4 = kernel_4.reshape(736, 1120, 1)
        # kernel_5 = kernel_5.reshape(736, 1120, 1)
        # kernel_6 = kernel_6.reshape(736, 1120, 1)
        #
        # kernel_1 = np.concatenate((kernel_1, kernel_1, kernel_1), axis=2) * 255
        # kernel_2 = np.concatenate((kernel_2, kernel_2, kernel_2), axis=2) * 255
        # kernel_3 = np.concatenate((kernel_3, kernel_3, kernel_3), axis=2) * 255
        # kernel_4 = np.concatenate((kernel_4, kernel_4, kernel_4), axis=2) * 255
        # kernel_5 = np.concatenate((kernel_5, kernel_5, kernel_5), axis=2) * 255
        # kernel_6 = np.concatenate((kernel_6, kernel_6, kernel_6), axis=2) * 255
        #
        # kernel_1 = cv2.copyMakeBorder(kernel_1, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[255, 0, 0])
        # kernel_2 = cv2.copyMakeBorder(kernel_2, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[255, 0, 0])
        # kernel_3 = cv2.copyMakeBorder(kernel_3, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[255, 0, 0])
        # kernel_4 = cv2.copyMakeBorder(kernel_4, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[255, 0, 0])
        # kernel_5 = cv2.copyMakeBorder(kernel_5, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[255, 0, 0])
        # kernel_6 = cv2.copyMakeBorder(kernel_6, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[255, 0, 0])
        #
        # res = np.concatenate((kernel_1, kernel_2, kernel_3, kernel_4, kernel_5, kernel_6), axis=1)
        # print('saved kernels.')
        # cv2.imwrite('vis_kernels.png', res)
        # exit()

        label = pse(kernels, cfg.test_cfg.min_area)

        # image size
        org_img_size = img_meta['org_img_size'][0]
        img_size = img_meta['img_size'][0]

        label_num = np.max(label) + 1
        label = cv2.resize(label, (img_size[1], img_size[0]), interpolation=cv2.INTER_NEAREST)
        score = cv2.resize(score, (img_size[1], img_size[0]), interpolation=cv2.INTER_NEAREST)

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(
                det_pse_time=time.time() - start
            ))

        scale = (float(org_img_size[1]) / float(img_size[1]),
                 float(org_img_size[0]) / float(img_size[0]))

        bboxes = []
        scores = []
        for i in range(1, label_num):
            ind = label == i
            points = np.array(np.where(ind)).transpose((1, 0))

            if points.shape[0] < cfg.test_cfg.min_area:
                label[ind] = 0
                continue

            score_i = np.mean(score[ind])
            if score_i < cfg.test_cfg.min_score:
                label[ind] = 0
                continue

            if cfg.test_cfg.bbox_type == 'rect':
                rect = cv2.minAreaRect(points[:, ::-1])
                bbox = cv2.boxPoints(rect) * scale
            elif cfg.test_cfg.bbox_type == 'poly':
                binary = np.zeros(label.shape, dtype='uint8')
                binary[ind] = 1
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                bbox = contours[0] * scale

            bbox = bbox.astype('int32')
            bboxes.append(bbox.reshape(-1))
            scores.append(score_i)

        outputs.update(dict(
            bboxes=bboxes,
            scores=scores
        ))

        return outputs

    def loss(self, out, gt_texts, gt_kernels, training_masks):
        # output
        texts = out[:, 0, :, :]
        kernels = out[:, 1:, :, :]
        # text loss
        selected_masks = ohem_batch(texts, gt_texts, training_masks)

        loss_text = self.text_loss(texts, gt_texts, selected_masks, reduce=False)
        iou_text = iou((texts > 0).long(), gt_texts, training_masks, reduce=False)
        losses = dict(
            loss_text=loss_text,
            iou_text=iou_text
        )

        # kernel loss
        loss_kernels = []
        selected_masks = gt_texts * training_masks
        for i in range(kernels.size(1)):
            kernel_i = kernels[:, i, :, :]
            gt_kernel_i = gt_kernels[:, i, :, :]
            loss_kernel_i = self.kernel_loss(kernel_i, gt_kernel_i, selected_masks, reduce=False)
            loss_kernels.append(loss_kernel_i)
        loss_kernels = torch.mean(torch.stack(loss_kernels, dim=1), dim=1)
        iou_kernel = iou(
            (kernels[:, -1, :, :] > 0).long(), gt_kernels[:, -1, :, :], training_masks * gt_texts, reduce=False)
        losses.update(dict(
            loss_kernels=loss_kernels,
            iou_kernel=iou_kernel
        ))

        return losses
