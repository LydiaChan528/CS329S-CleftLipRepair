# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os
import shutil

import torch
import numpy as np

from .evaluation import decode_preds, compute_nme

import matplotlib.pyplot as plt
from PIL import Image
import cv2

logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(config, train_loader, model, criterion, optimizer,
          epoch, writer_dict):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    nme_count = 0
    nme_batch_sum = 0

    end = time.time()

    for i, (inp, target, meta) in enumerate(train_loader):
        # measure data time
        data_time.update(time.time()-end)

        # compute the output
        output = model(inp)

        if torch.cuda.is_available():
            target = target.cuda(non_blocking=True)

        loss = criterion(output, target)

        # NME
        score_map = output.data.cpu()
        preds = decode_preds(score_map, meta['center'], meta['scale'], [64, 64])

        nme_batch = compute_nme(preds, meta)
        nme_batch_sum = nme_batch_sum + np.sum(nme_batch)
        nme_count = nme_count + preds.size(0)

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), inp.size(0))

        batch_time.update(time.time()-end)
        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=inp.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            logger.info(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

        end = time.time()
    nme = nme_batch_sum / nme_count
    msg = 'Train Epoch {} time:{:.4f} loss:{:.6f} nme:{:.6f}'\
        .format(epoch, batch_time.avg, losses.avg, nme)
    logger.info(msg)
    return losses.avg, nme


def validate(config, val_loader, model, criterion, epoch, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()

    num_classes = config.MODEL.NUM_JOINTS
    predictions = torch.zeros((len(val_loader.dataset), num_classes, 2))

    model.eval()

    nme_count = 0
    nme_batch_sum = 0
    count_failure_008 = 0
    count_failure_010 = 0
    end = time.time()

    with torch.no_grad():
        for i, (inp, target, meta) in enumerate(val_loader):
            data_time.update(time.time() - end)
            output = model(inp)
            if torch.cuda.is_available():
                target = target.cuda(non_blocking=True)

            score_map = output.data.cpu()
            # loss
            loss = criterion(output, target)

            preds = decode_preds(score_map, meta['center'], meta['scale'], [64, 64])

            trans_target = decode_preds(target.data.cpu(), meta['center'], meta['scale'], [64, 64])

            # NME
            nme_temp = compute_nme(preds, meta)
            # Failure Rate under different threshold
            failure_008 = (nme_temp > 0.08).sum()
            failure_010 = (nme_temp > 0.10).sum()
            count_failure_008 += failure_008
            count_failure_010 += failure_010

            nme_batch_sum += np.sum(nme_temp)
            nme_count = nme_count + preds.size(0)
            for n in range(score_map.size(0)):
                predictions[meta['index'][n], :, :] = preds[n, :, :]

            losses.update(loss.item(), inp.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    nme = nme_batch_sum / nme_count
    failure_008_rate = count_failure_008 / nme_count
    failure_010_rate = count_failure_010 / nme_count

    msg = 'Test Epoch {} time:{:.4f} loss:{:.6f} nme:{:.6f} [008]:{:.4f} ' \
          '[010]:{:.4f}'.format(epoch, batch_time.avg, losses.avg, nme,
                                failure_008_rate, failure_010_rate)
    logger.info(msg)

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', losses.avg, global_steps)
        writer.add_scalar('valid_nme', nme, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    return nme, predictions, losses.avg


def inference(config, data_loader, model, model_file):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    num_classes = config.MODEL.NUM_JOINTS
    predictions = torch.zeros((len(data_loader.dataset), num_classes, 2))

    model.eval()

    nme_count = 0
    nme_batch_sum = 0
    count_failure_008 = 0
    count_failure_010 = 0
    end = time.time()

    assert model_file.rfind('/') < len(model_file) - 1
    model_folder = model_file[:model_file.rfind('/')]

    output_folder = os.path.join(model_folder, 'output_images')
    if os.path.exists(output_folder):
        for root, dirs, files in os.walk(output_folder):
            for name in files:
                os.remove(os.path.join(root, name))
    else:
        os.mkdir(output_folder)

    num_output_images = 0


    with torch.no_grad():
        for i, (inp, target, meta) in enumerate(data_loader):
            data_time.update(time.time() - end)
            output = model(inp)
            score_map = output.data.cpu()
            preds = decode_preds(score_map, meta['center'], meta['scale'], [64, 64])

            # NME
            nme_temp = compute_nme(preds, meta)

            failure_008 = (nme_temp > 0.08).sum()
            failure_010 = (nme_temp > 0.10).sum()
            count_failure_008 += failure_008
            count_failure_010 += failure_010

            nme_batch_sum += np.sum(nme_temp)
            nme_count = nme_count + preds.size(0)
            for n in range(score_map.size(0)):
                predictions[meta['index'][n], :, :] = preds[n, :, :]

            if num_output_images < 15:
                num_output_images += save_images(meta, preds, output_folder)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    nme = nme_batch_sum / nme_count
    failure_008_rate = count_failure_008 / nme_count
    failure_010_rate = count_failure_010 / nme_count

    msg = 'Test Results time:{:.4f} loss:{:.6f} nme:{:.6f} [008]:{:.4f} ' \
          '[010]:{:.4f}'.format(batch_time.avg, losses.avg, nme,
                                failure_008_rate, failure_010_rate)
    logger.info(msg)

    return nme, predictions

def save_images(meta, preds, output_folder):
    points = ['lala','rala','lsbal','rsbbal','lc','rc','sn',
            'lcphs','rcphs','rcphi','mcphi','lcphi','ls','sto',
            'lch','rch','prn','rlr(r)','rlr(p)','rla','cp']

    for j in range(len(meta['img_path'])):
        img = np.array(Image.open(meta['img_path'][j]).convert('RGB'), dtype=np.float32)
        white = np.zeros([img.shape[0],img.shape[1],3],dtype=np.uint8)
        white.fill(255) # or white[:] = 255
        for i, pt in enumerate(preds[j]):
            x, y = pt
            # paint predictions on original images
            cv2.circle(img, (int(x), int(y)), 2, (255, 255, 0), -1)
            cv2.putText(img, points[i], (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0))
            # paint human annotations also
            # c_x, c_y = meta['pts'][j][i]
            # cv2.circle(img, (c_x, c_y), 2, (255, 255, 255), -1)
            # cv2.arrowedLine(img, (c_x, c_y), (x, y), (0, 0, 0))

            # paint predictions on white canvas
            cv2.circle(white, (int(x), int(y)), 2, (255, 0, 0), -1)
            # cv2.putText(white, points[i], (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0))
            

        image_name = meta['img_path'][j][meta['img_path'][j].rfind('/') + 1:]
        image_output = os.path.join(output_folder, image_name)
        image_output_white = os.path.join(output_folder, "projections/proj_"+image_name)
        plt.imsave(image_output, img / 255) # save original
        plt.imsave(image_output_white, white / 255) # save white
        plt.close()

    return len(meta['img_path'])

