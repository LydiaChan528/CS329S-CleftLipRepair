# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

import os
import pprint
import argparse
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import lib.models as models
from lib.config import config, update_config
from lib.datasets import get_dataset
from lib.core import function
from lib.utils import utils


def parse_args():

    parser = argparse.ArgumentParser(description='Train Face Alignment')

    parser.add_argument('--cfg', help='experiment configuration filename',
                        required=True, type=str)

    args = parser.parse_args()
    update_config(config, args)
    return args


def main(args):

    logger, final_output_dir, tb_log_dir = \
        utils.create_logger(config, args.cfg, 'train')

    print(os.path.abspath(final_output_dir))

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    model = models.get_face_alignment_net(config)

    # copy model files
    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    gpus = list(config.GPUS)
    if (torch.cuda.is_available()):
        model = nn.DataParallel(model, device_ids=gpus).cuda()

        # loss
        criterion = torch.nn.MSELoss(size_average=True).cuda()
    else:
        print('no cuda')
        criterion = torch.nn.MSELoss()

    optimizer = utils.get_optimizer(config, model)
    best_nme = 100
    last_epoch = config.TRAIN.BEGIN_EPOCH

    train_losses = []
    val_losses = []
    train_nmes = []
    val_nmes = []

    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir,
                                        'latest.pth')
        if os.path.islink(model_state_file):
            checkpoint = torch.load(model_state_file)
            last_epoch = checkpoint['epoch']
            best_nme = checkpoint['best_nme']
            model.load_state_dict(checkpoint['state_dict'])
            # model = torch.load(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            train_losses = checkpoint['train_losses']
            val_losses = checkpoint['val_losses']
            train_nmes = checkpoint['train_nmes']
            val_nmes = checkpoint['val_nmes']

            print("=> loaded checkpoint (epoch {})"
                  .format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found")

    if isinstance(config.TRAIN.LR_STEP, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch-1
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch-1
        )
    dataset_type = get_dataset(config)
    train_dataset = dataset_type(config, is_train=True)

    # Let's get the first half of data and train on that
    num_clients = 2
    width = len(train_dataset)//num_clients
    split_lens = [width for i in range(num_clients)]
    split_lens.append(len(train_dataset) - sum(split_lens))
    train_splits = torch.utils.data.random_split(train_dataset, split_lens)

    client_train_loaders = [
        DataLoader(
            dataset=train_splits[i], #dataset_type(config, is_train=True),
            batch_size=config.TRAIN.BATCH_SIZE_PER_GPU*len(gpus),
            shuffle=config.TRAIN.SHUFFLE,
            num_workers=config.WORKERS,
            pin_memory=config.PIN_MEMORY)
        for i in range(num_clients)
    ]

    val_loader = DataLoader(
        dataset=dataset_type(config, is_train=False),
        batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        lr_scheduler.step()

        # train_loss, train_nme = function.train(config, train_loader, model, criterion,
        #                optimizer, epoch, writer_dict)
        # ADD FEDERATED LEARNING
        local_model = copy.deepcopy(model)
        #local_model.load_state_dict(copy.deepcopy(model.state_dict()))
        local_optim = utils.get_optimizer(config, local_model)
        local_loss, local_nme = function.train(config, client_train_loaders[0], local_model,
                                                criterion, local_optim, epoch, writer_dict)
        local_weights = local_model.state_dict()
        print("\n\nLoading local weights into server model\n")
        model.load_state_dict(local_weights)

        # evaluate
        val_nme, predictions, val_loss = function.validate(config, val_loader, model,
                                             criterion, epoch, writer_dict)

        is_best = val_nme < best_nme
        best_nme = min(val_nme, best_nme)

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        print("best:", is_best)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_nmes.append(train_nme)
        val_nmes.append(val_nme)

        utils.save_checkpoint(
            {"state_dict": model.state_dict(),
             "epoch": epoch + 1,
             "best_nme": best_nme,
             "optimizer": optimizer.state_dict(),
             "train_losses": train_losses,
             "val_losses": val_losses,
             "train_nmes": train_nmes,
             "val_nmes": val_nmes
             }, predictions, is_best, final_output_dir, 'checkpoint.pth')

        utils.save_graphs(train_losses, val_losses, 'Loss', final_output_dir, 'losses')
        utils.save_graphs(train_nmes, val_nmes, 'NME', final_output_dir, 'nmes')



    final_model_state_file = os.path.join(final_output_dir,
                                          'final_state.pth')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    try:
        torch.save(model.module.state_dict(), final_model_state_file)
    except AttributeError:
        torch.save(model.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


def default_train():
    args = parse_args()
    main(args)

if __name__ == '__main__':
    default_train()








