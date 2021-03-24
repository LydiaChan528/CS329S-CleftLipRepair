# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

import os
import pprint
import argparse
import copy
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import lib.models as models
from lib.config import config, update_config
from lib.datasets import get_dataset
from lib.core import function
from lib.utils import utils
from federated_utils import *

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
    weights = model.state_dict()

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
    #make client dataloaders
    dataset_type = get_dataset(config)
    train_dataset = dataset_type(config, is_train=True)
    client_datasets_lens = []

    num_clients = config.TRAIN.CLIENTS
    sum_dataset_lens = 0
    for i in range(num_clients-1):
        dataset_len = int(len(train_dataset) / num_clients)
        client_datasets_lens.append(dataset_len)
        sum_dataset_lens += dataset_len
    client_datasets_lens.append(len(train_dataset) - sum_dataset_lens)

    train_datasets = torch.utils.data.random_split(train_dataset, client_datasets_lens)
    
    train_loaders = []
    for dataset in train_datasets:
        train_loader = DataLoader(
            dataset=dataset,
            batch_size=config.TRAIN.BATCH_SIZE_PER_GPU*len(gpus),
            shuffle=config.TRAIN.SHUFFLE,
            num_workers=config.WORKERS,
            pin_memory=config.PIN_MEMORY)
        train_loaders.append(train_loader)

    val_loader = DataLoader(
        dataset=dataset_type(config, is_train=False),
        batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    logger.info("Federated training: {} with {} clients".format(config.TRAIN.FEDERATED, num_clients))

    val_stats = np.zeros((config.TRAIN.END_EPOCH - last_epoch, 1 + 2 + 2*num_clients))
    val_stats[:,0] = np.arange(last_epoch, config.TRAIN.END_EPOCH)

    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        local_weights, local_losses, local_nmes = [], [], []
        local_models = []

        print("Saving to "+final_output_dir)

        #train each client model with their own data
        for idx in range(num_clients): #idx_users:
            logger.info(f"\nTraining model {idx}")
            #local_model = LocalModel(config=config, criterion=criterion, optimizer=optimizer, train_loader=train_loaders[idx],
            #                val_loader=val_loader)
            #local_models.append(local_model)
            #local_w, local_loss, local_nme = local_model.train(model=copy.deepcopy(model),
            #        epoch=epoch, writer_dict=writer_dict)
            
            local_model = copy.deepcopy(model)
            local_optim = utils.get_optimizer(config, local_model)
            local_loss, local_nme = function.train(config, train_loaders[idx], local_model, 
                                                   criterion, local_optim, epoch, writer_dict)
            local_w = local_model.state_dict()
            local_models.append(local_model)

            logger.info(f"Training model {idx} has local_nme={local_nme}, local_loss={local_loss}")
            local_weights.append(copy.deepcopy(local_w))
            local_losses.append(copy.deepcopy(local_loss))
            local_nmes.append(copy.deepcopy(local_nme))
        lr_scheduler.step()

        #combine weights into global model
        weights = average_weights(local_weights)

        model.load_state_dict(weights)

        train_loss = sum(local_losses) / len(local_losses)
        train_nme = sum(local_nmes) / len(local_nmes)

        #evaluate
        local_val_nmes, local_val_losses = [], []
        
        for i, local_model in enumerate(local_models):
            #local_val_nme, predictions, local_val_loss = local_model.validate(model, epoch, writer_dict)

            local_val_nme, __, local_val_loss = function.validate(config, val_loader, local_model,
                                                         criterion, epoch, writer_dict)
            local_val_nmes.append(local_val_nme)
            local_val_losses.append(local_val_loss)
            logger.info(f"\n[VALIDATION] Local model {i} has val_nme={local_val_nme}, val_loss={local_val_loss}")
            val_stats[epoch-last_epoch,1+i*2] = local_val_loss
            val_stats[epoch-last_epoch,2+i*2] = local_val_nme

        # Get nme and loss of monolithic model
        val_nme, preds, val_loss = function.validate(config, val_loader, model, criterion, epoch, writer_dict)
        logger.info(f"\n[VALIDATION] Server model has val_nme={val_nme}, val_loss={val_loss}")
        val_stats[epoch-last_epoch, -2] = val_loss
        val_stats[epoch-last_epoch, -1] = val_nme
        ##Averages of local models
        #val_nme = sum(local_val_nmes) / len(local_val_nmes)
        #val_loss = sum(local_val_losses) / len(local_val_losses)

        val_stats_path = (final_output_dir+"/val_stats")
        print("Saving validation stats to ", val_stats_path)
        np.save(val_stats_path, val_stats)

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

default_train()








