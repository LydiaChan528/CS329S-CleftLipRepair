import torch
from torch.utils.data import DataLoader, Dataset

from lib.core import function
import copy

class LocalModel(object):
    def __init__(self, config, criterion, optimizer, train_loader, val_loader):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self, model, epoch, writer_dict):
        train_loss, train_nme = function.train(self.config, self.train_loader, 
                model, self.criterion, self.optimizer, epoch, writer_dict)
        return model.state_dict(), train_loss, train_nme

    def validate(self, model, epoch, writer_dict):
        return function.validate(self.config, self.val_loader, model, self.criterion,
                    epoch, writer_dict)

def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg
