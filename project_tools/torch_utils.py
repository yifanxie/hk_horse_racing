import os, sys, random
import numpy as np
import pandas as pd
from project_tools import project_utils, project_class

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize
from torch.utils.data import Dataset


import shutil
import pickle
import time
import copy
from datetime import datetime
from glob import glob

try:
    import apex
    from apex import amp
    has_apex=True
except:
    has_apex = False
    pass


# utility function:
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


# define RMSE loss
class RMSELoss(nn.Module):
# usage:
# criterion = RMSELoss()
# loss = criterion(torch.tensor(y_true), torch.tensor(y_pred))
    def __init__(self, eps=1e-8):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
    def forward(self, y_true, y_preds):
        loss = torch.sqrt(self.mse(y_true.flatten(), y_preds.flatten()) + self.eps)
        return loss


class Invert_PCC(nn.Module):
    """
    implementation of Pearson Correlation
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    def forward(self, y_true, y_preds):
        # loss = torch.sqrt(self.mse(y_true, y_preds) + self.eps)
        vt = y_true - torch.mean(y_true)
        vp = y_preds - torch.mean(y_preds)
        corr = torch.sum(vt * vp) / (torch.sqrt(torch.sum(vt ** 2)) * torch.sqrt(torch.sum(vp ** 2))) + self.eps
        return 1-corr


class Tabular_Dataset(Dataset):
    def __init__(self, dat_df, features, target=None):
        self.data = dat_df
        self.features = features
        self.target = target
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        x = self.data.iloc[idx, :][self.features].fillna(0).values
        if self.target is not None:
            y = self.data.iloc[idx, :][self.target]
            return x, y
        else:
            return x


class Era_Dataset(Dataset):
    def __init__(self, dat_df, features, era_feat='era', target=None):
        self.data = dat_df
        self.features = features
        self.target = target
        self.era_feat = era_feat
        self.era_list = self.data[era_feat].unique().tolist()
    def __len__(self):
        return len(self.era_list)
    def __getitem__(self, idx):
        era = self.era_list[idx]
        idx = self.data[self.data[self.era_feat]==era].index
        x = self.data.loc[idx][self.features].fillna(0).values
        if self.target is not None:
            y = self.data.loc[idx][self.target].values
            return x, y
        else:
            return x



# classifc fully connect NN with specified number of nodes per layer
class FCNet(torch.nn.Module):
    def __init__(self, n_feature, n_output, n_hidden = [64, 32], use_dropout=True):
        super(FCNet, self).__init__()
        num_nodes = [n_feature] + n_hidden
        fcs = []
        for i in range(len(num_nodes)-1):
            fcs.append(nn.Linear(num_nodes[i], num_nodes[i+1]))
        self.fcs = nn.ModuleList(fcs)
        self.dropout = nn.Dropout(0.25)
        self.predict = nn.Linear(num_nodes[-1], n_output)   # output layer
        self.use_dropout = use_dropout

    def forward(self, x):
        for fc in self.fcs:
            if self.use_dropout:
                x = F.relu(self.dropout(fc(x)))
            else:
                x = F.relu(fc(x))
        x = self.predict(x)             # linear output
        return x



class GaussianNoise(nn.Module):
    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.register_buffer('noise', torch.tensor(0))

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.expand(*x.size()).float().normal_() * scale
            x = x + sampled_noise
        return x 


class Denoise_Autoencoder(nn.Module):
    def __init__(self, in_dimension, embedding_dimension=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(in_dimension, 256),
            nn.BatchNorm1d(256),
            nn.Hardswish(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Hardswish(),
            nn.Dropout(p=0.1),
            nn.Linear(128, embedding_dimension),)
        self.decoder = nn.Sequential(
            nn.BatchNorm1d(embedding_dimension),
            nn.Hardswish(),
            nn.Dropout(p=0.1),
            nn.Linear(embedding_dimension, 128),
            nn.BatchNorm1d(128),
            nn.Hardswish(),
            nn.Dropout(p=0.1),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.Hardswish(),
            nn.Dropout(p=0.1),
            nn.Linear(256, in_dimension),)
    def forward(self, x):
        embedding = self.encoder(x)
        decode = self.decoder(embedding)
        return embedding, decode


class TrainGlobalConfig:
    num_workers = 8
    batch_size = 256
    n_epochs = 500
    early_stopping = 5
    early_stopping_eps = 1e-8
    verbose = True
    verbose_step = 1
    criterion = RMSELoss()
    step_scheduler = False  # do scheduler.step after optimizer.step
    validation_scheduler = True  # do scheduler.step after validation stage loss
    # lr = 0.01
    scheduler_class = torch.optim.lr_scheduler.ReduceLROnPlateau #torch.optim.lr_scheduler.CosineAnnealingLR
    scheduler_params = dict(factor = 0.1, patience=2)
    optimizer_class = torch.optim.SGD #torch.optim.AdamW
    optimizer_params = dict(lr=0.01)  #dict(lr=lr, eps=1e-4)

class TorchModelFitter():
    def __init__(self, model, optimizer, device, config, model_path, use_apex=True, multi_gpus=False, prefix='', postfix=''):
        self.config = config
        self.epoch = 0
        self.base_dir = model_path
        self.log_path = os.path.join(self.base_dir, 'log.txt')
        self.best_summary_loss = 10 ** 5
        self.best_weight_auc = 0
        self.current_weight_auc = 0
        self.current_summary_loss = 10 ** 5
        self.device = device
        self.multi_gpus = multi_gpus
        self.use_apex = use_apex
        self.criterion = config.criterion #RMSELoss()
        if len(prefix)>0:
            prefix = prefix + '_'
        if len(postfix)>0:
            postfix = '_' + postfix
        self.prefix = prefix
        self.postfix = postfix
        self.best_weight_file = os.path.join(self.base_dir,f'{self.prefix}best_checkpoint{self.postfix}.bin')
            # self.best_weight_file = os.path.join(self.base_dir,f'best_checkpoint.bin')
        if 'early_stopping' in config.__dict__:
            self.early_stopping = config.early_stopping
            self.early_stopping_count = 0
            self.early_stopping_eps = config.early_stopping_eps
        if self.use_apex:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)

        if self.multi_gpus:
            print('use multi-gpu')
            model = torch.nn.DataParallel(model)

        self.model = model
        self.optimizer = optimizer

        self.scheduler = config.scheduler_class(self.optimizer, **config.scheduler_params)

    def fit(self, train_loader, validation_loader):
        for e in range(self.config.n_epochs):
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
                self.log(f'\n{timestamp}\nLR: {lr}')
            t = time.time()
            summary_loss = self.train_one_epoch(train_loader)

            self.log(
                f'[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.8f}, time: {(time.time() - t):.2f}')
            self.save(f'{self.base_dir}/{self.prefix}last_checkpoint{self.postfix}.bin')

            t = time.time()
            val_summary_loss = self.validation(validation_loader)
            self.current_summary_loss = val_summary_loss.avg
            self.log(
                f'[RESULT]: Val. Epoch: {self.epoch}, summary_loss: {self.current_summary_loss:.8f}, time: {(time.time() - t):.2f}')

            if (self.current_summary_loss + self.early_stopping_eps) < self.best_summary_loss:
                self.best_summary_loss = self.current_summary_loss  # final_scores.avg
                self.model.eval()
                # self.save(f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')
                self.save(self.best_weight_file)
                self.early_stopping_count = 0
            else:
                self.early_stopping_count +=1
            # if not self.config.use_warmup:
            self.config.verbose and print('[EARLY STOPPING COUNT]: %d of %d' % (self.early_stopping_count, self.early_stopping))
            try:
                self.scheduler.step(metrics=self.current_summary_loss)
            except:
                self.scheduler.step()
            if self.early_stopping_count>self.early_stopping:
                self.config.verbose and print('Early stopping count reached, stopping')
                break
            else:
                self.epoch += 1

    def validation(self, val_loader):
        self.model.eval()
        val_loss = []
        summary_loss = AverageMeter()
        t = time.time()
        for step, (x_batch, y_batch_true) in enumerate(val_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Train Step {step}/{len(val_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.8f} ' + \
                        f'time: {(time.time() - t):.2f}', end='\r'
                    )
            with torch.no_grad():
                y_batch_true = y_batch_true.to(self.device).float()
                x_batch = x_batch.to(self.device).float()
                batch_size = len(x_batch)
                y_batch_preds = self.model(x_batch)
                loss = self.criterion(y_batch_true, y_batch_preds)
                val_loss.append(loss.detach().cpu().numpy())
                summary_loss.update(loss.detach().item(), batch_size)
        return summary_loss

    def train_one_epoch(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()
        t = time.time()
        for step, (x_batch, y_batch_true) in enumerate(train_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Train Step {step}/{len(train_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.8f} ' + \
                        f'time: {(time.time() - t):.2f}', end='\r'
                    )
            # print(x_batch.shape, y_batch_true.shape)
            y_batch_true = y_batch_true.flatten().to(self.device).float()
            x_batch = x_batch.to(self.device).float()
            batch_size = len(x_batch)
            self.optimizer.zero_grad()
            y_batch_preds = self.model(x_batch)
            loss = self.criterion(y_batch_true, y_batch_preds)

            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            summary_loss.update(loss.detach().item(), batch_size)
            self.optimizer.step()
            if self.config.step_scheduler:
                self.scheduler.step()
        return summary_loss

    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_weight_auc': self.best_weight_auc,
            'epoch': self.epoch,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        try:
            self.best_weight_auc = checkpoint['best_weight_auc']
        except:
            pass
        self.epoch = checkpoint['epoch'] + 1

    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')


class Denoise_Autoencoder(nn.Module):
    def __init__(self, in_dimension, embedding_dimension=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(in_dimension, 256),
            nn.BatchNorm1d(256),
            nn.Hardswish(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Hardswish(),
            nn.Dropout(p=0.1),
            nn.Linear(128, embedding_dimension),
        )
        self.decoder = nn.Sequential(
            nn.BatchNorm1d(embedding_dimension),
            nn.Hardswish(),
            nn.Dropout(p=0.1),
            nn.Linear(embedding_dimension, 128),
            nn.BatchNorm1d(128),
            nn.Hardswish(),
            nn.Dropout(p=0.1),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.Hardswish(),
            nn.Dropout(p=0.1),
            nn.Linear(256, in_dimension),
        )
    def forward(self, x):
        embedding = self.encoder(x)
        decode = self.decoder(embedding)
        return embedding, decode


class DAE_fitter(TorchModelFitter):
    def __init__(self, model, optimizer, device, config, model_path, use_apex=True, multi_gpus=False):
        super().__init__(model, optimizer, device, config, model_path, use_apex, multi_gpus)

    def validation(self, val_loader):
        self.model.eval()
        val_loss = []
        summary_loss = AverageMeter()
        t = time.time()
        for step, x_batch in enumerate(val_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Train Step {step}/{len(val_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.8f} ' + \
                        f'time: {(time.time() - t):.2f}', end='\r'
                    )
            with torch.no_grad():
                x_batch = x_batch.to(self.device).float()
                batch_size = len(x_batch)
                x_embedding, x_decode = self.model(x_batch)
                loss = self.criterion(x_batch, x_decode)
                val_loss.append(loss.detach().cpu().numpy())
                summary_loss.update(loss.detach().item(), batch_size)
        return summary_loss

    def train_one_epoch(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()
        t = time.time()
        for step, x_batch in enumerate(train_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Train Step {step}/{len(train_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.8f} ' + \
                        f'time: {(time.time() - t):.2f}', end='\r'
                    )
            # print(x_batch.shape, y_batch_true.shape)
            x_batch = x_batch.to(self.device).float()
            batch_size = len(x_batch)
            self.optimizer.zero_grad()
            x_embedding, x_decode = self.model(x_batch)
            loss = self.criterion(x_batch, x_decode)
            # print(y_batch_true[0:5], y_batch_preds[0:5])
            # tt1 = targets1
            # tp1 = logits1
            # tt2 = targets2
            # tp2 = logits2

            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            summary_loss.update(loss.detach().item(), batch_size)
            self.optimizer.step()
            if self.config.step_scheduler:
                self.scheduler.step()
        return summary_loss



class TorchModel():
    def __init__(self, model_params, device, config, model_path, use_apex=True, multi_gpus=False, prefix='', postfix='', clean_model_dir=False, best_weight_file=''):
        self.config = config
        self.device = device
        if model_params['network'] == 'FCNet':
            in_features = model_params['in_features']
            out_features = model_params['out_features']
            hidden_params = model_params['hidden_params']
            use_dropout = model_params['use_dropout']
            model = FCNet(in_features, out_features, hidden_params, use_dropout).to(device)
            optimizer = config.optimizer_class(model.parameters(), **config.optimizer_params)
        else:
            print('not supported')

        self.model_path = model_path
        self.clean_model_dir = clean_model_dir
        self.fitter = TorchModelFitter(model, optimizer, device, config, model_path, use_apex, multi_gpus, prefix, postfix)
        if len(best_weight_file)>0:
            self.fitter.best_weight_file = best_weight_file

    def fit(self, train_df, val_df, features, y_label='target'):
        if self.clean_model_dir:
            project_utils.empty_folder(self.model_path)
        train_ds = Tabular_Dataset(train_df, features, y_label)
        valid_ds = Tabular_Dataset(val_df, features, y_label)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=self.config.batch_size, pin_memory=False,
                                                   drop_last=True, num_workers=self.config.num_workers)
        valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=self.config.batch_size, pin_memory=False,
                                                   drop_last=True, num_workers=self.config.num_workers)
        self.fitter.fit(train_loader, valid_loader)


    def predict(self, test_df, features):
        # if not(os.path.isfile(self.fitter.best_weight_file)):
        #     print('best weight file is not found, failed to load model')
        #     return np.nan
        # else:
        try:
            self.fitter.load(self.fitter.best_weight_file)
        except:
            print(f'error at loading specified weight file {self.fitter.best_weight_file}')
        test_ds = Tabular_Dataset(test_df, features, target=None)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=self.config.batch_size, pin_memory=False, drop_last=False, num_workers=self.config.num_workers)
        test_preds = []
        with torch.no_grad():
            for step, x_batch in enumerate(test_loader):
                self.config.verbose and print(f'Step {step}/{len(test_loader)}', end='\r')
                x_batch = x_batch.to(self.device).float()
                y_tour_batch_preds = self.fitter.model(x_batch)
                test_preds.append(y_tour_batch_preds.flatten())
        test_preds = torch.cat(test_preds).cpu().numpy().flatten()
        return test_preds





