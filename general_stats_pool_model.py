data_path = '~/final_project/data/'

from general_datasets import SimpleDayHeatDataSet

import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader

class StatsPoolModel(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("StatsPoolModel")
        parser.add_argument("--lat_pool", type=int, default=(16))
        parser.add_argument("--lon_pool", type=int, default=32)
        parser.add_argument("--stats", type=tuple, default=('mean','median','max','min','std'))
        parser.add_argument("--opt", type=str, default='adam')
        parser.add_argument("--loss", type=str, default='huber,.5')
        parser.add_argument("--lr", type=float, default=.001)
        parser.add_argument("--convs", type=str, default=None)
        parser.add_argument("--layers", type=str, default=None)
        return parent_parser

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.relu = torch.nn.ReLU(inplace=False)
        if args.convs!=None:
            self.convs = torch.nn.ModuleList()
            self.convs.append(torch.nn.Conv3d(in_channels=len(args.stats), out_channels=8, kernel_size=(4,4,4), stride=1, padding='same', dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None))
            x = torch.zeros(1,len(args.stats),8,8,8)
            for conv in self.convs: x = conv(x)
            linear_in = len(x.ravel())
        else:
            linear_in = 8*8*8*len(args.stats)
        if args.layers!=None:
            self.hl = torch.nn.Linear(linear_in, linear_in, bias=True, device=None, dtype=torch.float32)
        if args.convs!=None or args.layers!=None:
            self.bn = torch.nn.BatchNorm1d(linear_in,
                        eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
        self.l = nn.Linear(linear_in, 2, bias=True, device=None, dtype=torch.float32)
        self.opt_str = args.opt
        if args.loss[0:5] == 'huber':
            self.loss, delta = args.loss.split(',')
            self.delta = float(delta)
        else:
            self.loss = args.loss
            self.delta = None
        self.loss = self.configure_loss(self.loss)
        self.lr = args.lr
    def forward(self, x):
        if isinstance(x,list):
            x = x[0] # FIXME: why is this happening? x is [x,y]
        if len(x.shape)<5: # if no batch - single element
            x = x.unsqueeze(0)
        if self.args.convs!=None:
            x = x.permute(0,4,2,3,1) # put stats second so it's mb, stats, ...
            for conv in self.convs:
                x = conv(x)
                x = self.relu(x)
        x = x.reshape(x.shape[0],-1)
        if self.args.layers!=None:
            try:
                x = self.hl(x)
            except:
                print(x.shape)
                import pdb;pdb.set_trace()
            x = self.relu(x)

        if self.args.convs!=None or self.args.layers!=None:
            x = self.bn(x)

        x = self.l(x)
        x = x / torch.sqrt( x[:,0]**2 + x[:,1]**2 ).reshape(-1,1)
        if self.training:
            return x 
        else:
            x = torch.atan2( x[:,0], x[:,1] )
            x = x + 2*torch.pi*(x<0)
            return torch.div( x*365 + torch.pi, (2*torch.pi), rounding_mode='trunc')
        # FIXME: output is wrong - lists ??
        
    def training_step(self, batch, batch_idx):
        self.train()
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        return loss

    def validation_step(self, batch, ind):
        self.eval()
        x, y = batch
        y_hat = self(x)
        diff = torch.abs( y_hat-y )
        diff[ diff>365//2 ] = 365 - diff[ diff>365//2 ]
        diff = torch.mean(diff.ravel())
        w = 2*torch.pi/365
        y = y.reshape(-1,1)
        y = torch.cat( ( torch.sin(w*y), torch.cos(w*y) ), axis=1)
        y_hat = y_hat.reshape(-1,1)
        y_hat = torch.cat(( torch.sin(w*y_hat), torch.cos(w*y_hat) ),axis=1)
        loss = self.loss(y_hat,y)
        self.log('val_loss', loss)
        self.log('val_day_diff', diff, prog_bar=True)
        return diff

    def configure_optimizers(self):
        if self.opt_str=='adam':
            return torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.opt_str=='nadam':
            return torch.optim.NAdam(self.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, momentum_decay=0.004)
        elif self.opt_str=='adam_amsgrad':
            return torch.optim.Adam(parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)

    def configure_loss(self, loss):
        if loss=='huber':
            return torch.nn.HuberLoss(reduction='mean', delta=self.delta)
        if loss=='mse':
            return torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')

#args = {'delta':.5, 'opt_str':'nadam', 'lr':.001, 'batch_size':128, 'loss':'huber', 'path':data_path, 'lat_pool':128//8, 'lon_pool':256//8, 'stats':('mean','median','max','min')}
#args2 = {'lat_pool': 128//8, 'lon_pool':256,
#             'stats':('mean','median','max','min','std')}
#args['in_dim'] = 40
