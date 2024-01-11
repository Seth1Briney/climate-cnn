# ----------------
# trainer_main.py
# ----------------
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks

parser = ArgumentParser()
parser.add_argument("--batch_size", type=int, default=128)
from general_stats_pool_model import *
parser = StatsPoolModel.add_model_specific_args(parser)

# add all the available trainer options to argparse
# ie: now --accelerator --devices --num_nodes ... --fast_dev_run all work in the cli
parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()

#import yaml # Since we've chosen a config don't argparse (above) just load the config.
#with open('lat_pool=16_lon_pool=32_convs=True_layers=True_lr=0.0001/hparams.yaml', 'r') as f:
#    args = yaml.safe_load(f)
#import pdb;pdb.set_trace()
early_stopping = pytorch_lightning.callbacks.EarlyStopping(monitor='val_day_diff', min_delta=0.0, patience=5, verbose=False, mode='min', strict=True, check_finite=True, stopping_threshold=None, divergence_threshold=None, check_on_train_epoch_end=None)

checkpoint = pytorch_lightning.callbacks.ModelCheckpoint(dirpath=None, filename='{epoch}-{val_day_diff:.4f}', monitor='val_day_diff', verbose=False, save_last=None, save_top_k=2, save_weights_only=False, mode='min', auto_insert_metric_name=True, every_n_train_steps=None, train_time_interval=None, every_n_epochs=1, save_on_train_epoch_end=None)

trainer = Trainer.from_argparse_args(args, callbacks=[checkpoint, early_stopping])

model = StatsPoolModel(args)

train = SimpleDayHeatDataSet(split='train', args=args)

train_dataloader = DataLoader(train,batch_size=args.batch_size,shuffle=True,sampler=None,batch_sampler=None,num_workers=8,collate_fn=None,pin_memory=True,drop_last=False,timeout=0,worker_init_fn=None)

dev = SimpleDayHeatDataSet(split='dev', args=args) 
val_dataloader = DataLoader(dev,batch_size=args.batch_size,shuffle=False,sampler=None,batch_sampler=None,num_workers=8,collate_fn=None,pin_memory=True,drop_last=False,timeout=0,worker_init_fn=None)

trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

## automatically auto-loads the best weights from the previous run
# predictions = trainer.predict(model, dataloaders=val_dataloader, ckpt_path='best') 

#test = SimpleDayHeatDataSet(path='/tmp/brineys/dl_data/test',split='dev', args=args) 
#trainer.validate(test)
