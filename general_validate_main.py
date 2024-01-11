# ----------------
# trainer_main.py
# ----------------
'''
script takes a single argument - the path to the model checkpoint which is to be validated
'''
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from general_stats_pool_model import *
import sys
model = StatsPoolModel.load_from_checkpoint(sys.argv[1])
#FIXME: callbacks
trainer = Trainer.from_argparse_args(args=model.args, logger=False)
dev = SimpleDayHeatDataSet(split='dev', args=model.args, override_transform=True)
train = SimpleDayHeatDataSet(split='train', args=model.args, override_transform=True)
#test = SimpleDayHeatDataSet(split='test', args=model.args, override_transform=True)
train_dataloader = DataLoader(train,batch_size=model.args.batch_size,shuffle=False,sampler=None,batch_sampler=None,num_workers=8,collate_fn=None,pin_memory=True,drop_last=False,timeout=0,worker_init_fn=None)
val_dataloader = DataLoader(dev,batch_size=model.args.batch_size,shuffle=False,sampler=None,batch_sampler=None,num_workers=8,collate_fn=None,pin_memory=True,drop_last=False,timeout=0,worker_init_fn=None)
#test_dataloader = DataLoader(test,batch_size=model.args.batch_size,shuffle=False,sampler=None,batch_sampler=None,num_workers=8,collate_fn=None,pin_memory=True,drop_last=False,timeout=0,worker_init_fn=None)
print('dev set')
def validate():
    print('val set')
    trainer.validate(model=model, dataloaders=val_dataloader, ckpt_path=sys.argv[1], verbose=True, datamodule=None)
    print('training set')
    trainer.validate(model=model, dataloaders=train_dataloader, ckpt_path=sys.argv[1], verbose=True, datamodule=None)
#predictions = trainer.predict(model, dataloaders=val_dataloader, ckpt_path='best') 

def check_days():
    model.eval()
    for x,y in dev:
        try:
            print(y, model( torch.tensor(x) ))
            exec(input('>'))
        except:
            return False
    return True

def check_days_train():
    model.eval()
    for x,y in train:
        try:
            print(y, model( torch.tensor(x) ))
            exec(input('>'))
        except:
            return False
    return True

def check_days_test():
    model.eval()
    for x,y in test:
        try:
            print(y, model( torch.tensor(x) ))
            exec(input('>'))
        except:
            return False
    return True

def test():
    test = SimpleDayHeatDataSet(path='/tmp/brineys/dl_data/test/',split='dev', args=model.args) 
    test_dataloader = DataLoader(dev,batch_size=model.args.batch_size,shuffle=False,sampler=None,batch_sampler=None,num_workers=8,collate_fn=None,pin_memory=True,drop_last=False,timeout=0,worker_init_fn=None)
    trainer.validate(model=model, dataloaders=test_dataloader, ckpt_path=sys.argv[1], verbose=True, datamodule=None)

# main
#validate()
#test()
check_days()
check_days_train()
#check_days_test()
