import numpy as np
from torch.utils.data import Dataset
import os
mean = np.load('mean.npy')
std = np.load('std.npy')
import os
import numpy as np

def norm_pool_stats(filenames, source,dest,args,mean,std):
    lat_pool = args.lat_pool
    lon_pool = args.lon_pool
    num_lats = 128//lat_pool
    num_lons = 256//lon_pool
    stats = args.stats
    str_to_stat = {'mean':np.mean,'median':np.median,'max':np.max,'min':np.min,'std':np.std}
    for filename in filenames:
        X=np.load(source+'/'+filename)
        Xp=np.zeros((365,num_lats,num_lons,len(stats)))
        for d in range(X.shape[0]): # note: had to change from 365 to X.shape[0] for test
            for y in range(num_lats):
                for x in range(num_lons):
                    for i,s in enumerate(stats):
                        stat = str_to_stat[s]
                        Xp[d,y,x,i] = stat( 
                            (X[d:d+1,
                              y*lat_pool:(y+1)*lat_pool,
                              x*lon_pool:(x+1)*lon_pool
                              ].ravel()-mean)/std) 
        Xp = Xp.astype(np.float32)
        np.save(dest + filename,Xp)

class SimpleDayHeatDataSet(Dataset):
    def __init__(self, path='/tmp/brineys/dl_data/',split='train', override_transform=False,
        args = {'lat_pool': 128//8, 'lon_pool':256, 
            'stats':('mean','median','max','min','std')}
        ):
        self.filenames = os.listdir(path+split)
        if split=='train' and not override_transform:
            self.transform = 'unit_circle'
        else:
            self.transform = None
        try: 
            args_str = '_'+str(args.lat_pool)+'_'+str(args.lon_pool)+'_'+str(args.stats)
        except:
            import pdb;pdb.set_trace()
        dest = path+split + args_str + '/'
        print('dest',dest)
        if not os.path.isdir(dest):
            print('making',dest)
            os.mkdir(dest)
            norm_pool_stats(self.filenames, source=path+split, dest=dest, args=args, mean=mean,std=std)
        self.path = dest
        if split=='test':
            self.file_size=1
        else:
            self.file_size = 365-8+1

    def __len__(self):
        return self.file_size*len(self.filenames)
    def __getitem__(self, ind):
        filenum = ind // self.file_size
        ind = ind % self.file_size
        x = np.load(self.path+self.filenames[filenum])[ind:ind+8,:,:]
        try:
            x.reshape(x.shape[0],-1)
        except:
            import pdb;pdb.set_trace()
        if self.transform=='unit_circle':
            w = 2*np.pi/365
            y = np.array(( np.sin(w*ind), np.cos(w*ind) ),dtype=np.float32).reshape(2,)
        elif self.transform==None:
            y = ind
        return x, y
    def get_filename(self,ind):
        return self.filenames[ind//self.file_size]
    def get_by_filename(self, filename):
        ind = self.filenames.index(filename)
        assert self.get_filename(ind)==filename
        return self.__getitem__(ind)
#class RavelSimpleDayHeatDataSet(SimpleDayHeatDataSet):
#    def __init__(self, path='/tmp/brineys/data',split='train', args):
#        super().__init__(path=path, split=split, args=args)
#    def __getitem__(self, ind):
#        x,y = super().__getitem__(ind)
#        x=x.ravel()
#        return x,y
class WeatherDayOfYearDataset(Dataset):
    def __init__(self, split='train', transform='unit_circle'):
        self.transform = transform
        #import pdb;pdb.set_trace()
        file_path = '/tmp/brineys/dl_data/'+split
        self.filenames = [file_path+'/'+f for f in os.listdir(file_path) if f[-4:]=='.npy']
        self.file_path = file_path
        self.file_size = 365-8+1
    def __len__(self):
        return self.file_size*len(self.filenames)
    def __getitem__(self, ind):
        filenum = ind // self.file_size
        ind = ind % self.file_size
        x = np.load(self.filenames[filenum])[ind:ind+8,:,:]
        #x = x.reshape(8,-1)
        x = (x-mean)/std
        if self.transform=='unit_circle':
            w = 2*np.pi/365
            y = np.array(( np.sin(w*ind), np.cos(w*ind) ),dtype=np.float32).reshape(2,)
        elif self.transform==None:
            y = ind
        return np.squeeze(x),y
class RavelWeatherDayOfYearDataset(WeatherDayOfYearDataset):
    def __init__(self, split='train', transform=None, pooling=None, tmp=False):
        super().__init__(split=split, transform=transform, pooling=pooling, tmp=tmp)
    def __getitem__(self, ind):
        x,y = super().__getitem__(ind)
        x=x.ravel()
        return x,y
