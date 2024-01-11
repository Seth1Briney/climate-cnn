'''
script takes a single argument - the path to the model checkpoint which is to be validated
'''
from general_stats_pool_model import *
import sys
import numpy as np

model = StatsPoolModel.load_from_checkpoint(sys.argv[1])

test = SimpleDayHeatDataSet(split='test', args=model.args, override_transform=True)

print(test.filenames[0])
print(test.filenames[-1])

output = np.empty((len(test),),dtype=np.int64)

def get_test_predictions():
    filenames = []
    with open ('task5.script.txt','r') as f:
        for line in f:
            filenames.append(line.strip().split('/')[1])
    model.eval()
    i = 0
#    for x,y in test:
    for f in filenames:
        x,_ = test.get_by_filename(f)
        y_hat = model( torch.tensor(x) )
        output[i] = y_hat
        i = i+1
    np.save('task5_predictions.npy',output)
get_test_predictions()
