from general_stats_pool_model import *
import yaml
model = StatsPoolModel.load_from_checkpoint('lat_pool=16_lon_pool=32_convs=True_layers=True_lr=0.0001/checkpoints/epoch=12-val_day_diff=1.3620.ckpt')
with open('config.yaml', 'w') as f:
    yaml.dump(model.args, f)
