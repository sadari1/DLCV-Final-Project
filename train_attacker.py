
#%%

import numpy as np
import os 
import argparse 
import json 
from imagenet_labels import * 
from itertools import product 
# from TREMBA.train_generator import train_tremba
from TREMBA.train_generator_modified import train_tremba
#%%

def main():

    config_path = 'configs/generator'

    configs = [f for f in os.listdir(config_path)]
    #%%
    # save_name = f"Imagenet_{}_target_{}"
    i = 0
    for c in configs:
        if 'modified' not in c.lower():
            continue 
        
        # if i < 1:
        #     i = i+1
        #     continue 
        read_path = os.path.join(config_path, c)
        with open(read_path, 'r') as reader:
            train_config = json.load(reader)
        train_config['epochs'] = 120
        if train_config['target'] != 217:
            continue 
        
        # train_config['learning_rate_G'] = 1.0

        # train_config['model_list'] = train_config['model_list'][:-1]
        # train_config['batch_size'] = 64
        train_tremba(train_config)
# %%

if __name__ == '__main__':
    main()