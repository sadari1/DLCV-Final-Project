#%%
import numpy as np
import os 
import argparse 
import json 
from imagenet_labels import * 
from itertools import product 

#%%


# def main():

# Load initial training configuration file 
parent_config_path = "CW_attack_config.json"
with open(parent_config_path, "r") as jn:
    config = json.load(jn)

#%%
video_path = "data/videos/"

config['video_path'] = video_path

video_names = os.listdir(video_path)
# %%
conv_model = ["vgg16"]#["densenet121", "resnet152", "vgg16"]
# conv_model = ['+'.join(["densenet121", "resnet152", "vgg16"])]#["densenet121", "resnet152", "vgg16"]
# target_classes = [217, 566, 701]
target_classes = [217]
k = [10, 30, 40]
c = [10]

# epsilons = [eps, 2*eps, 3*eps]
# video_names * target_classes * epsilon * conv_model
# %%
headers = ['video_name', 'source_model', 'target', 'k', 'c']

# generate cartesian product combinations of video names, conv models, targets and epsilons
cartesian_product = list(product(video_names, conv_model, target_classes, k, c))

# %%
config_save_path = "configs/attack_cw"
if not os.path.exists(config_save_path):
    os.makedirs(config_save_path)
#%%
for p in cartesian_product:
    tmp_config = config.copy()
    for f in range(len(p)):
        tmp_config[headers[f]] = p[f]
    
    config_save_name = f"{tmp_config['source_model']}_{tmp_config['target']}_{tmp_config['k']}_{tmp_config['c']}_{tmp_config['video_name'].split('.')[0]}.json"
    config_save = os.path.join(config_save_path, config_save_name)
    with open(config_save, 'w') as writer:
        json.dump(tmp_config, writer)
    
# %%
