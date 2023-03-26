import glob
import os

dataset_stereo_path = "/opt/datasets/stereo_blur_dataset"
dataset_depth_path = "/home/xzf/Projects/Datasets/depth_blur_dataset/"
config_stereo_path = "/opt/datasets/stereo_blur_dataset/stereo_deblur_data.json"
config_depth_path = "../datasets/depth_deblur_data.json"

#### file
def ln_soft(source, target):
    cmd = "ln -s  {} {}".format(source, target)
    print(cmd)
    os.system(cmd)

os.makedirs(dataset_depth_path, exist_ok=True)
seq_path = glob.glob(os.path.join(dataset_stereo_path, "HD*"))
for path in seq_path:
    seq_name = os.path.basename(path)
    for side in ["left", "right"]:
        seq_name_side = seq_name+"_"+side
        depth_path = os.path.join(dataset_depth_path, seq_name_side)
        os.makedirs(depth_path, exist_ok=True)
        # depth
        dir_path_stereo = os.path.join(path, "disparity_"+side)
        dir_path_depth = os.path.join(depth_path, "depth")
        ln_soft(dir_path_stereo, dir_path_depth)
        # blur
        dir_path_stereo = os.path.join(path, "image_"+side+"_blur_ga")
        dir_path_depth = os.path.join(depth_path, "blur")
        ln_soft(dir_path_stereo, dir_path_depth)
        # clear
        dir_path_stereo = os.path.join(path, "image_"+side)
        dir_path_depth = os.path.join(depth_path, "clear")
        ln_soft(dir_path_stereo, dir_path_depth)


#### config
import io
import json
import os
import copy

with io.open(config_stereo_path, encoding='utf-8') as file:
    config_stereo = json.loads(file.read())
config_depth = []
for config in config_stereo:
    name = config["name"]
    for side in ["left", "right"]:
        config["name"] = name+"_"+side
        config_depth.append(copy.deepcopy(config))
with open(config_depth_path, 'w') as file:
    json.dump(config_depth, file)



