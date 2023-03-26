'''
将原始的stereo blur dataset数据集转成MIMOUNet所需格式（train和test分开）
'''
import glob
import os
import io
import json


dataset_stereo_path = "/opt/datasets/stereo_blur_dataset"
config_stereo_path = "/opt/datasets/stereo_blur_dataset/stereo_deblur_data.json"
types = ["left"]
dataset_depth_path = "/home/xzf/Projects/Datasets/depth_blur_dataset_separate_left"


def ln_soft(source, target):
    cmd = "ln -s  {} {}".format(source, target)
    print(cmd)
    os.system(cmd)


with io.open(config_stereo_path, encoding='utf-8') as file:
    config_stereo = json.loads(file.read())
os.makedirs(dataset_depth_path, exist_ok=True)
seq_path = glob.glob(os.path.join(dataset_stereo_path, "HD*"))
for path in seq_path:
    seq_name = os.path.basename(path)
    phase = "no_phase"
    for seq_info in config_stereo:
        if seq_info['name'] == seq_name:
            phase = seq_info['phase'].lower()
            break
    for side in types:
        seq_name_side = seq_name+"_"+side
        depth_path = os.path.join(dataset_depth_path, phase, seq_name_side)
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




