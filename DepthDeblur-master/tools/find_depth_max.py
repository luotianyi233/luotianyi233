'''
@Author: orz
'''
import glob
import cv2
import pyexr
import numpy as np
import tqdm
import concurrent.futures

depth_path_template = "../datasets/depth_blur_dataset_separate_left/*/*/depth/*.png"
depth_scale = 1000


def find_depth_max(path):
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return depth.max().astype(float)/depth_scale


if __name__ == "__main__":
    depth_paths = glob.glob(depth_path_template)
    max = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(find_depth_max, path) for path in depth_paths]
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(depth_paths)):
            max.append(future.result())
    assert len(max) == len(depth_paths)
    # for path in tqdm.tqdm(depth_paths):
    #     replace_num = replace_nan_inf(path)
    print("最大深度距离为{}米".format(np.asarray(max).max()))
