'''
@Author: orz
'''
import glob
import cv2
import pyexr
import numpy as np
import tqdm
import concurrent.futures

depth_path_template = "/home/xzf/Projects/Datasets/depth_blur_dataset/*/depth/*.exr"
depth_in_format = ".exr"
depth_out_format = ".png"
depth_scale = 1000


def convert_depth_format(path, scale=1000, replace_inf_nan=False):
    depth = pyexr.open(path).get()  # https://github.com/tvogels/pyexr

    # remove inf and nan
    if replace_inf_nan:
        nan_position = np.isnan(depth)
        has_nan = nan_position.sum() > 0
        if has_nan:
            depth[nan_position] = 0.

        inf_position = np.isinf(depth)
        has_inf = inf_position.sum() > 0
        if has_inf:
            depth[inf_position] = 0.

        if has_nan or has_inf:
            print("Remove inf and nan in " + path)

    # convert
    depth *= scale
    np.clip(depth, 0, 2^16)
    depth = depth.astype(np.uint16)
    new_path = path.replace(depth_in_format, depth_out_format)
    cv2.imwrite(new_path, depth)
    return True


if __name__ == "__main__":
    depth_paths = glob.glob(depth_path_template)
    replace_num = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(convert_depth_format, path) for path in depth_paths]
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(depth_paths)):
            replace_num += future.result()
    # for path in tqdm.tqdm(depth_paths):
    #     replace_num = replace_nan_inf(path)
    print("Total convert format num = {}/{}".format(replace_num, len(depth_paths)))
