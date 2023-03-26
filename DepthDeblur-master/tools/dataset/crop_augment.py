import cv2 as cv
import os
import numpy as np
import argparse
import time
import json


def read_degree_map(dir):
    imgs = []
    imgs_list = sorted(os.listdir(dir))
    if len(imgs_list) != 100:
        print("警告：degree map的数量为{}，不等于100！".format(len(imgs_list)))
    for i in imgs_list:
        if os.path.splitext(i)[-1] == ".png":
            imgs.append(cv.imread(os.path.join(dir, i), flags=cv.IMREAD_GRAYSCALE))
        else:
            print("警告：发现非png格式文件{}".format(i))
    return imgs


def region_degree_sort(degree_maps: list, patch_size: list, select=0.01) -> list:
    '''
    input
    degree_maps：将所有degree map用opencv读取后存放在一个list中
    patch_size：patch的大小(H*W)，例如[256,256]
    select：要选取的最大的前i%
    ----------------------------------------------------------------
    output：值最大的前i%的patch的值和坐标范围
    '''
    # degree map求和
    degree_maps = np.array(degree_maps, dtype=np.float64)
    degree_maps_sum = degree_maps.sum(axis=0)
    # 计算积分图
    integral_img = cv.integral(degree_maps_sum)
    integral_img = integral_img[1:, 1:]  # 左边和上面会多一行0，所以要去掉
    # 使用积分图计算patch的和
    H = integral_img.shape[0] - patch_size[0]
    W = integral_img.shape[1] - patch_size[1]
    # patch_sum = np.zeros((H,W), dtype=np.float64)
    patch_sum=[]
    step = 2
    for y in range(H):  #
        for x in range(W):
            y_ = y + patch_size[0] - 1
            x_ = x + patch_size[1] - 1
            sum = integral_img[y_, x_] - integral_img[y, x_] - integral_img[y_, x] + integral_img[y, x]
            patch_sum.append({'sum':sum, 'location':(x, y, x_, y_)})
    # 排序
    patch_sum = sorted(patch_sum, key=(lambda x:x['sum']), reverse=True)
    # 选取最大的前i%
    return patch_sum[:int(H*W*select)]


if __name__ == "__main__":
    start=time.time()
    print("=" * 30, " start ", "=" * 30)
    parser = argparse.ArgumentParser(description="基于degree map的数据增强")
    parser.add_argument('--datasets', type=str, help="数据集路径。支持两种模式。模式一：路径下为视频名称；"
                                                     "模式二：路径下为test、train，然后其下为视频名称")
    parser.add_argument('--patch_size', type=int, nargs=2, default=[256, 256], help="patch的大小。默认为[256,256]")
    parser.add_argument('--select_ratio', type=float, default=0.01, help="选取patch的比例，取值为0~1之间。默认为0.01")
    # parser.add_argument('--outputdir',type=str,help="")
    args = parser.parse_args()

    # 参数检查
    if args.select_ratio < 0 or args.select_ratio > 1:
        print("错误：选取patch的比例必须在0~1之间，输入的为{}".format(args.select_ratio))
        exit(-2)

    if "test" in os.listdir(args.datasets) or "train" in os.listdir(args.datasets):
        print("数据集路径形式为模式二，即路径下为test、train，然后其下为视频名称")
        videos= []
        for phase in ["test", "train"]:
            videos += [[i, os.path.join(args.datasets, phase, i)]
                       for i in os.listdir(os.path.join(args.datasets, phase))
                       if os.path.isdir(os.path.join(args.datasets, phase, i))]
        # videos = [[i, os.path.join(args.datasets, "test", i)]
        #           for i in os.listdir(os.path.join(args.datasets, "test"))
        #           if os.path.isdir(os.path.join(args.datasets, "test", i))]
        # videos += [[i, os.path.join(args.datasets, "train", i)]
        #           for i in os.listdir(os.path.join(args.datasets, "train"))
        #           if os.path.isdir(os.path.join(args.datasets, "train", i))]
    else:
        print("数据集路径形式为模式一，即路径下为视频名称")
        videos = [[i, os.path.join(args.datasets, i)]
                  for i in os.listdir(args.datasets)
                    if os.path.isdir(os.path.join(args.datasets, i))]
    print("总共{}个视频".format(len(videos)))
    samples = {}
    for [video, video_dir] in videos:
        print("处理视频{}中。。。".format(video))
        dm_dir = os.path.join(video_dir, "degreemap")
        if not os.path.exists(dm_dir):
            print("错误：degree map不存在！".format(video))
            exit(-1)

        # 读取degree map序列
        imgs_list=read_degree_map(dm_dir)

        # 给所有patch的degree排序
        patches = region_degree_sort(imgs_list, args.patch_size, args.select_ratio)
        # samples.append({"video_name":video, "patch":patches})
        samples[video] = patches

    # 保存为json文件
    json_path = os.path.join(args.datasets, "degreemap_patch_{}_H{}W{}.json"
                             .format(args.select_ratio, args.patch_size[0], args.patch_size[1]))
    with open(json_path, 'w', encoding='utf-8') as file:
        json.dump(samples, file, sort_keys=False, indent=4)
    print("总共耗时：{}s".format(time.time()-start))
