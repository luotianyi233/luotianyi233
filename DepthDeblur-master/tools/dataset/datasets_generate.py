import os
import cv2
import argparse
import numpy as np
import random

# 例子
# python /home/xzf/Projects/STFAN/tools/dataset/datasets_generate.py --input /home/xzf/Projects/STFAN/datasets/oppo_low_addtion/cut/sunset/GH010452_1.MP4 --output /home/xzf/Projects/STFAN/datasets/oppo_low_addtion/done/sunset/GH010452_1


def check_path(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print("路径不存在，已创建 -> {}".format(directory))
    # else:
    #     print("\033[31m警告\033[0m：路径已存在 -> {}".format(directory))


def imgs_save(imgs_list: list, dir: str, idx: int):
    imgs = np.array(imgs_list)  # [n, h, w, c]

    sharp = imgs_list[int(len(imgs_list)/2)]    # 取中间的作为
    blur = np.mean(imgs, axis=0)
    degree_map = np.mean(np.std(imgs, axis=0), axis=2)

    if degree_map.max() > 255:
        print("\033[31midx\033[0m: {}\tdegree map max:{}".format(idx, degree_map.max()))

    sharp_path = os.path.join(dir, "GT", "{:05d}.png".format(idx))
    blur_path = os.path.join(dir, "blur", "{:05d}.png".format(idx))
    degree_map_path = os.path.join(dir, "degreemap", "{:05d}.png".format(idx))
    check_path(sharp_path); check_path(blur_path); check_path(degree_map_path)
    cv2.imwrite(sharp_path, sharp)
    cv2.imwrite(blur_path, blur)
    cv2.imwrite(degree_map_path, degree_map)

    # print("sharp: {}\nblur: {}\ndegree map: {}".format(sharp_path, blur_path, degree_map_path))


if __name__ == '__main__':
    print("=" * 50)
    parser = argparse.ArgumentParser(description='calculate degreemap')
    parser.add_argument('--input', type=str, help='The path of input video')
    parser.add_argument('--outputdir', type=str, default='./', help='The folder of output image sequences.')
    parser.add_argument('--frame', type=int, default=-1, help='The number of output image sequences. ')
    args = parser.parse_args()

    # 融合的帧数在35~49之间（为奇数，因为GT取中间那张）。
    # 因为插帧的frame=9，相当于是1帧变8帧（也是相邻2帧变9帧），而且拍摄帧率为120fps，是30fps的4倍数
    # 所以融合的帧数大概为8*4=32，可以大致达到变为30fps的效果，但是考虑到拍摄时移动速度（通常很慢），所以融合帧数有一个范围。
    # 可以同时运行多个本程序，从而产生不同的模糊，后续人工筛选保留想要的即可
    if args.frame == -1:
        frame = random.randint(18, 24) * 2 + 1
    elif ((args.frame - 1) % 2 == 0) and args.frame > 0:
        frame = args.frame
    else:
        print("错误：融合的帧数{}必须为奇数，且>0".format(frame))
        exit
    print("融合帧数为{}".format(frame))

    # 输出图像的文件夹
    outdir = os.path.splitext(args.outputdir)[0] + "_" + str(frame)

    # 从视频读取图片
    video = cv2.VideoCapture(args.input)
    idx = 0
    imgs_list = []
    print("strating...")
    while video.isOpened():
        ret, img = video.read()
        idx += 1
        if ret:
            imgs_list.append(img)
            if idx % frame == 0:
                imgs_save(imgs_list, outdir, int(idx / frame))
                imgs_list = []
        else:
            print("\033[31mWarining\033[0m: reading {}th frame filed".format(idx))
            break
