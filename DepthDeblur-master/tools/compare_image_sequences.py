import argparse
import os
import cv2 as cv
import numpy as np
from PIL import Image

''' using shell for batch processing
python /home/xzf/Projects/STFAN/tools/compare_image_sequences.py --input compare.csv --output ./output
'''
parser = argparse.ArgumentParser(description='compare with several image sequence')
parser.add_argument('--input', type=str, default='compare.csv',
                    help='Optional: preservation format, image or video. Default: image')
parser.add_argument('--output_mode', type=str, default='image',
                    help='Optional: preservation format, image or video. Default: image')
parser.add_argument('--frame', type=int, default=5,
                    help='Optional: the frame of output video. Default: 5')
parser.add_argument('--output', type=str, default='./output',
                    help='Optional: the path of output video or the dirction of images. '
                         'For video, only support .avi format. Default: ./output.avi')
args = parser.parse_args()

# 获取所有需要对比方法的信息
methods = {}
for line in open(args.input, "r"):
    [m, p, t] = line.strip("\n").split(",")
    if t == '':
        print("错误：标签（第3列）为空")
        exit(-1)
    methods[m] = [p, t]
# 获取视频序列名称
first_method = list(methods.keys())[0]
if methods[first_method][1] != "test_out_img":
    first_path = methods[first_method][0]
else:
    first_path = os.path.join(methods[first_method][0], methods[first_method][1])
videos = os.listdir(first_path)

for v in videos:
    paths = {}
    for m in methods.keys():
        if methods[m][1] != "test_out_img":
            paths[m] = os.path.join(methods[m][0], v, methods[m][1])
        else:
            paths[m] = os.path.join(methods[m][0], methods[m][1], v)

    # 读取图像文件列表
    image_format = [".jpg", ".png", ".jpeg"]
    images_path = {}
    for m in paths.keys():
        if os.path.exists(paths[m]):
            print(paths[m], " is valid")
            images_path[m] = [os.path.join(paths[m], i)
                              for i in sorted(os.listdir(paths[m]))  # os.listdir得到的列表可能是乱序的
                              if os.path.splitext(i)[-1] in image_format]  # 根据格式筛选
        else:
            print("错误: dir of input images doesn't exit -- {}. Exiting ... ".format(paths[m]))
            exit()

    # 检查输出参数
    if args.output_mode == "video":
        if args.frame < 1:
            print("错误: input wrong video frame -- {}, it should >= 1. Exiting ... ".format(args.frame))
            exit()
        if ".avi" not in args.output:
            print("错误: input wrong output format of video -- {}, it should be .avi format. Exiting ... "
                  .format(args.output))
            exit()
    elif args.output_mode == "image":
        if not os.path.exists(args.output):
            print("警告: dir of output images doesn't exit -- {}. Automatically created".format(args.output))
            os.makedirs(args.output)  # 创建路径
    else:
        print("错误: input wrong output mode -- {}, it should be image or video. Exiting ... ".format(args.output))
        exit()

    # 检查输入图片路径
    if len(images_path) == 0:
        print("错误: no valid directory")
        exit()

    # 查看图片数量是否一致
    nums = [len(images_path[i]) for i in images_path]
    if nums != sorted(nums):
        print('警告: image nums mismatch ', nums)
        # print('Error: image nums mismatch ', nums)
        # exit()

    # 二维列表转置
    # images_path = [[images_path[i][j] for i in range(len(images_path))] for j in range(len(images_path[0]))]
    # images_path = list(zip(images_path))

    # 读取图片对 并 拼接
    images_cat = []
    for i in range(len(images_path[list(images_path.keys())[0]])):
        image_pair = []
        # 读取图片
        for m in images_path.keys():
            img = cv.imread(images_path[m][i])
            image_pair.append(cv.putText(img, m, (50, 50), cv.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1,
                                         cv.LINE_AA, False))
        # 拼接
        if len(image_pair) < 4:
            images_cat.append(np.hstack(image_pair))
        else:
            if len(image_pair) % 2 != 0:
                image_pair.append(np.zeros(image_pair[0].shape, dtype=np.uint8))
            n = len(image_pair)
            images_cat.append(np.vstack((np.hstack(image_pair[:int(n / 2)]),
                                         np.hstack(image_pair[int(n / 2):]))))
    # 输出
    if args.output_mode == "video":
        # 保存视频
        o = os.path.join(args.output, v)
        if not os.path.exists(o):
            os.makedirs(o)
        video = cv.VideoWriter(o, cv.VideoWriter_fourcc(*'XVID'), args.frame,
                               images_cat[0].shape[:2][::-1], True)  # 图片维度为h*w*c，而视频要求输入w*h
        for img in images_cat:
            video.write(img)
        video.release()

    elif args.output_mode == "image":
        o = os.path.join(args.output, v)
        if not os.path.exists(o):
            os.makedirs(o)
        i = 1
        for img in images_cat:
            save_path = os.path.join(o, str(i).zfill(5) + ".png")
            cv.imwrite(save_path, img)
            i = i + 1

    print("=" * 10, " Done ", "=" * 10)
