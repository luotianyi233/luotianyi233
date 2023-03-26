import cv2
import os
import numpy as np
import argparse
import time

parser = argparse.ArgumentParser(description='cut and downscale video')
parser.add_argument('--input', type=str, help='The path of input video')
parser.add_argument('--output', type=str, help='The path of output video')
parser.add_argument('--frame', type=int, default=120, help='输出视频的帧数')
parser.add_argument('--start', type=int, default=1, help='剪切的起始帧数')
parser.add_argument('--end', type=int, default=999999, help='剪切的终止帧数')
args = parser.parse_args()

output_dir=os.path.split(args.output)[0]
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

video_in = cv2.VideoCapture(args.input)
video_out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), args.frame, (1280, 720))
idx = 1

print("="*50)
print("{} -> {}".format(args.input, args.output))

start_time = time.time()
while video_in.isOpened():
    ret, img = video_in.read()
    if ret:
        if args.start <= idx < args.end:
            # img_720p = cv2.resize(img,(1280,720))
            video_out.write(img)
    else:
        print("Warining: reading {}th frame filed".format(idx))
        break
    idx = idx + 1
video_in.release()
video_out.release()
print("总共耗时: {}".format(time.time()-start_time))