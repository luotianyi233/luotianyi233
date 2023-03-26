import os
import cv2
import argparse

parser = argparse.ArgumentParser(description='convert videos to image sequences')
parser.add_argument('--video', type=str, help='The folder of videos')
parser.add_argument('--imgseq', type=str, default='./', help='The folder of output image sequences. Default: ./')
parser.add_argument('--type', type=str, default='png', help='The type of output image sequences. Default: png')
args = parser.parse_args()

video_type = ['.mp4', '.avi']
videos_path = {}
if os.path.exists(args.video):
    print(args.video, " is valid")
    for i in sorted(os.listdir(args.video)):
        filename, type = os.path.splitext(i)
        if type in video_type:
            videos_path[filename] = os.path.join(args.video, i)

if len(videos_path) == 0:
    print("Error: could not find any video. Only support ", video_type)
    exit()

for (name, video_path) in videos_path.items():
    video = cv2.VideoCapture(video_path)
    id = 0
    images_path = os.path.join(args.imgseq, name)
    if not os.path.exists(images_path):
        os.makedirs(images_path)
    while video.isOpened():
        ret, img = video.read()
        if ret:
            image_name = str(id).zfill(4) + "." + args.type
            image_path = os.path.join(images_path, image_name)
            print(image_path)
            cv2.imwrite(image_path, img)
            id += 1
        else:
            break
    video.release()
