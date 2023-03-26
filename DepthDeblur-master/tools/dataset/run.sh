#!/bin/bash
# 查看文件个数
dir=/home/xzf/Projects/STFAN/datasets/oppo_low_addtion/second_final
for input in $(ls $dir ); do
    echo -e $input"\t"$(tree $input|grep -c png)
done

# 查看视频帧数
dir=/home/xzf/Projects/STFAN/datasets/oppo_low_addtion/original_second
for input in $(find $dir -name "*.MP4"|sort); do
    # https://qastack.cn/superuser/84631/how-do-i-get-the-number-of-frames-in-a-video-on-the-linux-command-line
    frame=$(ffmpeg -i $input -vcodec copy -f rawvideo -y /dev/null 2>&1 | tr ^M '\n' | awk '/^frame=/ {print $2}'|tail -n 1)
    echo "${input##*/} -> $frame"
done

# 视频连续。将原始视频使用ffmpeg编码一次,保证帧的连续,防止后续使用opencv读取视频的时候失败
dir=/home/xzf/Projects/STFAN/datasets/oppo_low_addtion/original_second
for input in $(find $dir -name "*.MP4"|sort); do
    echo "${input##*/}"
    ffmpeg -i $input -vcodec copy -an $input -y -hide_banner
done

# 数据集生成
dir=/home/xzf/Projects/STFAN/datasets/oppo_low_addtion
input_folder=insert9_first_bit30000
output_folder=done9_first_bit30000
pyfile=/home/xzf/Projects/STFAN/tools/dataset/datasets_generate.py
n=15
for input in $(find $dir/$input_folder -name "*.MP4"); do
    output=$(echo $input | sed "s/$input_folder/$output_folder/g")
    output=$(echo ${output%.*}) # 删掉路径中.的右边，相当于去掉文件格式
    # if [ ! -d "$output_dir" ];then
    #     echo "Create a directory "$output
    #     mkdir -p $output
    # fi
    echo "input: $input => output: $output"
    log=$(echo $input | sed "s/.MP4/_2.txt/g")
    e cho python $pyfile --input $input --output $output
    python $pyfile --input $input --output $output > $log &
    sleep 2s    # 等一会儿
    while true; do
        if [ $(ps -aux | grep 'datasets_generate' | grep -v color | wc -l) -ge $n ]; then # 最多允许n个程序同时跑。-ge表示大于等于
            echo "已有$(ps -aux | grep 'datasets_generate' | grep -v color | wc -l)个程序在跑，到达上限$n，等10分钟"
            sleep 10m
        else
            echo "已有$(ps -aux | grep 'datasets_generate' | grep -v color | wc -l)个程序在跑，少于上限$n"
            break
        fi
    done
done

# 插帧。验证通过
dir=/home/xzf/Projects/STFAN/datasets/oppo_low_addtion
input_folder=cut_first
output_folder=insert9_first_bit30000                     # 结尾的数字表示插帧的数量
n=10                                       # 同时跑的程序数量。在非0卡跑的程序，除了要在指定卡上占用2349M显存，还会在0卡上占用1081M显存（不知道为什么）。所以2080ti上n<=9，P100上n<=12
cd /home/xzf/Projects/STFAN/sepconv-slomo # 必须到项目目录里运行run.py，否则会报错找不到module
for input in $(find $dir/$input_folder -name "*.MP4"); do
    output=$(echo $input | sed "s/$input_folder/$output_folder/g")
    output_log=${output%.*}".txt"
    # output_dir=${output%/*}    # 获取输出视频所在的目录。这两个等价
    output_dir=$(dirname $output)
    if [ ! -d "$output_dir" ]; then
        echo "Create a directory "$output_dir
        mkdir -p $output_dir
    fi
    echo "input: $input => output: $output"
    echo python ./run.py --model lf --video $input --out $output
    # python ./run.py --model lf --video $input --out $output
    python ./run.py --model lf --video $input --out $output --frame 9 > $output_log &
    echo "给程序一点时间来占用显存"
    sleep 20s # 给程序一点时间来占用显存
    while true; do
        if [ $(ps -aux | grep 'model lf' | grep -v color | wc -l) -ge $n ]; then # 最多允许n个程序同时跑。-ge表示大于等于
            echo "已有$(ps -aux | grep 'model lf' | grep -v color | wc -l)个程序在跑，到达上限$n，等10分钟"
            sleep 10m
        else
            echo "已有$(ps -aux | grep 'model lf' | grep -v color | wc -l)个程序在跑，少于上限$n"
            break
        fi
    done
done

# 视频降采样 + 去音频
dir=/home/xzf/Projects/STFAN/datasets/oppo_low_addtion
input_folder=original_first
output_folder=720p_first
n=5                                      # 同时跑的程序数量
for input in $(find $dir/$input_folder -name "*.MP4"); do
    if [ -n "$(echo $input | grep trash)" ]; then
        continue
    fi
    output=$(echo $input | sed "s/$input_folder/$output_folder/g")
    # output_dir=${output%/*}    # 获取输出视频所在的目录。这两个等价
    output_dir=$(dirname $output)
    if [ ! -d "$output_dir" ]; then
        echo "Create a directory "$output_dir
        mkdir -p $output_dir
    fi
    echo "input: $input => output: $output"
    # ffmpeg -i $input -vf scale=1280:720 -profile:v -an $output -hide_banner
    # ffmpeg -i $input -vf scale=1280:720 -preset placebo -an $output -hide_banner
    # ffmpeg -i $input -vf scale=1280:720 -crf 13 -an $output -hide_banner -y
    # ffmpeg -i $input -vf scale=1280:720 -b:v 30000k -an $output -hide_banner -y     # 方法一: 降采样
    ffmpeg -i $input -vf crop=1280:720:320:180 -b:v 30000k -an $output -hide_banner -y -threads 20      # 方法二: 裁剪
done

# 去音频。验证通过
dir=/home/xzf/Projects/STFAN/datasets/oppo_low_addtion/temp
input_folder=cut
output_folder=cut_noaudio
for input in $(find $dir/$input_folder -name "*.MP4"); do
    output=$(echo $input | sed "s/$input_folder/$output_folder/g")
    output_dir=$(dirname $output) # 获取输出视频所在的目录
    if [ ! -d "$output_dir" ]; then
        echo "Create a directory "$output_dir
        mkdir -p $output_dir
    fi
    echo "input: $input => output: $output"
    ffmpeg -i $input -map 0:0 -vcodec copy $output
done

# 视频转图片序列
dir=/mnt/d
input_folder=test
for input in $(find $dir/$input_folder -name "*.MP4"); do
    output_dir=${input%.*}  # 删除后缀就是输出目录 https://blog.csdn.net/ljianhui/article/details/43128465
    if [ ! -d "$output_dir" ]; then
        echo "Create a directory "$output_dir
        mkdir -p $output_dir
    fi
    echo "input: $input => output: $output_dir"
    ffmpeg -i $input $output_dir/%04d.png
done

# MP4转mp4
dir=/home/xzf/Projects/STFAN/datasets/oppo_low_addtion/temp
input_folder=cut
for input in $(find $dir/$input_folder -name "*.MP4"); do
    output=$(echo $i | sed 's/MP4/mp4/g')
    echo "input: $input => output: $output"
    mv $input $output
done

commond=(
    'mkdir -p /home/xzf/Projects/Datasets/STFAN/oppo_low_addtion/cut/sunset/'
    'mkdir -p /home/xzf/Projects/Datasets/STFAN/oppo_low_addtion/cut/night/'
    'mkdir -p /home/xzf/Projects/Datasets/STFAN/oppo_low_addtion/cut/market/'
    'ffmpeg -i /home/xzf/Projects/Datasets/STFAN/oppo_low_addtion/720p/sunset/GH010452.MP4 -vf "select=between(n\,0\,2145)" -y -an -b:v 30000k /home/xzf/Projects/Datasets/STFAN/oppo_low_addtion/cut/sunset/GH010452_1.MP4 -hide_banner -y &'
    'ffmpeg -i /home/xzf/Projects/Datasets/STFAN/oppo_low_addtion/720p/sunset/GH010452.MP4 -vf "select=between(n\,2272\,2327)" -y -an -b:v 30000k /home/xzf/Projects/Datasets/STFAN/oppo_low_addtion/cut/sunset/GH011452_2.MP5 -hide_banner -y &'
    'ffmpeg -i /home/xzf/Projects/Datasets/STFAN/oppo_low_addtion/720p/sunset/GH010457.MP4 -vf "select=between(n\,0\,344)" -y -an -b:v 30000k /home/xzf/Projects/Datasets/STFAN/oppo_low_addtion/cut/sunset/GH012457.MP6 -hide_banner -y &'
    'ffmpeg -i /home/xzf/Projects/Datasets/STFAN/oppo_low_addtion/720p/sunset/GH010460.MP4 -vf "select=between(n\,0\,899)" -y -an -b:v 30000k /home/xzf/Projects/Datasets/STFAN/oppo_low_addtion/cut/sunset/GH013460_1.MP7 -hide_banner -y &'
    'ffmpeg -i /home/xzf/Projects/Datasets/STFAN/oppo_low_addtion/720p/sunset/GH010460.MP4 -vf "select=between(n\,1484\,2722)" -y -an -b:v 30000k /home/xzf/Projects/Datasets/STFAN/oppo_low_addtion/cut/sunset/GH014460_2.MP8 -hide_banner -y &'
    'ffmpeg -i /home/xzf/Projects/Datasets/STFAN/oppo_low_addtion/720p/sunset/GH010465.MP4 -vf "select=between(n\,0\,1442)" -y -an -b:v 30000k /home/xzf/Projects/Datasets/STFAN/oppo_low_addtion/cut/sunset/GH015465.MP9 -hide_banner -y &'
    'ffmpeg -i /home/xzf/Projects/Datasets/STFAN/oppo_low_addtion/720p/night/GH010479.MP4 -vf "select=between(n\,628\,2941)" -y -an -b:v 30000k /home/xzf/Projects/Datasets/STFAN/oppo_low_addtion/cut/night/GH016479.MP10 -hide_banner -y &'
    'ffmpeg -i /home/xzf/Projects/Datasets/STFAN/oppo_low_addtion/720p/night/GH010485.MP4 -vf "select=between(n\,0\,1811)" -y -an -b:v 30000k /home/xzf/Projects/Datasets/STFAN/oppo_low_addtion/cut/night/GH017485_1.MP11 -hide_banner -y &'
    'ffmpeg -i /home/xzf/Projects/Datasets/STFAN/oppo_low_addtion/720p/night/GH010485.MP4 -vf "select=between(n\,2468\,3370)" -y -an -b:v 30000k /home/xzf/Projects/Datasets/STFAN/oppo_low_addtion/cut/night/GH018485_2.MP12 -hide_banner -y &'
    'ffmpeg -i /home/xzf/Projects/Datasets/STFAN/oppo_low_addtion/720p/market/GH010489.MP4 -vf "select=between(n\,0\,987)" -y -an -b:v 30000k /home/xzf/Projects/Datasets/STFAN/oppo_low_addtion/cut/market/GH019489_1.MP13 -hide_banner -y &'
    'ffmpeg -i /home/xzf/Projects/Datasets/STFAN/oppo_low_addtion/720p/market/GH010489.MP4 -vf "select=between(n\,1845\,2067)" -y -an -b:v 30000k /home/xzf/Projects/Datasets/STFAN/oppo_low_addtion/cut/market/GH020489_2.MP14 -hide_banner -y &'
    'ffmpeg -i /home/xzf/Projects/Datasets/STFAN/oppo_low_addtion/720p/market/GH010496.MP4 -vf "select=between(n\,0\,2222)" -y -an -b:v 30000k /home/xzf/Projects/Datasets/STFAN/oppo_low_addtion/cut/market/GH021496.MP15 -hide_banner -y &'
    'ffmpeg -i /home/xzf/Projects/Datasets/STFAN/oppo_low_addtion/720p/market/GH010501.MP4 -vf "select=between(n\,0\,2289)" -y -an -b:v 30000k /home/xzf/Projects/Datasets/STFAN/oppo_low_addtion/cut/market/GH022501_1.MP16 -hide_banner -y &'
    'ffmpeg -i /home/xzf/Projects/Datasets/STFAN/oppo_low_addtion/720p/market/GH010501.MP4 -vf "select=between(n\,2310\,2534)" -y -an -b:v 30000k /home/xzf/Projects/Datasets/STFAN/oppo_low_addtion/cut/market/GH023501_2.MP17 -hide_banner -y &'
    'ffmpeg -i /home/xzf/Projects/Datasets/STFAN/oppo_low_addtion/720p/market/GH010502.MP4 -vf "select=between(n\,0\,1009)" -y -an -b:v 30000k /home/xzf/Projects/Datasets/STFAN/oppo_low_addtion/cut/market/GH024502_1.MP18 -hide_banner -y &'
    'ffmpeg -i /home/xzf/Projects/Datasets/STFAN/oppo_low_addtion/720p/market/GH010502.MP4 -vf "select=between(n\,1069\,1472)" -y -an -b:v 30000k /home/xzf/Projects/Datasets/STFAN/oppo_low_addtion/cut/market/GH025502_2.MP19 -hide_banner -y &'
    'ffmpeg -i /home/xzf/Projects/Datasets/STFAN/oppo_low_addtion/720p/market/GH010502.MP4 -vf "select=between(n\,1683\,3038)" -y -an -b:v 30000k /home/xzf/Projects/Datasets/STFAN/oppo_low_addtion/cut/market/GH026502_3.MP20 -hide_banner -y &'
    'ffmpeg -i /home/xzf/Projects/Datasets/STFAN/oppo_low_addtion/720p/market/GH010513.MP4 -vf "select=between(n\,0\,946)" -y -an -b:v 30000k /home/xzf/Projects/Datasets/STFAN/oppo_low_addtion/cut/market/GH027513.MP21 -hide_banner -y &'
    'ffmpeg -i /home/xzf/Projects/Datasets/STFAN/oppo_low_addtion/720p/market/GH010514.MP4 -vf "select=between(n\,0\,1559)" -y -an -b:v 30000k /home/xzf/Projects/Datasets/STFAN/oppo_low_addtion/cut/market/GH028514_1.MP22 -hide_banner -y &'
    'ffmpeg -i /home/xzf/Projects/Datasets/STFAN/oppo_low_addtion/720p/market/GH010514.MP4 -vf "select=between(n\,1850\,2636)" -y -an -b:v 30000k /home/xzf/Projects/Datasets/STFAN/oppo_low_addtion/cut/market/GH029514_2.MP23 -hide_banner -y &'
)
for i in ${commond[*]}; do
    echo $i
done

nohup ffmpeg -i /home/xzf/Projects/Datasets/STFAN/oppo_low_addtion/720p/sunset/GH010452.MP4 -vf "select=between(n\,0\,2145)" -y -an -b:v 30000k /home/xzf/Projects/Datasets/STFAN/oppo_low_addtion/cut/sunset/GH010452_1.MP4 -hide_banner -y 1>/dev/null 2>&1 &
