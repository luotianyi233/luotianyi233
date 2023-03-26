# STFAN
for i in $(ls -R ./ | grep / | sed 's/://g')
# for i in $(ls -R ./ | grep / | sed 's/://g')
do
    echo $i
    video=$(echo $i|cut -d '/' -f 2)
    echo $dirname $video
    dir="/home/xzf/Projects/STFAN/datasets/video/test/$video/input/"
    if [ ! -d "$dir" ]; then
        mkdir -p $dir
    fi
    cp -rf $i/* $dir
done


# CDVD
for i in $(ls -R ./ | grep / | sed 's/://g')
do
    # echo $i
    video=$(echo $i|cut -d '/' -f 2)
    # echo $dirname
    # blur
    dir_blur="/home/xzf/Projects/CDVD-TSP/dataset/DVD/test/blur/$video/"
    if [ ! -d "$dir_blur" ]; then
        mkdir -p $dir_blur
    fi
    cp -rf $i/* $dir_blur
    # gt
    dir_gt="/home/xzf/Projects/CDVD-TSP/dataset/DVD/test/gt/$video/"
    if [ ! -d "$dir_gt" ]; then
        mkdir -p $dir_gt
    fi
    cp -rf $i/* $dir_gt
done