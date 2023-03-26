for i in $(find ./ -name *DS*)
do
    echo $i
    rm $i
done

dir="/home/xzf/Projects/Results/STFAN/final/train_low_SSIMloss_test_low/train_low_SSIMloss0.01_test_low/test_out_img"
for video in $(ls -d $dir/*/ |rev | cut -d '/' -f 2|rev )
do
    py="/home/xzf/Projects/STFAN/tools/compare_image_sequences.py"
    img1="/home/xzf/Projects/STFAN/datasets/oppo_low/test/$video/input"
    img2="$dir/$video/"
    img3="/home/xzf/Projects/Results/STFAN/final/train_low_test_low/test_out_img/$video"
    output=$dir"compared/"$video    # if use "_", something is wrong
    echo python $py --output_mode image --output $output --img_dirs $img1 $img2 $img3
    python $py --output_mode image --output $output --img_dirs $img1 $img2 $img3
done

dir="/home/xzf/Projects/STFAN/output/train_real"
for video in $(ls -d $dir/*/ |rev | cut -d '/' -f 2|rev )
do
    py="/home/xzf/Projects/STFAN/compare_image_sequences.py"
    img1="$dir/$video/"
    img2="/home/xzf/Projects/STFAN/datasets/DeepVideoDeblurring_Dataset/qualitative_datasets/test/$video/input"
    output="$dir/$video.avi"
    echo python $py --output $output --img_dirs $img1 $img2
    python $py --frame 10 --output $output --img_dirs $img1 $img2
done

dir="/home/xzf/Projects/STFAN/datasets/our_low/test"
for video in $(ls -d $dir/*/ |rev | cut -d '/' -f 2|rev )
do
    py="/home/xzf/Projects/compare_image_sequences.py"
    img1="$dir/$video/input/"
    img2="/home/xzf/Projects/STFAN/output_our_low/$video/"
    img3="/home/xzf/Projects/STFAN/output_our_low_after/$video/"
    output="$dir/$video.avi"
    echo python $py --output $output --img_dirs $img1 $img2 $img3
    python $py --frame 10 --output $output --img_dirs $img1 $img2 $img3
done 


dir="/home/xzf/Projects/STFAN/datasets/DeepVideoDeblurring_Dataset/test"
for video in $(ls -d $dir/*/ |rev | cut -d '/' -f 2|rev )
do
    py="/home/xzf/Projects/compare_image_sequences.py"
    img1="$dir/$video/input/"
    img2="/home/xzf/Projects/STFAN/output_qualitative_datasets/$video/"
    img3="/home/xzf/Projects/STFAN/output_qualitative_after/$video/"
    output="$dir/$video.avi"
    echo python $py --output $output --img_dirs $img1 $img2 $img3
    python $py --frame 10 --output $output --img_dirs $img1 $img2 $img3
done 


dir="/home/xzf/Projects/STFAN/datasets/our_real/test"
for video in $(ls -d $dir/*/ |rev | cut -d '/' -f 2|rev )
do
    py="/home/xzf/Projects/compare_image_sequences.py"
    img1="$dir/$video/input/"
    img2="/home/xzf/Projects/STFAN/output_our_real/$video/"
    img3="/home/xzf/Projects/STFAN/output_our_real_after/$video/"
    output="$dir/$video.avi"
    echo python $py --output $output --img_dirs $img1 $img2 $img3
    python $py --frame 10 --output $output --img_dirs $img1 $img2 $img3
done 


dir="/home/xzf/Projects/STFAN/datasets/our_super_blur/test"
for video in $(ls -d $dir/*/ |rev | cut -d '/' -f 2|rev )
do
    py="/home/xzf/Projects/compare_image_sequences.py"
    img1="$dir/$video/input/"
    img2="/home/xzf/Projects/STFAN/log_our_super_blur/$video/"
    img3="/home/xzf/Projects/STFAN/log_our_super_blur_after/$video/"
    output="$dir/$video.avi"
    echo python $py --output $output --img_dirs $img1 $img2 $img3
    python $py --frame 10 --output $output --img_dirs $img1 $img2 $img3
done 


dir="/home/xzf/Projects/Datasets/superblur/"
for video in $(ls -d $dir/*/ |rev | cut -d '/' -f 2|rev )
do
    py="/home/xzf/Projects/compare_image_sequences.py"
    img1="$dir$video"
    img2="/home/xzf/Projects/STFAN/output_our_super_blur/$video"
    img3="/home/xzf/Projects/CDVD-TSP/infer_results/super_blur/$video"
    cp $img1/0015.png $img2/0015.png
    cp $img1/0016.png $img2/0016.png
    cp $img1/0017.png $img2/0017.png
    cp $img1/0018.png $img2/0018.png
done 