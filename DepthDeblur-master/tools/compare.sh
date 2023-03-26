py=/home/xzf/Projects/STFAN/tools/compare_image_sequences.py
output=/home/xzf/Projects/Results/STFAN/output/compare
for i in $(ls *csv)
do  
    echo =======================
    echo python $py --input $i --output $output/${i%.*}
    python $py --input $i --output $output/${i%.*}
done