for i in $(tree -i|grep MP4)
do
  ii=$(echo $i|sed 's/MP4/mp4/g')
  mv $i $ii
done