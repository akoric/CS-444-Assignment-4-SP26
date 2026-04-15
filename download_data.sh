#!/usr/bin/env bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <data_path>"
    exit 1
fi

DATA_DIR="$1"

mkdir -p "$DATA_DIR"

cd "$DATA_DIR" || { echo "Could not change directory to '$DATA_DIR'"; exit 1; }


wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
tar -xf VOCtrainval_06-Nov-2007.tar
mv VOCdevkit VOCdevkit_2007
rm VOCtrainval_06-Nov-2007.tar

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
tar -xf VOCtest_06-Nov-2007.tar
mv VOCdevkit/VOC2007 VOCdevkit_2007/VOC2007test
rmdir VOCdevkit
rm VOCtest_06-Nov-2007.tar


##############################################################################
# In case the above links are down: Install gdown in your environment and use 
# the following
##############################################################################

# gdown --folder https://drive.google.com/drive/folders/1SI9WWYZrAT-vxvuk4qf6yUnWYItjanvX

# cd MP4-Data

# unzip VOCtrainval_06-Nov-2007.zip
# mv VOCdevkit VOCdevkit_2007
# rm VOCtrainval_06-Nov-2007.zip

# unzip VOCtest_06-Nov-2007.zip
# mv VOCdevkit/VOC2007 VOCdevkit_2007/VOC2007test
# rm -r VOCdevkit
# rm VOCtest_06-Nov-2007.zip

# cd ..
# mv MP4-Data/VOCdevkit_2007 VOCdevkit_2007

# rm -r MP4-Data