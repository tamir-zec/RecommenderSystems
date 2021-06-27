#!/bin/bash

# Command line arguments
dataset_name=$1 # Human-friendly dataset name. Use the same in `hyper_params.py`
data_file_path=$2 # Path of data file

# Adjust as per your requirement
k_core="5"
perc_reviews="90"

# Where to store data?
cwd=$(pwd)
data_store_path=$cwd"/data/"
quick_data_deepconn_path=$cwd"/quick_data_deepconn/"
quick_data_narre_path=$cwd"/quick_data_narre/"

cd data_scripts;

# Calling below script will:
# 	- Create train, test, val sets
# 	- Train word2vec on all *train* reviews
echo -e "\e[31m\n\nMaking train/test/val splits..\n\n\e[0m"
python3 preprocess_random_split.py $dataset_name $data_file_path $k_core $perc_reviews $data_store_path

# Calling below script will:
#	- Create .hdf5 files that create padded reviews for train/test/val sets ONLY for NARRE
#	- These .hdf5 files will result in faster training/evaluation
echo -e "\e[31m\n\nMaking quick data for NARRE..\n\n\e[0m"
python3 make_quick_data.py $dataset_name $k_core $perc_reviews $quick_data_narre_path $data_store_path

cd ..;