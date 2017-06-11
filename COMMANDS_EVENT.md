DATASET_DIR=/Users/ahmetkucuk/Documents/Research/Solar_Image_Classification/Bbox_Data/
OUTPUT_DIR=/Users/ahmetkucuk/Documents/Research/Solar_Image_Classification/tf_records/

DATASET_DIR=/home/ahmet/workspace/data/full_disk_171/
OUTPUT_DIR=/home/ahmet/workspace/data/full_disk_171_tfrecords/
python tf_convert_data.py \
    --dataset_name=event \
    --dataset_dir=${DATASET_DIR} \
    --output_dir=${OUTPUT_DIR}

DATASET_DIR=/Users/ahmetkucuk/Documents/Research/Solar_Image_Classification/tfrecords/

DATASET_DIR=/home/ahmet/workspace/data/full_disk_171_tfrecords/
TRAIN_DIR=/home/ahmet/workspace/tensorboard/detection_ssd
python train_ssd_network.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=event \
    --dataset_split_name=train \
    --model_name=ssd_512_vgg \
    --save_summaries_secs=60 \
    --save_interval_secs=600 \
    --weight_decay=0.0005 \
    --optimizer=adam \
    --learning_rate=0.001 \
    --learning_rate_decay_factor=0.95 \
    --batch_size=32
    
 