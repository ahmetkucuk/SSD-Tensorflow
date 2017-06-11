DATASET_DIR=/Users/ahmetkucuk/Documents/Research/Solar_Image_Classification/Bbox_Data/
OUTPUT_DIR=/Users/ahmetkucuk/Documents/Research/Solar_Image_Classification/tf_records/
python tf_convert_data.py \
    --dataset_name=event \
    --dataset_dir=${DATASET_DIR} \
    --output_dir=${OUTPUT_DIR}

DATASET_DIR=/Users/ahmetkucuk/Documents/Research/Solar_Image_Classification/tfrecords/
TRAIN_DIR=./logs/ssd_512_vgg_3
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
    --batch_size=2
    
 