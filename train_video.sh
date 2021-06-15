CUDA_VISIBLE_DEVICES=6 /data/pyrun/bin/python3.6 trainer/main.py \
 --train True \
 --train_img_path /data/disk2/fuwangyi/data/livevqc/video/ \
 --val_img_path /data/disk2/fuwangyi/data/livevqc/video/ \
 --train_csv_file /data/disk2/fuwangyi/data/livevqc/label/livevqc_train_12.csv  \
 --val_csv_file /data/disk2/fuwangyi/data/livevqc/label/livevqc_test_12.csv  \
 --conv_base_lr 0.005 \
 --dense_lr 0.005 \
