#python trainstylenet.py --dataset-images {dataset_images} --styles-dir {styles_dir} --num-workers {num_workers} --model-dir {model_dir} --eval-image-dir {eval_image_dir} 
python trainstylenet.py --dataset-images /data/public_data/COCO2017/images/train2017/ --styles-dir ./dataset/styles --num-workers 3 --model-dir ./result/ --eval-image-dir dataset/images/trump.jpg
