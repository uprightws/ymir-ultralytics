# from ultralytics.yolo.v8.detect import DetectionTrainer
# from ultralytics.engine import train
import logging
import os
import subprocess
import sys

# from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

from easydict import EasyDict as edict
from ymir_exc import monitor
from ymir_exc.util import (YmirStage, find_free_port, get_bool, get_merged_config, write_ymir_monitor_process)
from ymir_exc.dataset_convert.ymir2yolov5 import convert_ymir_to_yolov5
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
ymir_cfg = get_merged_config()

print(ymir_cfg)

# data_yaml = '/out/data.yaml'
model = ymir_cfg.param.model
epochs = int(ymir_cfg.param.epochs)
batch = int(ymir_cfg.param.total_batch_size)
imgsz = int(ymir_cfg.param.img_size)
device = str(ymir_cfg.param.get('gpu_id', '0'))
save_period = int(ymir_cfg.param.get('save_period', '-1'))
if not os.path.isfile('/out/data.yaml'):
    data_yaml = convert_ymir_to_yolov5(ymir_cfg)
else:
    data_yaml = '/out/data.yaml'
trainer = DetectionTrainer(overrides={'data':data_yaml,'model':model,'epochs':epochs,'batch':batch,'imgsz':imgsz,'device':device,'save_dir':'/out/models','tensorboard_dir':'/out/tensorboard/','save_period':save_period})
trainer.train()