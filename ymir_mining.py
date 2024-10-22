import torch
from ultralytics import YOLO
from ymir_exc.util import  get_weight_files, get_merged_config , write_ymir_monitor_process , YmirStage
from ymir_exc import result_writer as rw
from ymir.util import get_weight_file
import os.path as osp
import numpy as np


ymir_cfg = get_merged_config()
imgsz = int(ymir_cfg.param.img_size)
device = str(ymir_cfg.param.get('gpu_id', '0'))
monitor_gap = 10
# Load a pretrained YOLOv8n model


model = YOLO(get_weight_file(ymir_cfg))
ymir_mining_result = []

with open(ymir_cfg.ymir.input.candidate_index_file, 'r') as f:
    lines = f.readlines()
    dataset_size = len(lines)
    for idx in range(dataset_size):
        if idx % monitor_gap == 0:
            write_ymir_monitor_process(ymir_cfg,
                                task='mining',
                                naive_stage_percent=idx/dataset_size ,
                                stage=YmirStage.TASK)
        line = lines[idx]
        image = line.strip()
        image_name = osp.basename(image)
        anns = []
        results = model.predict(image,imgsz=imgsz,device=device)
        boxes = results[0].boxes
        if len(boxes):
            conf = boxes.conf.data.cpu().numpy()
            score = -np.sum(conf * np.log2(conf)) 
        else:
            score = -10
        ymir_mining_result.append((image_name, score))
       
        
    rw.write_mining_result(mining_result=ymir_mining_result)