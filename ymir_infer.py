import torch
from ultralytics import YOLO
from ymir_exc.util import  get_weight_files, get_merged_config , write_ymir_monitor_process , YmirStage
from ymir_exc import result_writer as rw
from ymir.util import get_weight_file
import os.path as osp
import cv2

ymir_cfg = get_merged_config()
imgsz = int(ymir_cfg.param.img_size)
device = str(ymir_cfg.param.get('gpu_id', '0'))
monitor_gap = 10
# Load a pretrained YOLO model

model = YOLO(get_weight_file(ymir_cfg))
ymir_infer_result = dict()
with open(ymir_cfg.ymir.input.candidate_index_file, 'r') as f:
    lines = f.readlines()
    dataset_size = len(lines)
    for idx in range(dataset_size):
        if idx % monitor_gap == 0:
            write_ymir_monitor_process(ymir_cfg,
                                task='infer',
                                naive_stage_percent=idx/dataset_size ,
                                stage=YmirStage.TASK)
        line = lines[idx]
        image = line.strip()
        image_name = osp.basename(image)
        anns = []
        results = model.predict(image,imgsz=imgsz,conf=ymir_cfg.param.conf_thres,device=device,iou=ymir_cfg.param.iou_thres)
        boxes = results[0].boxes
        for index in range(len(boxes)):
            conf = boxes[index].conf
            x,y,w,h = boxes[index].xywh.flatten()
            ann = rw.Annotation(class_name=ymir_cfg.param.class_names[int(boxes[index].cls)],
                                            score=conf,
                                            box=rw.Box(x=int(x-w*1/2), y=int(y-h*1/2), w=int(w),
                                                       h=int(h)))
            anns.append(ann)
        ymir_infer_result[image_name] = anns
    rw.write_infer_result(infer_result=ymir_infer_result)