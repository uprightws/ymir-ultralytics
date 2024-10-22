import logging
import os
import subprocess
import sys

from easydict import EasyDict as edict
from ymir_exc import monitor
from ymir_exc.util import (
    YmirStage,
    find_free_port,
    get_bool,
    get_merged_config,
    write_ymir_monitor_process,
)
from ymir_exc.dataset_convert.ymir2yolov5 import convert_ymir_to_yolov5
from ultralytics import YOLO
from shutil import move


def start(cfg: edict) -> int:
    logging.info(f"merged config: {cfg}")

    if cfg.ymir.run_training:
        _run_training(cfg)
    elif cfg.ymir.run_infer:
        _run_infer(cfg)
    else:
        _run_mining(cfg)
    return 0


def move_files(src_dir: str, dst_dir: str) -> None:
    for root, _, files in os.walk(src_dir):
        for file in files:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(dst_dir, file)
            move(src_file, dst_file)


def _run_training(cfg: edict) -> None:
    commands = ["python3"]
    commands.extend(["ymir_train.py"])
    subprocess.run(commands, check=True)
    write_ymir_monitor_process(
        cfg, task="training", naive_stage_percent=1.0, stage=YmirStage.TASK
    )
    monitor.write_monitor_logger(percent=1.0)
    model = YOLO("/out/models/best.pt")
    imgsz = int(cfg.param.img_size)
    opset = int(cfg.param.opset)
    model.export(format="onnx", imgsz=imgsz, opset=opset)


def _run_infer(cfg: edict) -> None:
    commands = ["python3"]
    commands.extend(["ymir_infer.py"])
    subprocess.run(commands, check=True)
    write_ymir_monitor_process(
        cfg, task="infer", naive_stage_percent=1.0, stage=YmirStage.POSTPROCESS
    )


def _run_mining(cfg: edict) -> None:
    commands = ["python3"]
    commands.extend(["ymir_mining.py"])
    subprocess.run(commands, check=True)
    write_ymir_monitor_process(
        cfg, task="mining", naive_stage_percent=1.0, stage=YmirStage.POSTPROCESS
    )


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        format="%(levelname)-8s: [%(asctime)s] %(message)s",
        datefmt="%Y%m%d-%H:%M:%S",
        level=logging.INFO,
    )

    cfg = get_merged_config()
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    sys.exit(start(cfg))
