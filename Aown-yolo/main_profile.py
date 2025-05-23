import warnings
warnings.filterwarnings('ignore')
import torch
from ultralytics import YOLO

if __name__ == '__main__':
    # choose your yaml file
    model = YOLO(r"G:\sc\ultralytics-yolo11-main\Aown-yolo\基准模型配置文件\yolov10s.yaml")
    model.model.eval()
    model.info(detailed=True)
    try:
        model.profile(imgsz=[640, 640])
    except Exception as e:
        print(e)
        pass
    print('after fuse:', end='')
    model.fuse()