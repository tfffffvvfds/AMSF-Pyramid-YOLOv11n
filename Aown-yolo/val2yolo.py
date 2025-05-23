import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# 精度小数点保留位数修改问题可看<使用说明.md>下方的<YOLOV8源码常见疑问解答小课堂>第五点
# 最终论文的参数量和计算量统一以这个脚本运行出来的为准
# yolo推理脚本,因为此ultralytics版本较低 所以只适用于v8以及以下版本
if __name__ == '__main__':
    model = YOLO(r"G:\sc\ultralytics-yolo11-main\Aown-yolo\runs\detect\train68\weights\best.pt")
    model.val(data='./data.yaml',
              split='test', # split可以选择train、val、test 根据自己的数据集情况来选择.
              imgsz=640,
              batch=4,
              save_json=True, # if you need to cal coco metrice
              project='../runs/val',
              name='exp',
              )