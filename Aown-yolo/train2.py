
from ultralytics import YOLO
from ultralytics import RTDETR
import torch

if __name__=="__main__":
    # Load a model
    model_yaml=r"G:\sc\ultralytics-yolo11-main\Aown-yolo\基准模型配置文件\yolo11.yaml"
    data_yaml=r"G:\sc\ultralytics-yolo11-main\visdrone2019\data.yaml"
  #  pre_model=r"F:\Users\Administrator\Desktop\JQY\project\JQYultralytics-main\yolov8n.pt"

    # 首先，使用 YOLO 类，传入 model_yaml 配置文件路径以及指定任务类型为 detect
    # （明确告诉模型此次是用于目标检测任务，因为 YOLO 模型可以支持多种不同类型的视觉任务，所以需要指定具体的任务场景）来实例化一个 YOLO 模型对象。
    # 这个过程中，模型会依据配置文件中的信息构建出相应的网络结构，例如确定有多少层、每层的卷积核大小等参数设置。
    # 然后，调用实例化后的模型对象的 load 方法，将 pre_model 所指向的预训练权重文件加载到刚刚构建好的模型上，
    # 完成权重的迁移过程。这样一来，模型就既有了预定义的网络架构，又具备了之前通过大量数据训练得到的初始权重参数，为后续的进一步训练做好了准备。
   # model = YOLO(model_yaml,task='detect').load(pre_model)  # build from YAML and transfer weights
    model = YOLO(model_yaml)
   #  model = RTDETR(model_yaml, task='detect')

    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=400,
        imgsz=640,
        batch=-1,
        workers=4,
        patience=100
        # optimizer='SGD',
        # lr0=0.001,
        # momentum=0.95

    )
    # project='my_custom_path', name='my_train_run'
    # 这样，训练结果将会保存在my_custom_path\my_train_run路径下。
    # 其中project参数可以理解为一个更高层次的目录，用于对不同类型的项目进行分类；name参数则是在project目录下的具体运行名称相关的子目录。
