from ultralytics import YOLO

# 加载模型
model = YOLO(r"G:\thy\语智电网监护仪源码\whisper_yolo_erweima\data\best.pt")  # 这里可以替换为你的模型权重文件路径，如"path/to/your/weights.pt"

# 对图像进行预测
results = model.predict(source=r"D:\Users\Administrator\Desktop\Project\ultralytics-main\dataset\images\test",  # 替换为你要预测的图像路径
                        save=True,  # 是否保存预测结果图像
                        conf=0.25,   # 置信度阈值
                        # show=True
                        )

# 查看预测结果
for r in results:
    print(r.boxes)  # 打印预测框信息
    print(r.names)  # 打印类别名称