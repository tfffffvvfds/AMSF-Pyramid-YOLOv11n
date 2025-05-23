import os
import random
import shutil

# 数据集路径
image_dir = r'carDataset/images'
label_dir = r'carDataset/labels'

# 创建目标目录
train_image_dir = r'dataset/images/train'
val_image_dir = r'dataset/images/valid'
test_image_dir = r'dataset/images/test'

train_label_dir = r'dataset/labels/train'
val_label_dir = r'dataset/labels/valid'
test_label_dir = r'dataset/labels/test'

# 创建目标文件夹
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(test_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)
os.makedirs(test_label_dir, exist_ok=True)

# 获取所有图片文件
image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

# 打乱文件列表
random.shuffle(image_files)

# 计算每个集的大小
total_files = len(image_files)
train_size = int(total_files * 0.7)
val_size = int(total_files * 0.2)
test_size = total_files - train_size - val_size

# 划分数据集
train_files = image_files[:train_size]
val_files = image_files[train_size:train_size + val_size]
test_files = image_files[train_size + val_size:]


def move_files(files, image_dest_dir, label_dest_dir):
    for file in files:
        image_path = os.path.join(image_dir, file)
        label_path = os.path.join(label_dir, os.path.splitext(file)[0] + '.txt')

        if os.path.exists(image_path) and os.path.exists(label_path):
            shutil.move(image_path, image_dest_dir)
            shutil.move(label_path, label_dest_dir)


# 移动文件到相应目录
move_files(train_files, train_image_dir, train_label_dir)
move_files(val_files, val_image_dir, val_label_dir)
move_files(test_files, test_image_dir, test_label_dir)

print("数据集划分完成")

