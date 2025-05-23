import cv2
import numpy as np

# 读取图片
img = cv2.imread(r"G:\sc\ultralytics-yolo11-main\2\train\images\101_jpg.rf.952226261c4aea780a4db6b22196c2b5.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
original = img.copy()

# 1. 灰度化与降噪
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# 2. 裂缝检测（边缘检测）
canny = cv2.Canny(gray, 50, 150)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
crack_features = cv2.morphologyEx(canny, cv2.MORPH_DILATE, kernel)

# 3. 锈迹检测（颜色阈值）
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
lower_rust = np.array([0, 50, 50])
upper_rust = np.array([20, 255, 255])
rust_mask = cv2.inRange(hsv, lower_rust, upper_rust)
rust_features = cv2.morphologyEx(rust_mask, cv2.MORPH_DILATE, kernel)

# 4. 合并缺陷特征
defect_map = cv2.addWeighted(crack_features, 0.7, rust_features, 0.3, 0)
defect_map_color = cv2.cvtColor(defect_map, cv2.COLOR_GRAY2RGB)

# 5. 可视化标注
for y in range(defect_map.shape[0]):
    for x in range(defect_map.shape[1]):
        if defect_map[y, x] > 0:
            cv2.circle(original, (x, y), 1, (0, 0, 255), -1)

# 显示结果
cv2.imshow('Defect Feature Map', cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()