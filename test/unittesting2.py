from ultralytics import YOLO
import cv2

# 加载 YOLO 模型
model = YOLO('../runs/detect/train3/weights/best.pt')  # 使用 YOLOv8 的 nano 版本模型（体积小，速度快）

# 读取图像
image_path = '../assets/微信截图_20241121130226.png'
img = cv2.imread(image_path)

# 推理（检测目标）
results = model(img)

# 在图像上绘制检测结果
annotated_img = results[0].plot()



# 显示结果
cv2.imshow('YOLO Detection', annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
