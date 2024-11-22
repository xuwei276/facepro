from ultralytics import YOLO
import cv2

# 加载 YOLO 模型
model = YOLO('../runs/detect/train3/weights/best.pt')  # 使用 YOLOv8 的 nano 版本模型（体积小，速度快）

# 读取图像
image_path = '../assets/微信截图_20241121130218.png'
img = cv2.imread(image_path)

# 推理（检测目标）
# 推理
outputs = model.predict(source=image_path, conf=0.7)

# 提取结果
results = []
for detection in outputs[0].boxes.data.tolist():
    x1, y1, x2, y2, confidence, class_id = detection
    results.append({
        'bbox': [x1, y1, x2, y2],
        'confidence': confidence,
        'class': int(class_id)
    })
print(results)
# 遍历检测结果
for result in results:
    bbox = result['bbox']
    confidence = result['confidence']
    label = result['class']

    # 只绘制置信度大于 0.7 的区域
    if confidence > 0.0:
        x1, y1, x2, y2 = map(int, bbox)

        # 绘制矩形框 (绿色)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 添加标签和置信度
        text = f"{label}: {confidence:.2f}"
        cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# 显示图片
cv2.imshow('Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()