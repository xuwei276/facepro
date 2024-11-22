from insightface.app import FaceAnalysis
import cv2
import numpy as np

# 初始化 FaceAnalysis

app = FaceAnalysis(allowed_modules=['detection', 'recognition', 'genderage'])  # 启用检测和识别功能
app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 表示使用 GPU，det_size 设置检测分辨率

# 用 OpenCV 加载图片
img1 = cv2.imread("../assets/t1.png")

# 检测人脸并提取特征
faces1 = app.get(img1)


# 如果有检测到人脸，绘制矩形框
for face in faces1:
    kps = face['kps']  # 关键点坐标
    print(f"嘴巴左侧点坐标: {kps[3]}")
    print(f"嘴巴右侧点坐标: {kps[4]}")

    # 可视化嘴巴位置
    for i in [3, 4]:  # 绘制嘴巴左、右点
        cv2.circle(img1, (int(kps[i][0]), int(kps[i][1])), 3, (0, 255, 0), -1)

    # # 获取人脸的边界框
    # bbox = face.bbox  # bbox为[x1, y1, x2, y2]
    # x1, y1, x2, y2 = map(int, bbox)  # 使用 map 和 int 转换所有坐标为整数

    # 绘制矩形框
    # cv2.rectangle(img1, (x1, y1), (x2, y2), (0, 255, 0), 2)

# 显示结果
cv2.imshow('YOLO Detection',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
# cv2.imwrite('./output2.jpg', img1)