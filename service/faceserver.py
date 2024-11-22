from insightface.app import FaceAnalysis
import cv2
import numpy as np

# 初始化 FaceAnalysis
faceModel= FaceAnalysis(allowed_modules=['detection', 'recognition', 'genderage'])  # 启用检测和识别功能
faceModel.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 表示使用 GPU，det_size 设置检测分辨率

print("初始化模型完成!")


