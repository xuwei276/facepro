# from flask import Flask
# from routes import route
#
# app = Flask(__name__)
# route.configure_routes(app)
# if __name__ == '__main__':
#     app.run(debug=True,host='0.0.0.0')


# from ultralytics import YOLO
# import cv2
#
# # 加载 YOLO 模型
# model = YOLO('./runs/detect/train3/weights/last.pt')  # 使用 YOLO 的预训练模型
#
# # 开始训练
# model.train(data='./data.yaml', epochs=50, imgsz=640)