import json

from flask import Flask, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import cv2
import os
import redis
import uuid
from service.faceserver import faceModel
app = Flask(__name__)
# 创建 Redis 连接
redis_client = redis.Redis(host='localhost', port=6379,password='jkkj@123', decode_responses=True)

def configure_routes(app):

    # 测试 030_模型训练(列出全部任务）
    @app.route('/aps/l1', methods=['POST'])
    def l1():
        name = request.form.get('name')
        image = request.files['file']
        print(image.filename)
        # 将上传文件的二进制流读取为 NumPy 数组
        file_bytes = image.read()  # 读取文件流
        image = np.asarray(bytearray(file_bytes), dtype=np.uint8)
        img_cv2 = cv2.imdecode(image, cv2.IMREAD_COLOR)  # 解码为 BGR 图像
        faces1 = faceModel.get(img_cv2)
        if(len(faces1) > 0):
            embedding = faces1[0]['embedding']
            userInfo = {"name":name,"tzval":','.join(map(str, embedding))}
            uid = uuid.uuid4()
            redis_client.set("rl"+str(uid),json.dumps(userInfo))
            print(userInfo)
            return "注册成功"
        else:
            print("无法检测到人脸，请检查图片质量或更换图片")
            return "无法检测到人脸，请检查图片质量或更换图片"

    @app.route('/aps/l2', methods=['POST'])
    def l2():
        image = request.files['file']
        # 将上传文件的二进制流读取为 NumPy 数组
        file_bytes = image.read()  # 读取文件流
        image = np.asarray(bytearray(file_bytes), dtype=np.uint8)
        img_cv2 = cv2.imdecode(image, cv2.IMREAD_COLOR)  # 解码为 BGR 图像
        faces1 = faceModel.get(img_cv2)
        if (len(faces1) > 0):
            embedding = faces1[0]['embedding']
            # 自定义阈值
            threshold = 0.3 # 比默认值 0.6 更严格
            rls = redis_client.keys("rl*")
            for key in rls:
                rld = redis_client.get(key)
                rld = json.loads(rld)
                embedding2 = np.array([float(value) for value in rld["tzval"].split(",")])
                # 计算余弦相似度
                cosine_similarity = np.dot(embedding,embedding2) / (
                        np.linalg.norm(embedding) * np.linalg.norm(embedding2)
                )
                print(cosine_similarity)
                # 根据阈值判断是否为同一个人
                if cosine_similarity > 0.5:
                    print("两张图片是同一个人！")
                    return rld["name"]
                if cosine_similarity > 0.3:
                    return "可能是"+ rld["name"]

        else:
            return "无法检测到人脸，请检查图片质量或更换图片"




        return "没认出来"

    @app.route('/aps/l3', methods=['GET'])
    def l3():
        rls = redis_client.keys("rl*")
        for key in rls:
            redis_client.delete(key)

        return "清空完成!"