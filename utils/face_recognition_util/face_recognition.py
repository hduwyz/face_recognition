# 本代码主要是做人脸特征提取，并将提取到的特征显示出来
# 采用的方法是：通过深度学习人脸模型检测到人脸的位置，然后通过dlib库提取人脸特征，最后得到人脸特征向量，
# 将这些向量入库后，后面通过比对这些人脸特征向量从而达到人脸识别的目的
import cv2
import dlib
import numpy as np
import os


def face_descriptor(image_path):
    """
    人脸特征提取
    :param image_path: 人脸图片路径
    :return:
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 人脸关键点检测器
    predictor_path = os.path.join(current_dir, "model/shape_predictor_68_face_landmarks.dat")
    # 人脸识别模型、提取特征值
    face_rec_model_path = os.path.join(current_dir, "model/dlib_face_recognition_resnet_model_v1.dat")
    prototxt_path = os.path.join(current_dir, 'model/deploy.prototxt')
    model_path = os.path.join(current_dir, 'model/res10_300x300_ssd_iter_140000_fp16.caffemodel')

    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    sp = dlib.shape_predictor(predictor_path)  # 关键点检测
    facerec = dlib.face_recognition_model_v1(face_rec_model_path)  # 编码

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    startX, startY, endX, endY = 0, 0, 0, 0
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            break
    rect = dlib.rectangle(startX, startY, endX, endY)

    shape = sp(image, rect)
    # 提取特征
    face_descriptor = facerec.compute_face_descriptor(image, shape)  # 获取到128位的编码
    v = np.array(face_descriptor)
    return v, shape, rect


def face_match(face1, face2):
    """
    显示人脸特征向量匹配
    :param face1: 人脸1特征向量
    :param face2: 人脸2特征向量
    :return:
    """
    distance = np.linalg.norm(face1 - face2)
    print(distance)
    if distance > 0.4:
        print("人员不匹配")
        return False
    else:
        print("匹配成功")
        return True


def show_img(image_path, shape, rect):
    """
    显示人脸照片，并绘制人脸特征
    :param image_path: 人脸照片路径
    :param shape: 人脸特征
    :param rect: 人脸矩形
    :return:
    """
    # 生成 Dlib 的图像窗口
    img = cv2.imread(image_path)
    win = dlib.image_window()
    win.set_image(img)
    # 绘制面部轮廓
    win.add_overlay(shape)
    # 绘制矩阵轮廓
    win.add_overlay(rect)
    dlib.hit_enter_to_continue()


image_path1 = './face/B.jpg'
image_path2 = './face/A.png'

v1, shape1, rect1 = face_descriptor(image_path1)
v2, shape2, rect2 = face_descriptor(image_path2)
face_match(v1, v2)
# 绘制面部轮廓
show_img(image_path1, shape1, rect1)
