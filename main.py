import utils.face_recognition_util as fc

if __name__ == '__main__':
    image_path1 = './face/B.jpg'
    image_path2 = './face/A.png'
    v1, shape1, rect1 = fc.face_descriptor(image_path1)
    v2, shape2, rect2 = fc.face_descriptor(image_path2)
    fc.face_match(v1, v2)
    # 绘制面部轮廓
    fc.show_img(image_path1, shape1, rect1)
