# 介绍
本代码主要是做人脸特征提取，并将提取到的特征显示出来
采用的方法是：通过深度学习人脸模型检测到人脸的位置，然后通过dlib库提取人脸特征，最后得到人脸特征向量，
将这些向量入库后，后面通过比对这些人脸特征向量从而达到人脸识别的目的
# 使用方法
## 步骤1：先安装需要的依赖包，执行如下命令 
```python
pip3 install -r requirements.txt
```
## 步骤2：引用模块
```python
import utils.face_recognition_util as fc
```
## 步骤3：
```python
image_path1 = './face/B.jpg'
image_path2 = './face/A.png'
# 提取面部特征向量
v1, shape1, rect1 = fc.face_descriptor(image_path1)
v2, shape2, rect2 = fc.face_descriptor(image_path2)
# 人脸特征向量匹配
fc.face_match(v1, v2)
# 绘制面部轮廓
fc.show_img(image_path1, shape1, rect1)
```
## 写在最后
感谢使用，谢谢