from setuptools import setup, find_packages

setup(
    name='face recognition',
    version='0.1.0',
    description='face recognition',
    url='https://github.com/hduwyz/face_recognition',
    author='face_recognition',
    author_email='wangyz_hdu@163.com',
    license='MIT',
    keywords='python toolkit utils',
    packages=find_packages(),  # 包含所有的py文件
    include_package_data=True,  # 将数据文件也打包
    zip_safe=True,
    install_requires=['dlib==19.19.0', 'numpy==1.24.3', 'opencv_python==4.5.5.62'],
    python_requires='>=3.8'
)
