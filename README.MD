1、采集人脸信息

放在face_dataset 目录下，按照 name_number.jpg 来进行命名，采集完后进行特征提取和保存 运行 encoding.py

2、运行detect.py 识别人脸（demo）

3、人脸视频检测 detect_video

encoding：整体采用 retinaface 识别人脸在将识别的人脸裁剪后使用facenet 提取人脸特征向量并保存，在使用 detect_video 对比的时候 拿到一帧图之后也进行 retinaface 识别，并截取人脸 进行特征提取，提取到的特征和 之前人脸特征库生成的进行 欧氏距离比较，距离最小的就是目标人脸

人脸参考：retinaface   [https://github.com/biubug6/Pytorch_Retinaface.git](https://github.com/biubug6/Pytorch_Retinaface.git)

人脸对比参考  [https://github.com/bubbliiiing/facenet-retinaface-pytorch.git]( https://github.com/bubbliiiing/facenet-retinaface-pytorch.git)

