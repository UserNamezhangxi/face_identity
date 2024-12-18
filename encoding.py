import os

from retinface import Retinaface

'''
在更换facenet网络后一定要重新进行人脸编码，运行encoding.py。
'''
retinaface = Retinaface(is_encode=True)

list_dir = os.listdir("face_dataset")
image_paths = []
names = []
for name in list_dir:
    print("dir ", name)
    print("name ", name.split("_")[0])
    image_paths.append("face_dataset/"+name)
    names.append(name.split("_")[0])

retinaface.encode_face_dataset(image_paths, names)