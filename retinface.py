import cv2
import numpy as np
import torch
from PIL import Image
from torch.backends import cudnn
from tqdm import tqdm

from cfg.config import cfg_mnet, cfg_re50
from net_facenet.facenet import Facenet
from net_retinaface.retinaface import RetinaFace
from utils.retinface.nms.py_cpu_nms import py_cpu_nms
import time

from utils.retinface.prior_box import PriorBox
from utils.retinface.box_utils import decode, decode_landm
from utils.utils import Alignment_1, letterbox_image, compare_faces, cv2ImgAddText


class Retinaface:
    def __init__(self, is_encode=False):
        super(Retinaface, self).__init__()
        self.network = "mobile0.25"
        self.facenet_backbone = "mobilenet"
        self.nms_threshold = 0.4
        self.top_k = 5000
        self.keep_top_k = 750
        self.confidence_threshold = 0.02
        self.facenet_threhold = 0.9
        self.trained_model = './model_data/mobilenet0.25_Final.pth'
        self.facenet_model_path = './model_data/facenet_mobilenet.pth'
        self.facenet_input_shape = np.array([160, 160, 3])
        if not is_encode:
            self.known_face_encodings = np.load(
                "model_data/{backbone}_face_encoding.npy".format(backbone=self.facenet_backbone))
            self.known_face_names = np.load("model_data/{backbone}_names.npy".format(backbone=self.facenet_backbone))

        if self.network == "mobile0.25":
            self.cfg = cfg_mnet
        elif self.network == "resnet50":
            self.cfg = cfg_re50

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.net = RetinaFace(cfg=self.cfg, phase = 'test')
        state_dict = torch.load(self.trained_model, map_location=self.device)
        self.net.load_state_dict(state_dict)
        # self.net = load_model(self.net, self.trained_model, False)

        self.facenet = Facenet(backbone=self.facenet_backbone, mode="predict").eval()
        state_dict = torch.load('./model_data/facenet_mobilenet.pth', map_location=self.device)
        self.facenet.load_state_dict(state_dict, strict=False)

        self.net = self.net.to(self.device)
        self.facenet = self.facenet.to(self.device)

    def detect_image(self, image):
        if image is None:
            print('Open Error! Try again!')
        else:
            img = np.float32(image)
            dets = self.deal_image(img)
            img_raw = image.copy()
            img_show = self.face_compare(dets, img_raw)
            return img_show

    def deal_image(self, img):
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)

        tic = time.time()
        self.net.eval()
        loc, conf, landms = self.net(img)  # forward pass
        print('net forward time: {:.4f}'.format(time.time() - tic))

        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale  # / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1  # / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]


        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:self.keep_top_k, :]
        landms = landms[:self.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)
        return dets

    def face_compare(self, dets, img_raw):

        # -----------------------------------------------#
        #   Facenet编码部分-开始
        # -----------------------------------------------#
        face_encodings = []
        for i, boxes_conf_landm in enumerate(dets):
            if boxes_conf_landm[4] < self.facenet_threhold:
                continue

            # ----------------------#
            #   图像截取，人脸矫正
            # ----------------------#
            boxes_conf_landm = np.maximum(boxes_conf_landm, 0)
            crop_img = np.array(img_raw)[int(boxes_conf_landm[1]):int(boxes_conf_landm[3]),
                       int(boxes_conf_landm[0]):int(boxes_conf_landm[2])]
            landmark = np.reshape(boxes_conf_landm[5:], (5, 2)) - np.array(
                [int(boxes_conf_landm[0]), int(boxes_conf_landm[1])])
            crop_img, _ = Alignment_1(crop_img, landmark)

            # ----------------------#
            #   人脸编码
            # ----------------------#
            crop_img = np.array(letterbox_image(np.uint8(crop_img),
                                                (self.facenet_input_shape[1], self.facenet_input_shape[0]))) / 255
            crop_img = np.expand_dims(crop_img.transpose(2, 0, 1), 0)
            with torch.no_grad():
                crop_img = torch.from_numpy(crop_img).type(torch.FloatTensor)
                crop_img = crop_img.to(self.device)

                # -----------------------------------------------#
                #   利用facenet_model计算长度为128特征向量
                # -----------------------------------------------#
                face_encoding = self.facenet(crop_img)[0].cpu().numpy()
                face_encodings.append(face_encoding)

        # -----------------------------------------------#
        #   人脸特征比对-开始
        # -----------------------------------------------#
        face_names = []

        for i, face_encoding in enumerate(face_encodings):
            # -----------------------------------------------------#
            #   取出一张脸并与数据库中所有的人脸进行对比，计算得分
            # -----------------------------------------------------#
            matches, face_distances = compare_faces(self.known_face_encodings, face_encoding,
                                                    tolerance=self.facenet_threhold)
            name = "Unknown"
            # -----------------------------------------------------#
            #   取出这个最近人脸的评分
            #   取出当前输入进来的人脸，最接近的已知人脸的序号
            # -----------------------------------------------------#
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
                print("name {}, i {}".format(name, i))

            face_names.append(name)

        # -----------------------------------------------#
        #   人脸特征比对-结束
        # -----------------------------------------------#

        # 绘制人脸输出图像
        for i, b in enumerate(dets):

            if b[4] < self.facenet_threhold:
                continue

            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            # ---------------------------------------------------#
            #   b[0]-b[3]为人脸框的坐标，b[4]为得分
            # ---------------------------------------------------#
            cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(img_raw, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # ---------------------------------------------------#
            #   b[5]-b[14]为人脸关键点的坐标
            # ---------------------------------------------------#
            cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)

            name = face_names[i]
            print("i {} name {}".format(i, name))
            # cv2.putText(img_raw, name, (b[0] , b[3] - 15), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)
            # --------------------------------------------------------------#
            #   cv2不能写中文，加上这段可以，但是检测速度会有一定的下降。
            #   如果不是必须，可以换成cv2只显示英文。
            # --------------------------------------------------------------#
            old_image = cv2ImgAddText(img_raw, name, b[0] + 5, b[3] - 25)
            return old_image

    def encode_face_dataset(self, image_paths, names):
        face_encodings = []
        torch.set_grad_enabled(False)

        for index, path in enumerate(tqdm(image_paths)):
            # ---------------------------------------------------#
            #   打开人脸图片
            # ---------------------------------------------------#
            image = np.array(Image.open(path), np.float32)
            # ---------------------------------------------------#
            #   对输入图像进行一个备份
            # ---------------------------------------------------#
            old_image = image.copy()
            img = np.float32(image)

            dets = self.deal_image(img)

            # ---------------------------------------------------#
            #   选取最大的人脸框。
            # ---------------------------------------------------#
            best_face_location = None
            biggest_area = 0
            for i, result in enumerate(dets):
                if result[4] < self.facenet_threhold:
                    continue

                left, top, right, bottom = result[0:4]

                w = right - left
                h = bottom - top
                if w * h > biggest_area and result[4] >= self.facenet_threhold:
                    biggest_area = w * h
                    best_face_location = result

            if best_face_location is None:
                print("best_face_location nonoe", i)
                continue

            # ---------------------------------------------------#
            #   截取图像
            # ---------------------------------------------------#
            print("best_face_location ", best_face_location)
            crop_img = old_image[int(best_face_location[1]):int(best_face_location[3]),
                       int(best_face_location[0]):int(best_face_location[2])]
            landmark = np.reshape(best_face_location[5:], (5, 2)) - np.array(
                [int(best_face_location[0]), int(best_face_location[1])])
            crop_img, _ = Alignment_1(crop_img, landmark)

            crop_img = np.array(
                letterbox_image(np.uint8(crop_img), (self.facenet_input_shape[1], self.facenet_input_shape[0]))) / 255
            crop_img = crop_img.transpose(2, 0, 1)
            crop_img = np.expand_dims(crop_img, 0)
            # ---------------------------------------------------#
            #   利用图像算取长度为128的特征向量
            # ---------------------------------------------------#
            with torch.no_grad():
                crop_img = torch.from_numpy(crop_img).type(torch.FloatTensor)
                crop_img = crop_img.to(self.device)

                face_encoding = self.facenet(crop_img)[0].cpu().numpy()
                face_encodings.append(face_encoding)

        np.save("model_data/{backbone}_face_encoding.npy".format(backbone=self.facenet_backbone), face_encodings)
        np.save("model_data/{backbone}_names.npy".format(backbone=self.facenet_backbone), names)

    def test(self):
        crop_img = np.array(
            letterbox_image(np.uint8(cv2.imread('zhangxueyou0.jpg')), (self.facenet_input_shape[1], self.facenet_input_shape[0]))) / 255
        crop_img = crop_img.transpose(2, 0, 1)
        crop_img = np.expand_dims(crop_img, 0)
        # ---------------------------------------------------#
        #   利用图像算取长度为128的特征向量
        # ---------------------------------------------------#
        with torch.no_grad():
            crop_img = torch.from_numpy(crop_img).type(torch.FloatTensor)
            crop_img = crop_img.to(self.device)

            face_encoding = self.facenet(crop_img)[0].cpu().numpy()

        matches, face_distances = compare_faces(self.known_face_encodings, face_encoding,
                                                tolerance=self.facenet_threhold)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = self.known_face_names[best_match_index]
