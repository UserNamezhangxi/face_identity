import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torch.nn import functional as F

from net_facenet.mobilenet import MobileNetV1


class mobilenet(nn.Module):
    def __init__(self, pretrained):
        super(mobilenet, self).__init__()
        self.model = MobileNetV1()
        if pretrained:
            state_dict = load_state_dict_from_url("https://github.com/bubbliiiing/facenet-pytorch/releases/download/v1.0/backbone_weights_of_mobilenetv1.pth", model_dir="model_data",
                                                progress=True)
            self.model.load_state_dict(state_dict)

        del self.model.fc
        del self.model.avg

    def forward(self, x):
        x = self.model.stage1(x)
        x = self.model.stage2(x)
        x = self.model.stage3(x)
        return x

# class inception_resnet(nn.Module):
#     def __init__(self, pretrained):
#         super(inception_resnet, self).__init__()
#         self.model = InceptionResnetV1()
#         if pretrained:
#             state_dict = load_state_dict_from_url("https://github.com/bubbliiiing/facenet-pytorch/releases/download/v1.0/backbone_weights_of_inception_resnetv1.pth", model_dir="model_data",
#                                                 progress=True)
#             self.model.load_state_dict(state_dict)
#
#     def forward(self, x):
#         x = self.model.conv2d_1a(x)
#         x = self.model.conv2d_2a(x)
#         x = self.model.conv2d_2b(x)
#         x = self.model.maxpool_3a(x)
#         x = self.model.conv2d_3b(x)
#         x = self.model.conv2d_4a(x)
#         x = self.model.conv2d_4b(x)
#         x = self.model.repeat_1(x)
#         x = self.model.mixed_6a(x)
#         x = self.model.repeat_2(x)
#         x = self.model.mixed_7a(x)
#         x = self.model.repeat_3(x)
#         x = self.model.block8(x)
#         return x
        
class Facenet(nn.Module):
    def __init__(self, backbone="mobilenet", dropout_keep_prob=0.5, embedding_size=128, num_classes=None, mode="train", pretrained=False):
        super(Facenet, self).__init__()
        if backbone == "mobilenet":
            self.backbone = mobilenet(pretrained)
            flat_shape = 1024
        elif backbone == "inception_resnetv1":
            self.backbone = inception_resnet(pretrained)
            flat_shape = 1792
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, inception_resnetv1.'.format(backbone))
        self.avg        = nn.AdaptiveAvgPool2d((1,1))
        self.Dropout    = nn.Dropout(1 - dropout_keep_prob)
        self.Bottleneck = nn.Linear(flat_shape, embedding_size, bias=False)
        self.last_bn    = nn.BatchNorm1d(embedding_size, eps=0.001, momentum=0.1, affine=True)
        if mode == "train":
            self.classifier = nn.Linear(embedding_size, num_classes)

    def forward(self, x, mode = "predict"):
        if mode == 'predict':
            x = self.backbone(x)  # (2,3,160,160) ->  (2,1024,5, 5)
            x = self.avg(x)       # (2,1024,5, 5) ->  (2,1024,1, 1)
            x = x.view(x.size(0), -1)  # (2,1024,1, 1) -> (2,1024)
            x = self.Dropout(x)        # (2,1024)   -> (2,1024)
            x = self.Bottleneck(x)     # (2,1024) -> (2,128)
            x = self.last_bn(x)        # (2,128) -> (2,128)
            x = F.normalize(x, p=2, dim=1)  # (2,128) ->    (2,128)
            return x
        x = self.backbone(x)  # torch.Size([96, 3, 160, 160]) - > (96,1024,5, 5)
        x = self.avg(x)       # (96,1024,5, 5) ->  (96,1024, 1, 1)
        x = x.view(x.size(0), -1)  # (96,1024, 1, 1)-> (96,1024)
        x = self.Dropout(x)        # (96,1024) -> (96,1024)
        x = self.Bottleneck(x)     # (96,1024) - (96,128)
        before_normalize = self.last_bn(x)  # (96,128) - >  (96,128)
        
        x = F.normalize(before_normalize, p=2, dim=1)  # (96,128) L2 标准化
        cls = self.classifier(before_normalize)        # (96, 10575)
        return x, cls

    def forward_feature(self, x):
        x = self.backbone(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.Dropout(x)
        x = self.Bottleneck(x)
        before_normalize = self.last_bn(x)
        x = F.normalize(before_normalize, p=2, dim=1)
        return before_normalize, x

    def forward_classifier(self, x):
        x = self.classifier(x)
        return x
