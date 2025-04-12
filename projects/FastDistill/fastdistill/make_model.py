import torch
import torch.nn as nn
import numpy as np
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

def weights_init_kaiming(m):#使用 Kaiming 初始化方法初始化卷积层和全连接层的权重
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):#初始化分类器的权重
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(build_transformer, self).__init__()
        self.model_name = cfg.MODEL.BACKBONE.DEPTH
        self.cos_layer = False
        self.neck = 'bnneck'
        self.neck_feat = 'before'
        if self.model_name == 'vit_16':
            self.in_planes = 768 #特征维度
            self.in_planes_proj = 512
        self.num_classes=num_classes
        self.camera_num=camera_num
        self.view_num=view_num
        #两个全连接层，分别用于计算来自不同部分的图像特征的类别分数
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)

        #瓶颈层：为每个图像特征和投影特征应用 Batch Normalization 操作
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        #计算输出图像的分辨率，基于输入图像尺寸、步幅和卷积核大小。
        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0]-16)//cfg.MODEL.BACKBONE.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1]-16)//cfg.MODEL.BACKBONE.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.BACKBONE.STRIDE_SIZE[0]
        
        #加载 CLIP 模型：调用 load_clip_to_cpu 函数加载 CLIP 模型，转移到 GPU 上。
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size,self.model_name)
        clip_model.to("cuda")

        #从 CLIP 模型中提取出 visual 模块，作为图像的特征提取器
        self.image_encoder = clip_model.visual
        # print("image_encoder:")
        # for name, param in self.image_encoder.state_dict().items():
        #     param_size = param.numel() 
        #     print(f"{name}: {param.shape}, Total Parameters: {param_size}")
        
        #自定义视角嵌入（SIE）
        #SIE_CAMERA 或 SIE_VIEW，则根据相机数量和视角数量初始化一个参数矩阵 cv_embed，这个矩阵会在训练中学习
        #trunc_normal_ 用于初始化权重
        # if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
        #     self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
        #     trunc_normal_(self.cv_embed, std=.02)
        #     print('camera number is : {}'.format(camera_num))
        # elif cfg.MODEL.SIE_CAMERA:
        #     self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
        #     trunc_normal_(self.cv_embed, std=.02)
        #     print('camera number is : {}'.format(camera_num))
        # elif cfg.MODEL.SIE_VIEW:
        #     self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
        #     trunc_normal_(self.cv_embed, std=.02)
        #     print('camera number is : {}'.format(view_num))

        self.PIXEL_MEAN = torch.tensor([123.675, 116.28, 103.53])  # 每个通道的均值
        self.PIXEL_STD = torch.tensor([58.395, 57.12, 57.375])     # 每个通道的标准差

        self.Teacher_PIXEL_MEAN = torch.tensor([0.5, 0.5, 0.5])
        self.Teacher_PIXEL_STD = torch.tensor([0.5, 0.5, 0.5])

    def denormalize(self, images, mean, std):  # 反标准化操作
        device = images.device  # 获取 images 的设备信息
        mean = mean.to(device)  # 将 mean 移动到相同设备
        std = std.to(device)    # 将 std 移动到相同设备
        return images * std.view(1, 3, 1, 1) + mean.view(1, 3, 1, 1)

    def normalize(self, images, mean, std):  # 标准化操作
        device = images.device  # 获取 images 的设备信息
        mean = mean.to(device)  # 将 mean 移动到相同设备
        std = std.to(device)    # 将 std 移动到相同设备
        return (images - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)

    def forward(self, x, label=None, cam_label= None, view_label=None):
        
        x = self.denormalize(x, self.PIXEL_MEAN, self.PIXEL_STD)
        x = torch.clamp(x, 0, 255)
        x = x / 255.0
        x = self.normalize(x, self.Teacher_PIXEL_MEAN, self.Teacher_PIXEL_STD)
        #print("x[0][0]:",x[0][0])
        if self.model_name == 'vit_16':
            #对于 ViT-B-16，同样从 CLIP 编码器提取图像特征。还根据相机和视角标签进行嵌入
            if cam_label != None and view_label!=None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
            elif cam_label != None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label]
            elif view_label!=None:
                cv_embed = self.sie_coe * self.cv_embed[view_label]
            else:
                cv_embed = None
            #img_feature 是经过基础的图像特征提取网络（如 ResNet 或 Vision Transformer）得到的原始特征，表示的是图像的高层次语义信息。
            #img_feature_proj 是经过额外投影处理后的特征，通常用于分类或其他任务，其维度通常较小，并且与目标任务（如类别区分）更为相关
            #由image_encoder这个外部模块来完成特征的提取
            image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed) #B,512  B,128,512
            img_feature_last = image_features_last[:,0]
            img_feature = image_features[:,0]
            img_feature_proj = image_features_proj[:,0]

        #通过瓶颈层：将图像特征和投影特征传入 BatchNorm 层进行标准化
        feat = self.bottleneck(img_feature) 
        feat_proj = self.bottleneck_proj(img_feature_proj) 

        if self.training:
            cls_score = self.classifier(feat)
            cls_score_proj = self.classifier_proj(feat_proj)
            # print("cls_score_proj", cls_score_proj)
            
            
            return [cls_score, cls_score_proj], [img_feature_last, img_feature, img_feature_proj]
            #cls_score.shape: torch.Size([256, 702])
            #cls_score_proj.shape: torch.Size([256, 702])
            #img_feature_last.shape: torch.Size([256, 768])
            #img_feature.shape: torch.Size([256, 768])
            #img_feature_proj.shape: torch.Size([256, 512])
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return torch.cat([feat, feat_proj], dim=1)
            else:
                return torch.cat([img_feature, img_feature_proj], dim=1)
        # return torch.cat([img_feature, img_feature_proj], dim=1)   
        #img_feature.shape (256,768)
        #img_feature_proj.shape (256,512)   拼接后是（256,1280）

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            # if i.replace('module.', '') not in self.state_dict():
            #     print(f"Skipping loading parameter {i}")
            #     continue
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


def make_model(cfg, num_class, camera_num, view_num):
    model = build_transformer(num_class, camera_num, view_num, cfg)
    return model


from .clip import clip
def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size,model_name):
    backbone_name = {
        '50x': "RN50",
        '101x': "RN101",
        'vit_32': "ViT-B-32",
        'vit_16': "ViT-B-16"
    }[model_name]
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model