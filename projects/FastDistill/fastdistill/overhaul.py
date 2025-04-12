# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import logging
import math

import torch
import torch.nn.functional as F
from scipy.stats import norm
from torch import nn

from fastreid.modeling.meta_arch import META_ARCH_REGISTRY, Distiller

logger = logging.getLogger("fastreid.meta_arch.overhaul_distiller")

#计算学生网络（source）和教师网络（target）之间的蒸馏损失。
def distillation_loss(source, target, margin):
    target = torch.max(target, margin)
    loss = F.mse_loss(source, target, reduction="none")
    loss = loss * ((source > target) | (target > 0)).float()
    return loss.sum()

#将学生网络的特征和教师网络的特征对接
def build_feature_connector(t_channel, s_channel):
    C = [nn.Conv2d(s_channel, t_channel, kernel_size=1, stride=1, padding=0, bias=False),
         nn.BatchNorm2d(t_channel)]

    for m in C:#通过正态分布初始化卷积层的权重，批归一化层的权重初始化为1，偏置初始化为0
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return nn.Sequential(*C)

#从教师模型的批归一化层（bn）中计算 margin（即蒸馏损失的下限）
def get_margin_from_BN(bn):
    margin = []
    std = bn.weight.data
    mean = bn.bias.data
    for (s, m) in zip(std, mean):
        s = abs(s.item())
        m = m.item()
        if norm.cdf(-m / s) > 0.001:
            margin.append(- s * math.exp(- (m / s) ** 2 / 2) / \
                          math.sqrt(2 * math.pi) / norm.cdf(-m / s) + m)
        else:
            margin.append(-3 * s)

    return torch.tensor(margin, dtype=torch.float32, device=mean.device)


@META_ARCH_REGISTRY.register()
class DistillerOverhaul(Distiller):
    def __init__(self, cfg):
        super().__init__(cfg)

        #获取学生模型（self.backbone）的通道数。
        s_channels = self.backbone.get_channel_nums()

        for i in range(len(self.model_ts)):
            #对于每个教师模型（self.model_ts），为每一对通道创建特征连接器，并将其注册为模块。
            t_channels = self.model_ts[i].backbone.get_channel_nums()
            setattr(self, "connectors_{}".format(i), nn.ModuleList(
                [build_feature_connector(t, s) for t, s in zip(t_channels, s_channels)]))

            #计算每个教师模型中批归一化层的 margin，并将其注册为模型的缓冲区（register_buffer），这样 margin 可以参与计算但不会被优化。
            teacher_bns = self.model_ts[i].backbone.get_bn_before_relu()
            margins = [get_margin_from_BN(bn) for bn in teacher_bns]
            for j, margin in enumerate(margins):
                self.register_buffer("margin{}_{}".format(i, j + 1),
                                     margin.unsqueeze(1).unsqueeze(2).unsqueeze(0).detach())

    def forward(self, batched_inputs):
        if self.training:
            # 如果是训练模式：
            # 预处理输入图像，提取学生模型的特征。
            # 获取目标标签（targets），并计算学生模型的输出。
            # 使用教师模型计算其特征和输出（冻结梯度）。
            # 计算损失，并返回。

            images = self.preprocess_image(batched_inputs)
            # student model forward
            s_feats, s_feat = self.backbone.extract_feature(images, preReLU=True)
            assert "targets" in batched_inputs, "Labels are missing in training!"
            targets = batched_inputs["targets"].to(self.device)

            if targets.sum() < 0: targets.zero_()

            s_outputs = self.heads(s_feat, targets)

            t_feats_list = []
            t_outputs = []
            # teacher model forward
            with torch.no_grad():
                if self.ema_enabled:
                    self._momentum_update_key_encoder(self.ema_momentum)
                for model_t in self.model_ts:
                    t_feats, t_feat = model_t.backbone.extract_feature(images, preReLU=True)
                    t_output = model_t.heads(t_feat, targets)
                    t_feats_list.append(t_feats)
                    t_outputs.append(t_output)

            losses = self.losses(s_outputs, s_feats, t_outputs, t_feats_list, targets)
            return losses

        else:
            #如果是评估模式（即非训练模式），则调用父类的 forward 方法
            outputs = super(DistillerOverhaul, self).forward(batched_inputs)
            return outputs

    #计算学生模型和教师模型的输出损失
    def losses(self, s_outputs, s_feats, t_outputs, t_feats_list, gt_labels):
        """
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        loss_dict = super().losses(s_outputs, t_outputs, gt_labels)

        # Overhaul distillation loss
        feat_num = len(s_feats)
        loss_distill = 0
        for i in range(len(t_feats_list)):
            for j in range(feat_num):
                s_feats_connect = getattr(self, "connectors_{}".format(i))[j](s_feats[j])
                loss_distill += distillation_loss(s_feats_connect, t_feats_list[i][j].detach(), getattr(
                    self, "margin{}_{}".format(i, j + 1)).to(s_feats_connect.dtype)) / 2 ** (feat_num - j - 1)

        loss_dict["loss_overhaul"] = loss_distill / len(t_feats_list) / len(gt_labels) / 10000

        return loss_dict
