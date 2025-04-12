# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import logging

import torch
import torch.nn.functional as F

from fastreid.config import get_cfg
from fastreid.modeling.meta_arch import META_ARCH_REGISTRY, build_model, Baseline
from fastreid.utils.checkpoint import Checkpointer

logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class Distiller(Baseline):
    def __init__(self, cfg):
        super().__init__(cfg)

        # Get teacher model config
        model_ts = []
        for i in range(len(cfg.KD.MODEL_CONFIG)):
            cfg_t = get_cfg()
            cfg_t.merge_from_file(cfg.KD.MODEL_CONFIG[i])
            cfg_t.defrost()
            cfg_t.MODEL.META_ARCHITECTURE = "Baseline"
            # Change syncBN to BN due to no DDP wrapper
            if cfg_t.MODEL.BACKBONE.NORM == "syncBN":
                cfg_t.MODEL.BACKBONE.NORM = "BN"
            if cfg_t.MODEL.HEADS.NORM == "syncBN":
                cfg_t.MODEL.HEADS.NORM = "BN"

            model_t = build_model(cfg_t)

            # No gradients for teacher model
            for param in model_t.parameters():
                param.requires_grad_(False)

            logger.info("Loading teacher model weights ...")
            Checkpointer(model_t).load(cfg.KD.MODEL_WEIGHTS[i])

            model_ts.append(model_t)

        self.ema_enabled = cfg.KD.EMA.ENABLED
        self.ema_momentum = cfg.KD.EMA.MOMENTUM
        if self.ema_enabled:      #没用到
            cfg_self = cfg.clone()
            cfg_self.defrost()
            cfg_self.MODEL.META_ARCHITECTURE = "Baseline"
            if cfg_self.MODEL.BACKBONE.NORM == "syncBN":
                cfg_self.MODEL.BACKBONE.NORM = "BN"
            if cfg_self.MODEL.HEADS.NORM == "syncBN":
                cfg_self.MODEL.HEADS.NORM = "BN"
            model_self = build_model(cfg_self)
            # No gradients for self model  冻结了教师模型的所有参数，确保在训练过程中，教师模型的参数不会被更新
            for param in model_self.parameters():
                param.requires_grad_(False)

            if cfg_self.MODEL.WEIGHTS != '':
                logger.info("Loading self distillation model weights ...")
                Checkpointer(model_self).load(cfg_self.MODEL.WEIGHTS)
            else:
                # Make sure the initial state is same
                for param_q, param_k in zip(self.parameters(), model_self.parameters()):
                    param_k.data.copy_(param_q.data)

            model_ts.insert(0, model_self)
        # Not register teacher model as `nn.Module`, this is
        # make sure teacher model weights not saved
        self.model_ts = model_ts

    @torch.no_grad()
    def _momentum_update_key_encoder(self, m=0.999):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.parameters(), self.model_ts[0].parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)

    def forward(self, batched_inputs):
# print("batched_inputs:", batched_inputs)
# batched_inputs 包含了四种主要数据：  (['images', 'targets', 'camids', 'img_paths'])
# images: 这些是模型处理的输入图像数据。它是一个四维的张量，通常包含多个图像的批次，每个图像包含多个通道（例如RGB）和空间维度（高度和宽度）。这些图像数据是从输入数据中提取并传入模型进行特征学习的。
# targets: 这些是训练时的目标标签，通常是每张图像对应的类别或身份的ID。这些标签通常用于计算损失，帮助模型学习如何映射图像到正确的目标。
# camids: 这是摄像机ID，通常在行人重识别任务中用于表示每个图像的来源摄像机，用于进一步的区分或辅助训练。此数据在一些特定任务中可能用于损失计算。
# img_paths: 这是每张图像的路径，用于标识或记录图像的来源位置。
        if self.training:
            images = self.preprocess_image(batched_inputs) #torch.Size([256, 3, 256, 128])
            #images[0][0] tensor([[-2.1179, -2.1179, -2.1179,  ...,  0.8276,  1.1529,  1.3070], 应该已经经历过标准化处理,未经过标准化的范围是[0，255]
            
            # student model forward
            s_feat = self.backbone(images) #s_feat.shape: torch.Size([256, 512, 16, 8])

            #获取每个图像对应的目标标签（即类别或身份的ID），并确保它们是有效的。若目标标签的总和小于 0，则将其置为零。
            assert "targets" in batched_inputs, "Labels are missing in training!"
            targets = batched_inputs["targets"].to(self.device)
            if targets.sum() < 0: targets.zero_()

            s_outputs = self.heads(s_feat, targets)
            
            # s_outputs 包含三个主要字段:cls_outputs,pred_class_logits,features
            #cls_outputs size: torch.Size([256, 702])
            #pred_class_logits size: torch.Size([256, 702])
            #features size: torch.Size([256, 512])
            
            #cls_outputs: 计算得到的分类输出，tensor([[-7.7383, -7.8281, -7.8125, ..., -7.4648, -7.8438, -7.8359], ...]),但是还未经过softmax处理
            #pred_class_logits: 分类的原始得分（logits），乘以一个缩放因子 self.cls_layer.s，这通常用于调整得分范围。 tensor([[ 2.5547e+00,  7.7197e-01, -1.3037e+00, ..., -4.9375e+00, -1.1749e-03, -6.7041e-01], ...])
            #features: 经过池化和瓶颈处理后的特征表示 tensor([[ 1.1944,  0.9056, -1.0466, ..., -1.2170,  1.0566, -0.5922], ...])
            
            #print("s_outputs:", s_outputs)

            t_outputs = []
            # teacher model forward
            predicted_class=[]
            with torch.no_grad():
                if self.ema_enabled:
                    self._momentum_update_key_encoder(self.ema_momentum)  # update self distill model
                for model_t in self.model_ts:
                    # resnet101是教师模型时
                    # t_feat = model_t.backbone(images)
                    # t_output = model_t.heads(t_feat, targets)
                    # t_outputs.append(t_output) #如果有多个教师模型才需要用到
                    # teacher_cls_score = t_output['pred_class_logits']  
                    # max_prob, predicted_class = torch.max(teacher_cls_score, dim=1)

                    #若教师模型是vit-16
                    t_feat = model_t.backbone(images)  
                    t_outputs.append(t_feat[0][0]) #  t_feat[0][0]是cls_score
                    max_prob, predicted_class = torch.max(t_feat[0][0], dim=1)

            correct_predictions = (predicted_class == targets).sum().item()
            print(f"正确预测的数量: {correct_predictions} / 256")

            losses = self.losses(s_outputs, t_outputs, targets)
            return losses

        # Eval mode, just conventional reid feature extraction
        else:
            return super().forward(batched_inputs)


    def losses(self, s_outputs, t_outputs, gt_labels):
        """
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        #调用父类的 losses 方法来计算学生模型的损失
        loss_dict = super().losses(s_outputs, gt_labels)
        #提取学生模型的 logits

        # resnet101时：
        # s_logits = s_outputs['pred_class_logits']
        # loss_jsdiv = 0.
        # for t_output in t_outputs:
        #     t_logits = t_output['pred_class_logits'].detach()
        #     loss_jsdiv += self.jsdiv_loss(s_logits, t_logits)

        # vit-16时：
        s_logits = s_outputs['pred_class_logits']  #[256, 702]
        s_logits = F.softmax(s_logits, dim=1)   #我自己加的
        s_logits = s_logits*1
        loss_jsdiv = 0.
        for t_output in t_outputs:   #可能有多个教师模型，但本实验中就一个
            t_logits = t_output.detach()
            t_logits = F.softmax(t_logits, dim=1)
            t_logits = t_logits*1
            loss_jsdiv += self.jsdiv_loss(s_logits, t_logits)

        loss_dict["loss_jsdiv"] = loss_jsdiv / len(t_outputs)

        return loss_dict

    @staticmethod
    def _kldiv(y_s, y_t, t):
        p_s = F.log_softmax(y_s / t, dim=1)
        p_t = F.softmax(y_t / t, dim=1)
        loss = F.kl_div(p_s, p_t, reduction="sum") * (t ** 2) / y_s.shape[0]
        return loss

    def jsdiv_loss(self, y_s, y_t, t=16):   #t是温度
        loss = (self._kldiv(y_s, y_t, t) + self._kldiv(y_t, y_s, t)) / 2
        return loss
