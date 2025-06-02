import logging
import math
import torch
import torch.nn as nn
from .make_model import make_model
# from fastreid.layers.batch_norm import IBN
# from fastreid.layers.ca_layer import CA
import projects.ClipReid.model.clip.clip as clip

from collections import OrderedDict
from fastreid.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
#from .build import BACKBONE_REGISTRY
from fastreid.modeling.backbones import BACKBONE_REGISTRY
logger = logging.getLogger(__name__)

# class LayerNorm(nn.LayerNorm):
#     """Subclass torch's LayerNorm to handle fp16."""

#     #输入是16位精度的张量，中间计算用32位精度，最后输出在用回16位精度
#     def forward(self, x: torch.Tensor):
#         orig_type = x.dtype
#         ret = super().forward(x.type(torch.float32))
#         return ret.type(orig_type)

# #计算更快的激活函数
# class QuickGELU(nn.Module):
#     def forward(self, x: torch.Tensor):
#         return x * torch.sigmoid(1.702 * x)


# class ResidualAttentionBlock(nn.Module):
#     def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
#         super().__init__()

#         self.attn = nn.MultiheadAttention(d_model, n_head)
#         self.ln_1 = LayerNorm(d_model)
#         self.mlp = nn.Sequential(OrderedDict([
#             ("c_fc", nn.Linear(d_model, d_model * 4)),
#             ("gelu", QuickGELU()),
#             ("c_proj", nn.Linear(d_model * 4, d_model))
#         ]))
#         self.ln_2 = LayerNorm(d_model)
#         self.attn_mask = attn_mask

#     def attention(self, x: torch.Tensor):
#         self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
#         return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

#     def forward(self, x: torch.Tensor):
#         x = x + self.attention(self.ln_1(x))
#         x = x + self.mlp(self.ln_2(x))
#         return x

# #多个 ResidualAttentionBlock 层堆叠起来，形成一个完整的 Transformer 结构
# class Transformer(nn.Module):
#     def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
#         super().__init__()
#         self.width = width
#         self.layers = layers
#         self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

#     def forward(self, x: torch.Tensor):
#         return self.resblocks(x)

# class VisionTransformer(nn.Module):
#     def __init__(self, h_resolution: int, w_resolution: int, patch_size: int, stride_size: int, width: int, layers: int,
#                  heads: int, output_dim: int):
#         super().__init__()
#         self.h_resolution = h_resolution
#         self.w_resolution = w_resolution
#         self.output_dim = output_dim

#         #将输入的图像（3个通道，RGB）切割成多个 patch
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=stride_size,
#                                bias=False)

#         scale = width ** -0.5
#         #class_embedding 是一个可学习的参数，表示分类标记的嵌入。它被初始化为一个具有 width 维度的随机张量
#         self.class_embedding = nn.Parameter(scale * torch.randn(width))
#         #表示每个图像中的位置嵌入数量，加 1 是因为包括了分类标记
#         self.positional_embedding = nn.Parameter(scale * torch.randn(h_resolution * w_resolution + 1, width))
#         #一个层归一化层，应用于输入数据，使其在通过 Transformer 之前具有标准化的特征
#         self.ln_pre = LayerNorm(width)

#         self.transformer = Transformer(width, layers, heads)

#         #后归一化层，在 Transformer 输出之后应用
#         self.ln_post = LayerNorm(width)
#         #proj 是一个可学习的线性变换，用于将 Transformer 的输出映射到最终的输出维度 output_dim
#         self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

#     def forward(self, x: torch.Tensor, cv_emb=None):
#         x = self.conv1(x)  #将输入的图像 x 切割成多个 patch，并将每个 patch 映射到 width 维空间   shape = [*, width, grid, grid]
#         x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
#         x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        
#         x = torch.cat(
#             [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
#              x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        
#         if cv_emb != None: #如果提供了 cv_emb，则将其添加到第一个位置
#             x[:, 0] = x[:, 0] + cv_emb
        
#         #将位置嵌入添加到每个 patch 上，以便 Transformer 可以利用空间信息
#         x = x + self.positional_embedding.to(x.dtype)
#         #对输入数据进行归一化处理
#         x = self.ln_pre(x)

#         #交换维度，从 [batch_size, grid^2 + 1, width] 转换为 [grid^2 + 1, batch_size, width]，以符合 Transformer 的输入格式
#         x = x.permute(1, 0, 2)  # NLD -> LND

#         #由于 Transformer 被分成多个块（resblocks），首先通过前 11 层，之后再通过第 12 层
#         x11 = self.transformer.resblocks[:11](x)
#         x12 = self.transformer.resblocks[11](x11)

#         #交换维度，将其恢复为 [batch_size, grid^2 + 1, width]
#         x11 = x11.permute(1, 0, 2)  # LND -> NLD
#         x12 = x12.permute(1, 0, 2)  # LND -> NLD

#         #对 Transformer 输出进行归一化
#         x12 = self.ln_post(x12)

#         if self.proj is not None: #如果存在投影层（proj），则对 Transformer 的输出进行线性变换，得到最终的嵌入
#             xproj = x12 @ self.proj

#         img_feature = x12[:, 0] #Transformer 输出的分类 token（即 [batch_size, width]），代表整个图像的表示
#         img_feature_proj = xproj[:, 0] #经过线性投影后的分类 token

#         return [img_feature, img_feature_proj]


@BACKBONE_REGISTRY.register()
def build_clip_vit_backbone_distill(cfg):

    # fmt: off
    pretrain      = cfg.MODEL.BACKBONE.PRETRAIN
    pretrain_path = cfg.MODEL.BACKBONE.PRETRAIN_PATH
    # last_stride   = cfg.MODEL.BACKBONE.LAST_STRIDE
    # bn_norm       = cfg.MODEL.BACKBONE.NORM
    # with_ibn      = cfg.MODEL.BACKBONE.WITH_IBN
    # with_se       = cfg.MODEL.BACKBONE.WITH_SE
    # with_nl       = cfg.MODEL.BACKBONE.WITH_NL
    depth = cfg.MODEL.BACKBONE.DEPTH
    backbone_name = {
        '50x': "RN50",
        '101x': "RN101",
        'vit_32': "ViT-B-32",
        'vit_16': "ViT-B-16"
    }[depth]
    # fmt: on
    # url = clip._MODELS[backbone_name]
    # model_path = clip._download(url)

    # try:
    #     # loading JIT archive
    #     model = torch.jit.load(model_path, map_location="cpu").eval()
    #     state_dict = model.state_dict()

    # except RuntimeError:
    #     state_dict = torch.load(model_path, map_location="cpu")

    # #print("Available keys in state_dict:", state_dict.keys())  
    # print("info from url")
    # for name, param in state_dict.items():
    #     print(f"{name}: {param.shape}")
    # #获得VisionTransformer构建相关的超参数
    # h_resolution = int((cfg.INPUT.SIZE_TRAIN[0] - 16) // 16 + 1)
    # w_resolution = int((cfg.INPUT.SIZE_TRAIN[1] - 16) // 16 + 1)
    # counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in
    #                 [1, 2, 3, 4]]

    # #vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
    # vision_width = state_dict["visual.conv1.weight"].shape[0]
    # #output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
    # output_width = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    # #assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
    # assert output_width ** 2 + 1 == state_dict["visual.positional_embedding"].shape[0]
    # image_resolution = output_width * 32

    # vision_heads = vision_width * 32 // 64
    # embed_dim = state_dict["text_projection"].shape[1]
    # vision_heads = vision_width // 64
    # vision_width = state_dict["visual.conv1.weight"].shape[0]
    # vision_layers = len(
    #     [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
    # vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    # grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    # image_resolution = vision_patch_size * grid_size
    # vision_stride_size=16

    # print("cccccccccccccccccc")
    # #构建 Vision Transformer 模型
    # model = VisionTransformer(
    #     h_resolution=16,  # Image resolution should be same as the pretrained model
    #     w_resolution=16,
    #     patch_size=16,  # Make sure this matches with pretrained model
    #     stride_size=16,  # Make sure this matches with pretrained model
    #     width=768,  # Ensure this matches the pretrained model's width
    #     layers=12,  # Ensure this matches the pretrained model's layers
    #     heads=12,  # Ensure this matches the pretrained model's heads
    #     output_dim=512  # Ensure this matches the pretrained model's embedding dimension
    # )
    # model =  VisionTransformer(
    #     h_resolution=h_resolution,
    #     w_resolution=w_resolution,
    #     patch_size=vision_patch_size,
    #     stride_size=vision_stride_size,
    #     width=vision_width,
    #     layers=vision_layers,
    #     heads=vision_heads,
    #     output_dim=embed_dim
    # )
    #print("aaaaaaaaaaaaaa")

    
    model = make_model(cfg,702, 8, 1)
    print("aaaaaaaaaaaaaa")
    for name, param in model.named_parameters():
        print(name, param.size())
    #打印model每一层的参数量
    for name, param in model.named_parameters():
        print(name, param.numel())
    state_dict = model.state_dict()


    # # 3. 加载预训练模型权重（假设是 'ViT-B-16_60.pth' 文件）
    # pretrain_path = '/data/zyr/fast-reid-master/projects/FastDistill/logs/dukemtmc/vit_16/ViT-B-16_60.pth'  # 这里是预训练模型的路径
    # state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))
    
    incompatible = model.load_state_dict(state_dict, strict=False)
    
    if incompatible.missing_keys:
        logger.info(
            get_missing_parameters_message(incompatible.missing_keys)
        )
    if incompatible.unexpected_keys:
        logger.info(
            get_unexpected_parameters_message(incompatible.unexpected_keys)
        )
    print("bbbbbbbbbbb")
    return model


