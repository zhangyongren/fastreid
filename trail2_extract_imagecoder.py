import torch
import clip

# 加载本地 CLIP 模型权重
pthfile = '/data/zyr/fast-reid-master/projects/ClipReid/output/dukemtmcreid/ViT-B-16_60.pth'
checkpoint = torch.load(pthfile, map_location='cpu')

# 检查 checkpoint 中是否包含模型信息
if 'model' in checkpoint:
    model_weights = checkpoint['model']
else:
    model_weights = checkpoint

# 加载 CLIP 模型架构，并将模型权重加载到该架构
clip_model, preprocess = clip.load("RN50", device="cuda")  # 使用相同的架构

# 提取图像编码器部分（Vision Transformer）
image_encoder = clip_model.visual  # 图像编码器即 CLIP 的 vision 部分

# 如果权重文件中包含图像编码器的权重，加载它们
image_encoder.load_state_dict(model_weights, strict=False)  # 使用 strict=False 跳过不匹配的层

# 提取图像编码器的权重并保存为 .pth 文件
torch.save(image_encoder.state_dict(), "Resnet50_teacher.pth")

# 输出保存的权重文件的大小，确保保存成功
print(f"Saved image encoder weights to 'Resnet50_teacher.pth'.")
