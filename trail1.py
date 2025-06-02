# # ################################################################  resnet模型
# import torch

# # 加载 .pth 文件
# pthfile = '/data/zyr/fast-reid-master/vit-16_teacher.pth'  # 修改为你的 .pth 文件路径
# checkpoint = torch.load(pthfile, map_location='cpu')

# # 提取 'model' 中的权重
# model_weights = checkpoint['model']

# # 计算并打印总参数数量
# total_params = sum(p.numel() for p in model_weights.values() if isinstance(p, torch.Tensor))
# print(f'Total parameters: {total_params}')

# # 查看每一层的参数数量
# for name, param in model_weights.items():
#     if isinstance(param, torch.Tensor):
#         print(f"{name}: {param.numel()} parameters")


##############################################################  clip模型

import torch
# 加载 .pth 文件
#pthfile = '/data/zyr/fast-reid-master/projects/ClipReid/output/dukemtmcreid/ViT-B-16_60.pth'
#pthfile = '/data/zyr/fast-reid-master/projects/ClipReid/output/sports_cnn/RN50_120.pth'
#pthfile = '/data/zyr/fast-reid-master/ViT-B-16_stage1_120.pth'
#pthfile = '/data/zyr/fast-reid-master/projects/FastDistill/logs/sports/r34/model_best.pth'
pthfile = '/data/zyr/fast-reid-master/projects/ClipReid/output/sports_clipreid/ViT-B-16_60.pt'
checkpoint = torch.load(pthfile, map_location='cpu')
print(checkpoint.keys())
# 打印checkpoint的结构，查看是否包含模型信息
#print(checkpoint.keys())

# 如果有 'model' 键，可以提取模型权重
if 'model' in checkpoint:
    model_weights = checkpoint['model']
else:
    # 如果没有 'model' 键，直接使用checkpoint本身
    model_weights = checkpoint

# 计算并打印总参数数量
total_params = sum(p.numel() for p in model_weights.values() if isinstance(p, torch.Tensor))
print(f'Total parameters: {total_params}')

# 查看每一层的参数数量
for name, param in model_weights.items():
    if isinstance(param, torch.Tensor):
        print(f"{name}: {param.numel()} parameters")
