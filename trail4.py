import torch
import torchvision.models as models
ckpt = torch.load('/data/zyr/fast-reid-master/projects/FastDistill/logs/sports/r34/model_best.pth', map_location='cpu')

# 提取模型参数
state_dict = ckpt['model']

# 创建标准的 ResNet34 模型
model = models.resnet34()

# 加载参数（如有 key 不匹配，提示我我来帮你处理）
model.load_state_dict(state_dict)

# 统计参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

total_params = count_parameters(model)
print(f"ResNet34 参数总量: {total_params / 1e6:.2f}M") 