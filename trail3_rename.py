import torch

checkpoint_path = 'ViT-B-16_60.pth'
checkpoint = torch.load(checkpoint_path)

# 创建一个新的字典，将键名前加上 'backbone.' 前缀
new_checkpoint = {}
for key, value in checkpoint.items():
    # 修改键名前缀
    new_key = 'backbone.' + key
    new_checkpoint[new_key] = value

# 将修改后的 checkpoint 保存到新的文件中
torch.save(new_checkpoint, 'modified_checkpoint.pth')