# 创建两个张量
import torch

tensor1 = torch.tensor([[1, 2], [3, 4]])
tensor2 = torch.tensor([[5, 6], [7, 8]])
print(tensor1.shape)
print(tensor2.shape)
# dim=0：垂直拼接（增加行数）
concat_dim0 = torch.cat([tensor1, tensor2], dim=1)
print("dim=0 拼接结果:")
print(concat_dim0.shape)
# 输出:
# tensor([[1, 2],
#         [3, 4],
#         [5, 6],
#         [7, 8]])

# dim=1：水平拼接（增加列数）
concat_dim1 = torch.cat([tensor1, tensor2], dim=1)
print("\ndim=1 拼接结果:")
print(concat_dim1)
# 输出:
# tensor([[1, 2, 5, 6],
#         [3, 4, 7, 8]])