import torch
import random
# 创建示例张量
torch.manual_seed(42)
tensor = torch.rand([5,2])
print(tensor)
norm = torch.nn.LayerNorm([5,2])
output = norm(tensor)
print(output.shape)
print(output)
# 按第一维度倒序排列
# print(tensor.shape)
# reversed_tensor = tensor.flip(dims=[1])

# print("Original tensor:")
# print(tensor)
# print("\nReversed tensor:")
# print(reversed_tensor, reversed_tensor.shape)
conv = torch.nn.Conv1d(2,10,3,padding=1)
tensor2 = torch.rand([2,10])
tensor2out = conv(tensor2)
print(tensor2out.shape)

list1 = [0,1,2,3,4]
random.shuffle(list1)
print(list1)

m = torch.nn.GLU(dim=-2)
input = torch.randn(4, 3)
output = m(input)
print(output.shape)