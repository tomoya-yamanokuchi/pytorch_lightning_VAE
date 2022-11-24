import torch
import torch.nn as nn


relu = nn.ReLU(inplace=False)

inp = torch.tensor([-1, 0, 1])
out = relu(inp)

print(inp)
print(out)