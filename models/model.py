import torch
import torch.nn.functional as F
from torch import nn

from models.blocks import HBlock, LBlock


class Model(nn.Module):
    def __init__(self, depth=2):
        super(Model, self).__init__()
        self.depth = depth
        self.LowFrequency = LBlock()
        self.HighFrequency = nn.ModuleList([HBlock() for _ in range(self.depth)])

    def laplacian_pyramid_decomposition(self, img):
        current = img
        pyramid = []
        for i in range(self.depth):
            blurred = F.interpolate(current, scale_factor=0.5, mode='bicubic', align_corners=True)
            expanded = F.interpolate(blurred, current.shape[2:], mode='bicubic', align_corners=True)
            residual = current - expanded
            pyramid.append(residual)
            current = blurred
        pyramid.append(current)
        return pyramid

    def laplacian_pyramid_reconstruction(self, pyramid):
        current = pyramid[-1]
        for i in reversed(range(self.depth)):
            expanded = F.interpolate(current, pyramid[i].shape[2:], mode='bicubic', align_corners=True)
            current = expanded + pyramid[i]
            current = self.HighFrequency[i](current)
        return current

    def forward(self, inp):
        inps = self.laplacian_pyramid_decomposition(inp)
        inps[-1] = self.LowFrequency(inps[-1])
        res = self.laplacian_pyramid_reconstruction(inps)
        return res


if __name__ == '__main__':
    t = torch.randn(1, 3, 600, 400).cuda()
    model = Model().cuda()
    out = model(t)
    print(out.shape)
