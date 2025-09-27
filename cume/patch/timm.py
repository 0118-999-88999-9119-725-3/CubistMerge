import torch

from timm.models.vision_transformer import VisionTransformer
from cume.cume import CubistMerger

def apply_patch(model, loc: int, rh: int, rw: int, hw=(14,14), pad_token_count=1):
    class CuMeWrapper(torch.nn.Module):
        def __init__(self, original_block, rh, rw, hw, pad_token_count):
            super().__init__()
            self.original_block = original_block
            self.cubist_merger = CubistMerger()
            self.rh = rh
            self.rw = rw
            self.hw = hw
            self.pad_token_count = pad_token_count
            
        def forward(self, x):
            self.cubist_merger.init(self.hw)
            merged = self.cubist_merger.merge(x[:, self.pad_token_count:, :], self.rh, self.rw)
            x = torch.cat((x[:, :self.pad_token_count, :], merged), dim=1)
            return self.original_block(x)
    
    original_block = model.blocks[loc]
    model.blocks[loc] = CuMeWrapper(original_block, rh, rw, hw, pad_token_count)