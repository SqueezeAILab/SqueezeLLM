import numpy as np
import torch
import torch.nn as nn
import math
import quant_cuda

# drop-in layer replacement class
class QuantLinearLUT(nn.Module):
    def __init__(self, bits, infeatures, outfeatures, bias):
        super().__init__()
        if bits not in [3,4]:
            raise NotImplementedError("Only 3 and 4 bits is supported.")
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits

        self.register_buffer('qweight', torch.zeros((infeatures // 32 * self.bits, outfeatures), dtype=torch.int32))
        if bias:
            self.include_bias = True
            self.register_buffer('bias', torch.zeros((outfeatures)))
        else:
            self.include_bias = False
            self.bias = None

        self.register_buffer('lookup_table', torch.zeros((outfeatures, 2**self.bits), dtype=torch.float32))

    #replacement forward pass
    def forward(self, x):
        if x.shape[-1] == x.numel():
            outshape = list(x.shape)
            if self.bias is not None:
                y = self.bias.clone()
                outshape[-1] = self.bias.numel()
            else:
                y = torch.zeros((self.outfeatures), device='cuda', dtype=torch.float32)
                outshape[-1] = self.outfeatures
            dtype = x.dtype
            if self.bits == 3:
                x = x.float()
                quant_cuda.vecquant3matmul_nuq_perchannel(x, self.qweight, y, self.lookup_table)
            elif self.bits == 4:
                x = x.float()
                quant_cuda.vecquant4matmul_nuq_perchannel(x, self.qweight, y, self.lookup_table)
            y = y.to(dtype)
            return y.reshape(outshape)
        else:
            out_shape = x.shape[:-1] + (self.outfeatures, )
            x = x.reshape(-1,x.shape[-1])
            out = torch.zeros((x.shape[0], self.outfeatures), device='cuda', dtype=torch.float32)
            dtype = x.dtype
            if self.bits == 3:
                x = x.float()
                quant_cuda.vecquant3matmul_nuq_perchannel_batched(x, self.qweight, out, self.lookup_table)
            elif self.bits == 4:
                x = x.float()
                quant_cuda.vecquant4matmul_nuq_perchannel_batched(x, self.qweight, out, self.lookup_table)
            out = out.to(dtype)
            out = out.reshape(out_shape)
            out = out + self.bias if self.bias is not None else out
            return out

# function to iterate through model layers and replace with our LUT-based layer
def make_quant_lut(module, names, bits, name=''):
    if isinstance(module, QuantLinearLUT):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in names:
            delattr(module, attr)
            setattr(module, attr, QuantLinearLUT(bits, tmp.in_features, tmp.out_features, tmp.bias is not None))
    for name1, child in module.named_children():
        make_quant_lut(child, names, bits, name + '.' + name1 if name != '' else name1)
