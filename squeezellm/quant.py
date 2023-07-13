import numpy as np
import torch
import torch.nn as nn
import math
import quant_cuda

# drop-in layer replacement class
class QuantLinearLUT(nn.Module):
    def __init__(self, bits, infeatures, outfeatures, bias, include_sparse=False, numvals=0, topX=10):
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

        self.include_sparse = include_sparse
        self.numvals = numvals
        self.topX = topX
        if numvals > 0:
            self.register_buffer('rows', torch.zeros(outfeatures+1, dtype=torch.int32))
            self.register_buffer('cols', torch.zeros(numvals, dtype=torch.int32))
            self.register_buffer('vals', torch.zeros(numvals, dtype=torch.float32))
        if topX > 0:
            self.register_buffer('full_rows', torch.zeros((infeatures, topX), dtype=torch.float32))
            self.register_buffer('full_row_indices', torch.zeros(topX, dtype=torch.int32))

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
                if self.include_sparse and self.topX > 0:
                    quant_cuda.vecquant3matmul_spmv_hybrid_nuq_perchannel(
                        self.rows, 
                        self.cols, 
                        self.vals, 
                        x, 
                        self.full_rows, 
                        self.full_row_indices, 
                        y, 
                        self.outfeatures, 
                        self.qweight, 
                        self.lookup_table,
                    )
                elif self.include_sparse:
                    quant_cuda.vecquant3matmul_spmv_nuq_perchannel(
                        self.rows, 
                        self.cols, 
                        self.vals, 
                        x, 
                        y, 
                        self.outfeatures, 
                        self.qweight, 
                        self.lookup_table,
                    )
                else:
                    quant_cuda.vecquant3matmul_nuq_perchannel(
                        x, 
                        self.qweight, 
                        y, 
                        self.lookup_table,
                    )
            elif self.bits == 4:
                x = x.float()
                if self.include_sparse and self.topX > 0:
                    quant_cuda.vecquant4matmul_spmv_hybrid_nuq_perchannel(
                        self.rows, 
                        self.cols, 
                        self.vals, 
                        x, 
                        self.full_rows, 
                        self.full_row_indices, 
                        y, 
                        self.outfeatures, 
                        self.qweight, 
                        self.lookup_table,
                    )
                elif self.include_sparse:
                    quant_cuda.vecquant4matmul_spmv_nuq_perchannel(
                        self.rows, 
                        self.cols, 
                        self.vals, 
                        x, 
                        y, 
                        self.outfeatures, 
                        self.qweight, 
                        self.lookup_table,
                    )
                else:
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
                if self.include_sparse and self.topX > 0:
                    quant_cuda.vecquant3matmul_spmv_hybrid_nuq_perchannel_batched(
                        self.rows, 
                        self.cols, 
                        self.vals, 
                        x, 
                        self.full_rows, 
                        self.full_row_indices, 
                        out, 
                        self.outfeatures, 
                        self.qweight, 
                        self.lookup_table,
                    )
                elif self.include_sparse:
                    quant_cuda.vecquant3matmul_spmv_nuq_perchannel_batched(
                        self.rows, 
                        self.cols, 
                        self.vals, 
                        x, 
                        out, 
                        self.outfeatures, 
                        self.qweight, 
                        self.lookup_table,
                    )
                else:
                    quant_cuda.vecquant3matmul_nuq_perchannel_batched(
                        x, 
                        self.qweight, 
                        out, 
                        self.lookup_table,
                    )
            elif self.bits == 4:
                x = x.float()
                if self.include_sparse and self.topX > 0:
                    quant_cuda.vecquant4matmul_spmv_hybrid_nuq_perchannel_batched(
                        self.rows, 
                        self.cols, 
                        self.vals, 
                        x, 
                        self.full_rows, 
                        self.full_row_indices, 
                        out, 
                        self.outfeatures, 
                        self.qweight, 
                        self.lookup_table,
                    )
                elif self.include_sparse:
                    quant_cuda.vecquant4matmul_spmv_nuq_perchannel_batched(
                        self.rows, 
                        self.cols, 
                        self.vals, 
                        x, 
                        out, 
                        self.outfeatures, 
                        self.qweight, 
                        self.lookup_table,
                    )
                else:
                    quant_cuda.vecquant4matmul_nuq_perchannel_batched(
                        x, 
                        self.qweight, 
                        out, 
                        self.lookup_table,
                    )
            out = out.to(dtype)
            out = out.reshape(out_shape)
            out = out + self.bias if self.bias is not None else out
            return out

# function to iterate through model layers and replace with our LUT-based layer
def make_quant_lut(module, names, bits, name='', include_sparse=False, numvals=None, topX=0):
    if isinstance(module, QuantLinearLUT):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 not in names:
            continue
        num = 0
        if numvals:
            num = getattr(numvals[name1])
        delattr(module, attr)
        setattr(
            module, 
            attr, 
            QuantLinearLUT(
                bits, 
                tmp.in_features, 
                tmp.out_features, 
                tmp.bias is not None, 
                include_sparse=include_sparse, 
                numvals=num, 
                topX=topX,
            ),
        )

    for name1, child in module.named_children():
        make_quant_lut(
            child, 
            names, 
            bits, 
            name + '.' + name1 if name != '' else name1, 
            include_sparse=include_sparse, 
            numvals=numvals, 
            topX=topX,
        )
