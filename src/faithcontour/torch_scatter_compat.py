"""
Drop-in replacement for torch_scatter using PyTorch native ops.
Works on both CUDA and ROCm without needing the torch_scatter C++ extension.
"""
import torch


def scatter_sum(src, index, dim=0, dim_size=None, fill_value=0):
    if dim_size is None:
        dim_size = int(index.max()) + 1
    shape = list(src.shape)
    shape[dim] = dim_size
    out = src.new_full(shape, fill_value)
    idx = index
    for _ in range(src.dim() - 1):
        idx = idx.unsqueeze(-1)
    idx = idx.expand_as(src)
    return out.scatter_add_(dim, idx, src)


def scatter_mean(src, index, dim=0, dim_size=None, fill_value=0):
    s = scatter_sum(src, index, dim=dim, dim_size=dim_size, fill_value=0)
    ones = src.new_ones(src.shape)
    count = scatter_sum(ones, index, dim=dim, dim_size=dim_size, fill_value=0)
    count = count.clamp_min(1)
    return s / count


def scatter_max(src, index, dim=0, dim_size=None, fill_value=float('-inf')):
    if dim_size is None:
        dim_size = int(index.max()) + 1
    shape = list(src.shape)
    shape[dim] = dim_size
    out = src.new_full(shape, fill_value)
    arg = index.new_full(shape, -1)
    idx = index
    for _ in range(src.dim() - 1):
        idx = idx.unsqueeze(-1)
    idx = idx.expand_as(src)
    out.scatter_reduce_(dim, idx, src, reduce='amax', include_self=False)
    mask = out != fill_value
    arange = torch.arange(src.size(dim), device=src.device)
    dim_shape = [1] * src.dim()
    dim_shape[dim] = -1
    arange = arange.view(dim_shape).expand_as(src)
    arg.scatter_reduce_(dim, idx, arange, reduce='amax', include_self=False)
    return out, arg


def scatter_min(src, index, dim=0, dim_size=None, fill_value=float('inf')):
    if dim_size is None:
        dim_size = int(index.max()) + 1
    shape = list(src.shape)
    shape[dim] = dim_size
    out = src.new_full(shape, fill_value)
    arg = index.new_full(shape, -1)
    idx = index
    for _ in range(src.dim() - 1):
        idx = idx.unsqueeze(-1)
    idx = idx.expand_as(src)
    out.scatter_reduce_(dim, idx, src, reduce='amin', include_self=False)
    arange = torch.arange(src.size(dim), device=src.device)
    dim_shape = [1] * src.dim()
    dim_shape[dim] = -1
    arange = arange.view(dim_shape).expand_as(src)
    arg.scatter_reduce_(dim, idx, arange, reduce='amin', include_self=False)
    return out, arg


def scatter_softmax(src, index, dim=0, dim_size=None):
    if dim_size is None:
        dim_size = int(index.max()) + 1
    max_val, _ = scatter_max(src, index, dim=dim, dim_size=dim_size)
    idx = index
    for _ in range(src.dim() - 1):
        idx = idx.unsqueeze(-1)
    idx_exp = idx.expand_as(src)
    src_centered = src - max_val.gather(dim, idx_exp)
    exp_src = src_centered.exp()
    sum_exp = scatter_sum(exp_src, index, dim=dim, dim_size=dim_size)
    return exp_src / sum_exp.gather(dim, idx_exp).clamp_min(1e-12)
