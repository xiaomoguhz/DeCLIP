from math import pi
import torch
from torch import nn
from einops import rearrange, repeat
import logging
import torch.nn.functional as F


def broadcat(tensors, dim = -1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim = dim)

def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')


class VisionRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len,
        ft_seq_len=None,
        custom_freqs = None,
        freqs_for = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
    ):
        super().__init__()
        self.ft_seq_len = ft_seq_len
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        if ft_seq_len is None: ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs_h = torch.einsum('..., f -> ... f', t, freqs)
        freqs_h = repeat(freqs_h, '... n -> ... (n r)', r = 2)

        freqs_w = torch.einsum('..., f -> ... f', t, freqs)
        freqs_w = repeat(freqs_w, '... n -> ... (n r)', r = 2)

        freqs = broadcat((freqs_h[:, None, :], freqs_w[None, :, :]), dim = -1) 

        self.register_buffer("freqs_cos", freqs.cos())
        self.register_buffer("freqs_sin", freqs.sin())

        logging.info(f'Shape of rope freq: {self.freqs_cos.shape}')

    def interpolate_freq(self, t_len, freq):
        if t_len == self.ft_seq_len ** 2:
            return freq
        tar_size = int(t_len ** 0.5)
        freq = freq.view(1, self.ft_seq_len, self.ft_seq_len, freq.shape[-1]).permute(0, 3, 1, 2)
        freq = F.interpolate(freq, (tar_size, tar_size), mode='bicubic',
                             align_corners=False).view(-1, t_len).T

        return freq

    def forward(self, t, start_index = 0):
        rot_dim = self.freqs_cos.shape[-1]
        end_index = start_index + rot_dim
        assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
        t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
        # t = (t * self.freqs_cos) + (rotate_half(t) * self.freqs_sin)

        t = (t * self.interpolate_freq(t.shape[2], self.freqs_cos)) \
            + (rotate_half(t) * self.interpolate_freq(t.shape[2], self.freqs_sin))

        return torch.cat((t_left, t, t_right), dim = -1)


class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len,
        ft_seq_len=None,
        custom_freqs = None,
        freqs_for = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
        patch_dropout = 0.
    ):
        super().__init__()
        self.custom_freqs = custom_freqs
        self.pt_seq_len = pt_seq_len
        self.ft_seq_len = ft_seq_len
        self.freqs_for = freqs_for
        self.dim = dim
        self.theta = theta
        self.max_freq = max_freq
        self.num_freqs = num_freqs
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        if ft_seq_len is None: ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs = torch.einsum('..., f -> ... f', t, freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r = 2)
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim = -1)

        freqs_cos = freqs.cos().view(-1, freqs.shape[-1])
        freqs_sin = freqs.sin().view(-1, freqs.shape[-1])

        self.patch_dropout = patch_dropout

        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

        logging.info(f'Shape of rope freq: {self.freqs_cos.shape}')
        self.register_buffer("flag", torch.tensor(0, dtype=torch.long),
                             persistent=False)

    def forward(self, t, patch_indices_keep=None):
        if patch_indices_keep is not None:
            batch = t.size()[0]
            batch_indices = torch.arange(batch)
            batch_indices = batch_indices[..., None]

            freqs_cos = repeat(self.freqs_cos, 'i j -> n i m j', n=t.shape[0], m=t.shape[1])
            freqs_sin = repeat(self.freqs_sin, 'i j -> n i m j', n=t.shape[0], m=t.shape[1])

            freqs_cos = freqs_cos[batch_indices, patch_indices_keep]
            freqs_cos = rearrange(freqs_cos, 'n i m j -> n m i j')
            freqs_sin = freqs_sin[batch_indices, patch_indices_keep]
            freqs_sin = rearrange(freqs_sin, 'n i m j -> n m i j')

            return  t * freqs_cos + rotate_half(t) * freqs_sin
        freqs_cos, freqs_sin = self.recalculate(t)
        return t * freqs_cos + rotate_half(t) * freqs_sin
        # return  t * self.freqs_cos + rotate_half(t) * self.freqs_sin
        # return t * self.interpolate_freq(t.shape[2], self.freqs_cos) \
        #     + rotate_half(t) * self.interpolate_freq(t.shape[2], self.freqs_sin)

    def interpolate_freq(self, t_len, freq):
        if t_len == self.ft_seq_len ** 2:
            return freq
        tar_size = int(t_len ** 0.5)
        freq = freq.view(1, self.ft_seq_len, self.ft_seq_len, freq.shape[-1]).permute(0, 3, 1, 2)
        freq = F.interpolate(freq, (tar_size, tar_size), mode='bicubic',
                             align_corners=False).view(-1, t_len).T

        return freq

    def recalculate(self, x):
        # TODO: fix it, do not calculate it every time
        x_len = x.shape[2]
        if x_len == self.ft_seq_len ** 2:
            return self.freqs_cos, self.freqs_sin
        elif hasattr(self, f"freqs_cos_{x_len}"):
            return getattr(self, f"freqs_cos_{x_len}"), getattr(self, f"freqs_sin_{x_len}")
        assert self.flag <= 4
        ft_seq_len = int(x_len ** 0.5)
        if self.custom_freqs:
            freqs = self.custom_freqs
        elif self.freqs_for == 'lang':
            freqs = 1. / (self.theta ** (torch.arange(0, self.dim, 2)[:(self.dim // 2)].float() / self.dim))
        elif self.freqs_for == 'pixel':
            freqs = torch.linspace(1., self.max_freq / 2, self.dim // 2) * pi
        elif self.freqs_for == 'constant':
            freqs = torch.ones(self.num_freqs).float()
        else:
            raise ValueError(f'unknown modality {self.freqs_for}')

        t = torch.arange(ft_seq_len) / ft_seq_len * self.pt_seq_len

        freqs = torch.einsum('..., f -> ... f', t, freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r = 2)
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim = -1)

        freqs_cos = freqs.cos().view(-1, freqs.shape[-1]).to(x)
        freqs_sin = freqs.sin().view(-1, freqs.shape[-1]).to(x)
        # TODO this is just a workaround
        self.register_buffer(f"freqs_cos_{x_len}", freqs_cos, persistent=False)
        self.register_buffer(f"freqs_sin_{x_len}", freqs_sin, persistent=False)
        self.flag.data += 1
        logging.info(f'Add a new rope freq of shape: {freqs_cos.shape}')
        print(f'Add a new rope freq of shape: {freqs_cos.shape}', flush=True)

        return freqs_cos, freqs_sin
