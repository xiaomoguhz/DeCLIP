from torch import Tensor
from torch.nn import MultiheadAttention
from torch.nn import functional as F
from typing import Optional, Tuple


class MultiheadSelfAttention(MultiheadAttention):
    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None, return_tokens: bool = False) \
            -> Tuple[Tensor, Tensor]:
        assert query is value and value is key       # self-attention
        if return_tokens:
        # in_projection
            tokens = F.linear(value, self.in_proj_weight, bias=self.in_proj_bias)[..., -self.embed_dim:]
            # out_projection
            tokens = F.linear(tokens, self.out_proj.weight, bias=self.out_proj.bias)
        else:
            tokens = None

        attn_output, attn_output_weights = F.multi_head_attention_forward(
            query=query, key=key, value=value,
            embed_dim_to_check=self.embed_dim,
            num_heads=self.num_heads,
            in_proj_weight=self.in_proj_weight,
            in_proj_bias=self.in_proj_bias,
            bias_k=None, bias_v=None,
            add_zero_attn=False,
            dropout_p=0.,
            out_proj_weight=self.out_proj.weight,
            out_proj_bias=self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask, need_weights=need_weights,
            attn_mask=attn_mask)

        return attn_output, tokens   # , attn_output_weights
