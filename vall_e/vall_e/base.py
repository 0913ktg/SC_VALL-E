import math
import numpy as np 
from functools import partial
from typing import Literal, overload
import torch.nn.init as init


import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, einsum, nn
from torch.distributions import Categorical
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint
import pickle


def _create_mask(l, device):
    """1 is valid region and 0 is invalid."""
    seq = torch.arange(max(l), device=device).unsqueeze(0)  # (1 t)
    stop = torch.tensor(l, device=device).unsqueeze(1)  # (b 1)
    return (seq < stop).float()  # (b t)


def list_to_tensor(x_list: list[Tensor], pattern="t b c -> b t c"):
    """
    Args:
        x_list: [(t d)]
    Returns:
        x: (? ? ?)
        m: (? ? ?), same as x
    """
    l = list(map(len, x_list))
    x = rearrange(pad_sequence(x_list), pattern)
    m = _create_mask(l, x_list[0].device)
    m = m.t().unsqueeze(-1)  # (t b 1)
    m = rearrange(m, pattern)
    m = m.to(x)
    return x, m


###########################
# style token layer start
###########################


class MultiHeadAttention(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
    output:
        out --- [N, T_q, num_units]
    '''
    def __init__(self, query_dim, key_dim, num_units, num_heads):
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query, key):
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)

        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

        # score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim ** 0.5)
        scores = F.softmax(scores, dim=3)

        # out = score * V
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

        return out
    
class STL(nn.Module):
    '''
    inputs --- [N, E//2]
    '''
    def __init__(self, E, token_num, num_heads):
        super().__init__()
        self.embed = nn.Parameter(torch.FloatTensor(token_num, E // num_heads))
        d_q = E // 2
        d_k = E // num_heads
        self.attention = MultiHeadAttention(query_dim=d_q, key_dim=d_k, num_units=E, num_heads=num_heads)
        # style controllable vall-e test
        self.lastlin = nn.Linear(E, E//2)

        init.normal_(self.embed, mean=0, std=0.5)
    def forward(self, inputs, is_scale = False):        
        N = inputs.size(0)
        query = inputs.unsqueeze(1)  # [N, 1, E//2]
        if is_scale:
            # style scaling을 하기 위해 style embed의 값을 조정
            multiplier = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.float16)
            scaled_embed = self.embed * multiplier.unsqueeze(1).cuda()
            keys = torch.tanh(scaled_embed).unsqueeze(0).expand(N, -1, -1)  # [N, token_num, E // num_heads]
        else:
            keys = torch.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)  # [N, token_num, E // num_heads]
            
        style_embed = self.attention(query, keys)            
        
        # style controllable vall-e test
        style_embed = self.lastlin(style_embed)

        return style_embed

#########################
# style token layer end
#########################


class SinusodialEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        exponent = torch.arange(self.d_half, dtype=torch.float32)
        exponent = exponent / self.d_half
        omega = torch.exp(-math.log(1e4) * exponent)
        self.omega: torch.Tensor
        self.register_buffer("omega", omega, persistent=False)

    @property
    def d_half(self):
        assert self.d_model % 2 == 0, "Only support even d_model."
        return self.d_model // 2

    def forward(self, x):
        """
        Args:
            x: (...)
        Returns:
            pe: (... d)
        """
        omega = self.omega

        while omega.dim() <= x.dim():
            omega = omega.unsqueeze(0)  # (... d)

        x = x.unsqueeze(-1)  # (... 1)
        x = omega * x
        x = torch.cat([x.sin(), x.cos()], dim=-1)

        return x

    def get_pe(self, n: int):
        """
        Args:
            n: int
        Returns:
            pe: (n d)
        """
        device = self.omega.device
        return self.forward(torch.arange(n, device=device))

    def add_pe(self, x):
        """
        Args:
            x: (b t c)
        """
        e = self.get_pe(x.shape[1])  # t d
        e = e[None]  # b t d
        x = x + e
        return x


class Attention(nn.Module):
    def __init__(self, d_model, n_heads, casual):
        super().__init__()
        assert d_model % n_heads == 0
        dim_head = d_model // n_heads
        self.casual = casual
        self.n_heads = n_heads
        self.scale = dim_head**-0.5
        self.to_qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.to_out = nn.Linear(d_model, d_model)
    
    def softmax(self, input_tensor, dimension=2):
        # Logit scaling
        max_value, _ = torch.max(input_tensor, dim=dimension, keepdim=True)
        scaled_input = input_tensor - max_value

        # Exponential and sum
        exp_values = torch.exp(scaled_input)
        sum_exp = torch.sum(exp_values, dim=dimension, keepdim=True)

        # Softmax
        softmax_output = exp_values / sum_exp
        
        # Apply exception handling for specific vectors
        if dimension == 2:
            invalid_mask = torch.isnan(softmax_output)
            softmax_output[invalid_mask] = 0.0
        
        return softmax_output
    
    def forward(self, x, m):
        """
        Args:
            x: (b t c)
            m: (b t c), 1 is data, 0 is padding
        Returns:
            x: (b t c)
        """
        h = self.n_heads

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)        
        q, k, v = map(lambda t: rearrange(t, "b t (h d) -> b t h d", h=h), (q, k, v))

        e = einsum("b i h d, b j h d -> b i j h", q, k)
        e = e * self.scale
        kpm = m.unsqueeze(1) * m.unsqueeze(2)  # b i j 1

        if self.casual:
            kpm = kpm.squeeze(-1).tril().unsqueeze(-1)  # b i j 1

        e = e.masked_fill(kpm == 0, -torch.finfo(e.dtype).max)      
        a = self.softmax(e, dimension = 2)
        o = einsum("b i j h, b j h d -> b i h d", a, v)
        o = o.flatten(-2)
        o = self.to_out(o)  # b t c
        o = o * m
        
        return o

class AdaLN(nn.Module):
    def __init__(self, d_model, n_levels, eps=1e-5, k=0.1, c=2):
        super().__init__()
        self.eps = eps
        self.emb = nn.Embedding(n_levels, d_model * 2)
        self.k = k
        self.c = c
        nn.init.zeros_(self.emb.weight)

    def forward(self, x, l):
        logγ, β = self.emb(l).unsqueeze(1).chunk(2, dim=-1)

        h = F.layer_norm(x, x.shape[-1:], eps=self.eps)

        # The initial implementation (https://github.com/enhuiz/vall-e/blob/fbf023448c08e55c0422eefed7fc234cf8b76680/vall_e/vall_e/base.py#L135)
        # performed worse than vanilla LayerNorm.
        # The authors mentioned another AdaNorm paper (https://openreview.net/pdf?id=HyxndNrxLB) as they introduce AdaLN.
        # Did they use AdaNorm inside AdaLN? (as follows)
        h = self.c * (1 - (self.k * h).detach()) * h

        y = logγ.exp() * h + β

        return y


class PrenormResidual(nn.Module):
    def __init__(
        self,
        block,
        d_model,
        p_dropout,
        requires_mask=False,
        norm_type="ln",
        n_levels: int | None = None,
    ):
        super().__init__()
        self.block = block
        self.requires_mask = requires_mask
        self.norm_type = norm_type
        if norm_type == "ln":
            self.norm = nn.LayerNorm(d_model)
        elif norm_type == "adaln":
            assert n_levels is not None
            self.norm = AdaLN(d_model, n_levels)
        else:
            raise NotImplementedError(norm_type)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x, m, l):
        """
        Args:
            x: input (b t d)
            m: mask (b t 1), 1 is valuable and 0 is padding
            l: level to use, required only for AdaLN
        """
        nopts = {"l": l} if self.norm_type == "adaln" else {}
        bopts = {"m": m} if self.requires_mask else {}
        x = x + self.dropout(self.block(self.norm(x, **nopts) * m, **bopts))
        return x * m


class Block(nn.Sequential):
    def __init__(self, d_model, n_heads, p_dropout, casual, norm_type, n_levels):
        super().__init__()
        self.attn = PrenormResidual(
            Attention(d_model, n_heads, casual),
            d_model=d_model,
            p_dropout=p_dropout,
            requires_mask=True,
            norm_type=norm_type,
            n_levels=n_levels,
        )
        self.ffn = PrenormResidual(
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(p_dropout),
                nn.Linear(d_model * 4, d_model),
            ),
            d_model=d_model,
            p_dropout=p_dropout,
            norm_type=norm_type,
            n_levels=n_levels,
        )

    def forward(self, x, m, l):
        """
        Args:
            x: (b t c)
            m: (b t 1)
            l: (b)
        """
        poor_in_vram = True
        if x.requires_grad and poor_in_vram:
            x = checkpoint(self.attn, x, m, l)
        else:
            x = self.attn(x, m, l)
        attn_output = x
        x = self.ffn(x, m, l)
        return x, attn_output


class Embedding(nn.Embedding):
    def forward(self, x_list: list[Tensor]) -> list[Tensor]:
        
        # with open('x_list_2.pkl', 'wb') as f:
        #     pickle.dump(x_list, f)
        
        if len(x_list) == 0:
            return []
        
        return super().forward(torch.cat(x_list)).split([*map(len, x_list)])


class MultiEmbedding(nn.Module):
    """
    This embedding sums embeddings on different levels.
    """

    def __init__(self, max_n_levels, n_tokens, token_dim):
        super().__init__()
        self.max_n_levels = max_n_levels
        self.n_tokens = n_tokens
        self.weight = nn.Parameter(torch.randn(max_n_levels, n_tokens, token_dim))

    def forward(self, x_list: list[Tensor]) -> list[Tensor]:
        if len(x_list) == 0:
            return []

        w = self.weight

        padded_x_list = []
                
        for xi in x_list:
            xi = F.one_hot(xi, num_classes=self.n_tokens)  # t l' k
            xi = F.pad(xi, (0, 0, 0, w.shape[0] - xi.shape[1]))  # t l k
            padded_x_list.append(xi.to(w))

        x = torch.cat(padded_x_list)  # n l k
        x = einsum("l k d, n l k -> n d", w, x)

        x_list = x.split([*map(len, x_list)])
        
        return x_list


def _join(x: tuple[Tensor], sep: Tensor):
    """
    Args:
        x: (k t d)
        sep: (d)
    """
    ret = x[0]
    for i in range(1, len(x)):
        ret = torch.cat((ret, sep[None], x[i]), dim=0)
        
    return ret


class Base(nn.Module):
    @property
    def casual(self) -> bool:
        raise NotImplementedError

    @property
    def n_resp_levels(self) -> int:
        raise NotImplementedError

    @property
    def use_stop_token(self) -> bool:
        raise NotImplementedError

    @property
    def norm_type(self):
        raise NotImplementedError

    @property
    def n_prom_levels(self) -> int:
        return 8

    @property
    def resp_loss_only(self):
        raise NotImplementedError

    def __init__(
        self,
        n_tokens: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 12,
        p_dropout: float = 0.1,
        is_style_layer: bool = False,
        style_token_num = 10,
        style_layer_head_num = 8,
        E = 2048
    ):
        super().__init__()
        self.n_tokens = n_tokens

        casual = self.casual
        self.is_style_layer = is_style_layer

        # +1 to include the stop token
        n_stop_tokens = 1 if self.use_stop_token else 0
        n_resp_tokens = n_tokens + n_stop_tokens
        
        self.text_emb = Embedding(n_tokens, d_model)

        # Here I simply use all prom levels        
        self.proms_emb = MultiEmbedding(self.n_prom_levels, n_tokens, d_model)
        self.resps_emb = MultiEmbedding(self.n_resp_levels, n_resp_tokens, d_model)

        # style token layer 추가
        if self.is_style_layer:
            self.style_layer = STL(E, style_token_num, style_layer_head_num)
            
        self.sin_emb = SinusodialEmbedding(d_model)
        self.sep = nn.Parameter(torch.randn(d_model))

        blocks = [
            Block(
                d_model=d_model,
                n_heads=n_heads,
                p_dropout=p_dropout,
                casual=casual,
                norm_type=self.norm_type,
                n_levels=self.n_resp_levels,
            )
            for _ in range(n_layers)
        ]

        self.blocks = nn.ModuleList(blocks)

        self.classifier = nn.Linear(d_model, n_resp_tokens)

    @property
    def stop_token(self):
        if not self.use_stop_token:
            raise ValueError("Not using stop token!")
        return self.n_tokens

    @property
    def ignore_index(self):
        return -100

    @staticmethod
    def _samplewise_merge_tensors(*l, sep: Tensor | None):
        
        if sep is None:
            cat = torch.cat
        else:
            cat = partial(_join, sep=sep)
            
            
        return [*map(cat, zip(*l))]

    @overload
    def forward(
        self,
        text_list: list[Tensor],
        proms_list: list[Tensor],
        resps_list: list[Tensor],
        targ_list: list[Tensor] | None = None,
        quant_levels: Tensor | None = None,
        shift_targ_list: bool = False,
        return_all_resp: Literal[False] = False,
        sampling_temperature: float = 1.0,
    ) -> Tensor:
        ...

    @overload
    def forward(
        self,
        text_list: list[Tensor],
        proms_list: list[Tensor],
        resps_list: list[Tensor],
        targ_list: list[Tensor] | None = None,
        quant_levels: Tensor | None = None,
        shift_targ_list: bool = False,
        return_all_resp: Literal[True] = True,
        sampling_temperature: float = 1.0,
    ) -> list[Tensor]:
        ...

    def forward(
        self,
        text_list: list[Tensor],
        proms_list: list[Tensor],
        resps_list: list[Tensor],
        targ_list: list[Tensor] | None = None,
        quant_levels: Tensor | None = None,
        shift_targ_list: bool = False,
        return_all_resp: bool = False,
        sampling_temperature: float = 1.0,
    ):
        """
        Args:
            text_list: [t] * b
            proms_list: [t' l] * b, l quantization levels.
            resps_list: [t'' l] * b, l quantization levels.
            targ_list: [t''] * b, one quantization level only, when given, loss will be computed
            quant_levels: specify which quant_levels to feed forward, used in NAR mode.
            shift_targ_list: whether to shift target list when computing loss. True if AR.
            return_all_resp: True if NAR.
            sampling_temperature: a lower temperature makes the result more robust but less diverse.
        Returns:
            y: sampled tokens
        """
        if self.is_style_layer:
            p_emb = self.proms_emb(proms_list)
            proms_emb_vector = tuple([self.style_layer(x).reshape(-1,self.n_tokens) for x in p_emb])            
        else:
            proms_emb_vector = self.proms_emb(proms_list)
        
        x_list = self._samplewise_merge_tensors(
            self.text_emb(text_list),
            proms_emb_vector,
            self.resps_emb(resps_list),
            sep=self.sep,
        )

        x, m = list_to_tensor(x_list)
        
        x = self.sin_emb.add_pe(x)
        
        block_outputs = []

        for i, block in enumerate(self.blocks):
            x, attn_output = block(x, m, quant_levels)
            block_outputs.append((i, x, attn_output))

        h = self.classifier(x) * m

        # Remove padding
        h_list = [hi[:li] for hi, li in zip(h, map(len, x_list))]
        
        if targ_list is not None:
            if any([l == 0 for l in map(len, targ_list)]):
                raise ValueError("Cannot compute loss given empty targ_list.")
            
            device = h.device

            ignore_sep = torch.tensor(self.ignore_index, device=device)

            # Ignore prom in the target
            prom_list = [
                torch.full_like(t[..., 0], self.ignore_index) for t in proms_list
            ]

            text_prom_list = self._samplewise_merge_tensors(
                text_list, prom_list, sep=ignore_sep
            )

            # Make every token earlier as it is future that is unknown
            # If we don't want compute loss, set all to ignored
            for i in range(len(text_prom_list)):
                if self.resp_loss_only: 
                    text_prom_list[i][:] = self.ignore_index
                else:
                    text_prom_list[i] = text_prom_list[i].roll(-1, dims=0)
                    text_prom_list[i][-1] = self.ignore_index

            if shift_targ_list:
                # Also make target earlier if in autoregressive mode
                targ_list = [*targ_list]
                for i in range(len(targ_list)):
                    targ_list[i] = targ_list[i].roll(-1, dims=0)
                    targ_list[i][-1] = self.stop_token

            y_list = self._samplewise_merge_tensors(
                text_prom_list, targ_list, sep=ignore_sep
            )

            self.loss = dict(
                nll=F.cross_entropy(
                    torch.cat(h_list),
                    torch.cat(y_list),
                    ignore_index=self.ignore_index,
                )
            )        
        try:
            if return_all_resp:
                logits = [hi[-li:] for hi, li in zip(h_list, map(len, resps_list))]
                
                ret = [
                    Categorical(logits=hi / sampling_temperature).sample() for hi in logits
                ]
            else:
                logits = torch.stack([hi[-1] for hi in h_list])
                ret = Categorical(logits=logits / sampling_temperature).sample()
        except ValueError as e:
                # with open('/data/vall-e/logits.pkl', 'wb') as f:
                #     pickle.dump(block_outputs, f)
                raise e

        return ret
