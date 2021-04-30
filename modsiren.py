import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

def exists(val):
    return val is not None


def cast_tuple(val, repeat = 1):
    return val if isinstance(val, tuple) else ((val,) * repeat)


class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)


class Siren(nn.Module):
    def __init__(self, dim_in, dim_out, w0 = 1., c = 6., is_first = False, use_bias = True, activation = None):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c = c, w0 = w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if exists(bias):
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out =  F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        return out



class ReLUNet(nn.Module):
    def __init__(self, dim_in, dim_out, w0 = 1., c = 6., is_first = False, use_bias = True, activation = None):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c = c, w0 = w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        #self.activation = nn.ReLU(True) 
        self.activation = nn.LeakyReLU(0.2, True) 

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if exists(bias):
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out =  F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        return out



class SirenNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, latent_dim, num_layers, w0 = 1., w0_initial = 30., use_bias = True, final_activation = None):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        # Siren = ReLUNet

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            self.layers.append(Siren(
                dim_in = layer_dim_in,
                dim_out = dim_hidden,
                w0 = layer_w0,
                use_bias = use_bias,
                is_first = is_first
            ))

        self.modulator = Modulator(
            dim_in = latent_dim,
            dim_hidden = dim_hidden,
            num_layers = num_layers
        )

        final_activation = nn.Identity() if not exists(final_activation) else final_activation
        self.last_layer = Siren(dim_in = dim_hidden, dim_out = dim_out, w0 = w0, use_bias = use_bias, activation = final_activation)

    def forward(self, z, x):
        mods = self.modulator(z)

        for layer, mod in zip(self.layers, mods):
            x = layer(x)
            x = x * mod.unsqueeze(1)

        return self.last_layer(x)


class Modulator(nn.Module):
    def __init__(self, dim_in, dim_hidden, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([])

        for ind in range(num_layers):
            is_first = ind == 0
            dim = dim_in if is_first else (dim_hidden + dim_in)

            self.layers.append(nn.Sequential(
                nn.Linear(dim, dim_hidden),
                nn.ReLU()
            ))

    def forward(self, z):
        x = z
        hiddens = []

        for layer in self.layers:
            x = layer(x)
            hiddens.append(x)
            x = torch.cat((x, z), -1)
        return tuple(hiddens)



class PreNormTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(PreNormTransformerEncoderLayer, self).__init__(d_model, nhead, dim_feedforward, dropout, activation)

    def forward(self, src, src_mask, src_key_padding_mask):
        src2 = self.norm1(src)
        src2 = self.self_attn(src2, src2, src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)

        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src


class ProjectionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, activ='none'):
        super(ProjectionLayer, self).__init__()
        self.layer = nn.Linear(input_dim, output_dim)
        if activ == 'none':
            self.activ = nn.Identity()
        elif activ == 'relu':
            self.activ = nn.ReLU(True)
        elif activ == 'leaky_relu':
            self.activ = nn.LeakyReLU(0.2, True)
        else:
            raise ValueError

    def forward(self, h, t):
        assert h.ndim == t.ndim
        out = torch.cat([h, t], dim=-1)
        out = self.layer(out)
        out = self.activ(out)
        return out 
