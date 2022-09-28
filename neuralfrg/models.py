from typing import Sequence, Optional

from torch import nn
from torchdiffeq import odeint

from .utils import make_dense_layers, make_conv_layers

####################################################################


class Encoder(nn.Module):
    def __init__(self, dim: int, ldim: int, layer_sizes: Optional[Sequence[int]] = None):

        super(Encoder, self).__init__()

        self.dim = int(dim)
        self.ldim = int(ldim)

        if layer_sizes is not None:
            self.layer_sizes = [self.dim] + list(layer_sizes) + [self.ldim]
        else:
            self.layer_sizes = [self.dim, 8, 64, 128, 256, 512, self.ldim]

        layers = make_dense_layers(self.layer_sizes)
        self.model = nn.Sequential(*layers)

    def forward(self, x0):
        return self.model(x0)


####################################################################


class NODE(nn.Module):
    def __init__(
        self, ldim: int, ode_method: str = "dopri5", layer_sizes: Optional[Sequence[int]] = None
    ):

        super(NODE, self).__init__()

        self.ldim = int(ldim)
        self.method = str(ode_method)

        if layer_sizes is not None:
            self.layer_sizes = [self.ldim] + list(layer_sizes) + [self.ldim]
        else:
            self.layer_sizes = [self.ldim, 512, 512, self.ldim]

        layers = make_dense_layers(self.layer_sizes)
        self.kernel = nn.Sequential(*layers)

    def forward(self, z0, ts):
        zs = odeint(self.eval_kernel, z0, ts, method=self.method)
        zs.transpose_(0, 1)
        return zs

    def eval_kernel(self, t, z):
        return self.kernel(z)


####################################################################


class Decoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dims: Sequence[int] = (48, 48, 48),
        dense_layer_sizes: Optional[Sequence[int]] = None,
        conv_layers: Optional[Sequence[Sequence[int]]] = None,
        final_upsample: bool = False,
    ):

        super(Decoder, self).__init__()

        self.in_dim = int(in_dim)
        self.out_dims = tuple(out_dims)

        assert len(self.out_dims) == 3, f"Only 3D output is supported, got shape {self.out_dims}"

        if dense_layer_sizes is not None:
            self.dense_layer_sizes = [self.in_dim] + list(dense_layer_sizes)
        else:
            self.dense_layer_sizes = [self.in_dim, 128, 128]

        if conv_layers is not None:
            self.conv_layer_sizes = list(conv_layers)
        else:
            c = self.dense_layer_sizes[-1]

            self.conv_layer_sizes = [
                (c, 32, 8, 2),
                (32, 16, 8, 2),
                (16, 8, 4, 2),
                (8, 4, 2, 1),
                (4, 2, 2, 1),
                (2, 1, 1, 1),
            ]

        if self.dense_layer_sizes[-1] != self.conv_layer_sizes[0][0]:
            msg = f"Last dense layer size must match first conv layer size: got {self.dense_layer_sizes[-1]} and {self.conv_layer_sizes[0][0]}"
            raise RuntimeError(msg)

        dense_layers = make_dense_layers(self.dense_layer_sizes, final_activation=True)
        conv_layers = make_conv_layers(self.conv_layer_sizes)

        if final_upsample:
            conv_layers.append(nn.Upsample(size=self.out_dims, mode="trilinear"))

        self.denses = nn.Sequential(*dense_layers)
        self.model = nn.Sequential(*conv_layers)

    def forward(self, z):
        h = self.denses(z).view(-1, self.dense_layer_sizes[-1], 1, 1, 1)
        x = self.model(h)
        return x.view(*z.shape[:2], *self.out_dims)


####################################################################


class PNODE(nn.Module):
    def __init__(
        self,
        in_dim: int,
        ldim: int,
        out_dims: Sequence[int] = (48, 48, 48),
        ode_method: str = "dopri5",
        encoder_layer_sizes: Optional[Sequence[int]] = None,
        node_layer_sizes: Optional[Sequence[int]] = None,
        decoder_dense_layer_sizes: Optional[Sequence[int]] = None,
        decoder_conv_layers: Optional[Sequence[Sequence[int]]] = None,
    ):

        super(PNODE, self).__init__()

        self.in_dim = int(in_dim)
        self.ldim = int(ldim)
        self.out_dims = tuple(out_dims)

        self.encoder = Encoder(self.in_dim, self.ldim, layer_sizes=encoder_layer_sizes)
        self.node = NODE(self.ldim, ode_method=ode_method, layer_sizes=node_layer_sizes)
        self.decoder = Decoder(
            self.ldim,
            self.out_dims,
            dense_layer_sizes=decoder_dense_layer_sizes,
            conv_layers=decoder_conv_layers,
        )

    def forward(self, x0, ts):

        z0 = self.encoder(x0)
        zs = self.node(z0, ts)
        gs = self.decoder(zs)

        return gs
