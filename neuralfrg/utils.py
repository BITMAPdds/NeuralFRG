from typing import Optional, Union, Sequence, List

import torch
from torch import nn
from torch import Tensor, BoolTensor
from torch.nn import Module
from torch.optim import Optimizer

########################################################################################################################

@torch.jit.script
def masked_mse(data, pred, mask):
    mask = mask[:,:,None,None,None].expand_as(data).to(data.dtype)
    return torch.nn.functional.mse_loss(data*mask, pred*mask, reduction='sum')/mask.sum()

def normalize_data(vertices: Tensor, masks: BoolTensor) -> Tensor:
    vertices[vertices.isnan()] = 0
    masked = (vertices*masks[:,:,None,None,None]).reshape(vertices.shape[0], -1)
    norms, _ = masked.abs().max(dim=-1)
    return vertices/norms[:,None,None,None,None], norms

########################################################################################################################

def make_dense_layers(layer_sizes: Sequence[int], activation: nn.Module = nn.ELU, final_activation: bool = False, **kwargs) -> List[Module]:

    layers = []

    for size_in, size_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        layers.append(nn.Linear(size_in, size_out, **kwargs))
        layers.append(activation())

    return layers[:-1] if not final_activation else layers

def make_conv_layers(layer_sizes: Sequence[int], activation: nn.Module = nn.ELU, final_activation: bool = False, **kwargs) -> List[Module]:

    conv_layers = []

    for in_channels, out_channels, kernel_size, stride in layer_sizes:
        conv_layers.append(nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, **kwargs))
        conv_layers.append(activation())

    return conv_layers[:-1] if not final_activation else conv_layers

def count_params(model) -> int:
    return sum( map(torch.numel, model.parameters()) )

########################################################################################################################

def checkpoint(
    model: Union[Module, dict],
    optim: Optional[Union[Optimizer, dict]] = None,
    epoch: int = 0,
    losses: Optional[Tensor] = None,
    norms: Optional[Tensor] = None,
    path: str = 'pnode_checkpoint.pt',
    **rest):

    model_sd = model.state_dict() if not isinstance(model, dict) else model

    data = {'pnode_state_dict': model_sd, 'epoch': epoch}

    if optim is not None:
        optim_sd = optim.state_dict() if not isinstance(optim, dict) else optim
        data['optim_state_dict'] = optim_sd
    
    if losses is not None:
        data['losses'] = losses.detach().cpu()
        
    if norms is not None:
        data['norms'] = norms.detach().cpu()
    
    torch.save(dict(data, **rest), path)

def load_checkpoint(path: str, model: Module, optim: Optional[Optimizer] = None):

    device = next(model.parameters()).device
    data = torch.load(path, map_location=device)

    model.load_state_dict(data['pnode_state_dict'])

    if 'optim_state_dict' in data and optim is not None:
        optim.load_state_dict(data['optim_state_dict'])

    epoch = data['epoch']
    losses = data['losses'] if 'losses' in data else torch.Tensor([])

    return model, optim, epoch, losses 