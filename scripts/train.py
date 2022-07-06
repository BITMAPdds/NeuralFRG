from time import perf_counter
import psutil
import argparse
import warnings

import numpy as np
import h5py

import torch
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

import neuralfrg as nfrg

def read_and_preprocess_data(args):

    print('Loading data...', flush=True)

    with h5py.File(args.path, 'r') as file:
        
        print('Reading in vertex data...', end='', flush=True)
        vertices = torch.Tensor(np.array(file['vertices']))
        print('done')

        print('Reading in interpolation times...', end='', flush=True)
        times = torch.Tensor(np.array(file['times']))
        print('done')
        
        print('Reading in trajectory masks...', end='', flush=True)
        masks = torch.BoolTensor(np.array(file['masks']))
        print('done')
        
        print('Reading in initial couplings...', end='', flush=True)
        couplings = torch.Tensor(np.array(file['couplings']))
        print('done')

    print('Processing data...', end='', flush=True)
    
    if not args.no_normalization:

        vertices, norms = nfrg.normalize_data(vertices, masks)

        torch.save(norms, 'norms.pt')
        print(f'(normalizing data - norms saved to ./norms.pt)...', end='', flush=True)
    
    print('done', flush=True)

    return vertices, times, masks, couplings

def printout_helper(epoch: int, cost: float, clock: float, delay: float = 10.0):
    
    if perf_counter() - clock < delay:
        return clock

    else:
        avail_mem = GPU_MEM - torch.cuda.memory_reserved(0)/(1024**2)
        ram_usage = psutil.virtual_memory()[2]

        tags = [
            f'Epoch: {(epoch+1):5}',
            f'Cost: {cost:.2e}',
            f'RAM usage: {ram_usage}%',
            f'GPU memory left: {avail_mem:.2f} MB'
        ]

        msg = ' | '.join(tags)
        print(msg, flush=True)
        
        return perf_counter()
        
def main(args):

    vertices, times, masks, couplings = read_and_preprocess_data(args)

    batch_size = vertices.shape[0] if args.bsize == -1 else args.bsize
    
    if batch_size <= vertices.shape[0]:
        dset = TensorDataset(vertices, couplings, masks)
        loader = DataLoader(dset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
    else:
        raise RuntimeError(
            'Batch size must be smaller than the number of training '
            f'trajectories {vertices.shape[0]}. Got {batch_size}'
        )

    in_dim = couplings.shape[-1]
    pnode = nfrg.PNODE(
        in_dim=in_dim, ldim=args.ldim, out_dims=args.out_dims,
        encoder_layer_sizes=args.encoder_layers,
        node_layer_sizes=args.node_layers,
        decoder_dense_layer_sizes=args.decoder_dense_layers,
    )

    pnode.to(device)
    
    print('Encoder # parameters:', nfrg.count_params(pnode.encoder))
    print('Node    # parameters:', nfrg.count_params(pnode.node))
    print('Decoder # parameters:', nfrg.count_params(pnode.decoder), flush=True)

    optimizer = optim.Adam(pnode.parameters(), lr=args.lr)

    n_batches = len(loader)
    losses = torch.zeros(n_batches*args.epochs).requires_grad_(False)
    
    if args.load_path:
        print(f'Loading model weights and optimizer parametrs from {args.load_path}...', flush=True)
        pnode, optimizer, epoch, losses_ = nfrg.load_checkpoint(args.load_path, pnode, optimizer)

        losses = torch.concat([losses_, losses]).requires_grad_(False)
    else:
        epoch = 0
    
    times = times.to(device=device)
    best_loss = 1e8
    best_epoch = 0
    state_dict = pnode.state_dict()
    optim_state_dict = optimizer.state_dict()
    
    print('Starting training...', flush=True)
    clock = perf_counter()

    for n in range(epoch, epoch+args.epochs):

        epoch_losses = np.empty(n_batches, dtype=np.float32)

        for j, (v, c, m) in enumerate(loader):

            v = v.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True)
            
            v_ = pnode(c, times)
            cost = nfrg.masked_mse(v, v_, m)
            
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            losses[n*n_batches+j] = cost.detach()
            epoch_losses[j] = cost.detach()

        epoch_loss = np.mean(epoch_losses)
        clock = printout_helper(n, epoch_loss, clock)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            state_dict = pnode.state_dict()
            optim_state_dict = optimizer.state_dict()
            best_epoch = n

        if n%args.save_epochs==0 and n>0:
            nfrg.checkpoint(state_dict, optim_state_dict, best_epoch, losses)
                
    print(f'Best cost value: {best_loss}', flush=True)

    nfrg.checkpoint(state_dict, optim_state_dict, best_epoch, losses)  

if __name__=='__main__':
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('GPU:', torch.cuda.get_device_name(0))
        print('CUDNN version:', torch.backends.cudnn.version())
        
        GPU_MEM = torch.cuda.get_device_properties(0).total_memory/(1024**2)
        print(f'Available GPU memory: {GPU_MEM:.2f} MB')
        
    else:
        warnings.warn('No GPU available, using CPU. This will be slow!')
        
    parser = argparse.ArgumentParser()
    
    parser.add_argument('path', type=str, help='Data file path')
    parser.add_argument('--epochs', type=int, help='# of epochs')
    parser.add_argument('--bsize', type=int, default=-1, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--ldim', type=int, default=32, help='Latent dimension')
    parser.add_argument('--no_normalization', help="Normalize data before training", action="store_true")
    parser.add_argument('--disable_cudnn', help="Disable CUDNN during training", action="store_true")
    parser.add_argument('--benchmark_cudnn', help="Run CUDNN performance benchmarks before training", action="store_true")
    parser.add_argument('--save_epochs', type=int, default=20, help='Frequency of saving the network state, in epochs')
    parser.add_argument('--load_path', type=str, default=None, help='PNODE state_dict file path')
    parser.add_argument('--out_dims', type=int, nargs='+', default=(48,48,48), help='Shape of the output tensor')
    parser.add_argument('--encoder_layers', type=int, nargs='+', default=None, help='A sequence of encoder layer widths')
    parser.add_argument('--node_layers', type=int, nargs='+', default=None, help='A sequence of layer widths in the NODE kernel')
    parser.add_argument('--decoder_dense_layers', type=int, nargs='+', default=None, help='A sequence of decoder dense layer widths')
    
    args = parser.parse_args()
    
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
        print('CUDNN disabled', flush=True)
        
    if args.benchmark_cudnn and not args.disable_cudnn:
        torch.backends.cudnn.benchmark = True
        print('CUDNN benchmarking enabled, note that initial run might take a while', flush=True)
    
    main(args)