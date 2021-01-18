"""
An adaptation over the continuous attention module to tackle seq2seq models.
"""

import torch
from torch import nn

from quati.modules.continuous_sparsemax import ContinuousSparsemax
from quati.modules.continuous_softmax import ContinuousSoftmax
from quati.modules.basis_functions import (PowerBasisFunctions,
                                           SineBasisFunctions,
                                           CosineBasisFunctions,
                                           GaussianBasisFunctions)


def add_power_basis_functions(min_d=0, max_d=2, device=None):
    degrees = torch.arange(min_d, max_d + 1, device=device).float().to(device)
    return PowerBasisFunctions(degrees)


def add_wave_basis_functions(nb_waves, wave_b, max_seq_len, device=None):
    # sin/cos basis functions similar to Transformers' positional embeddings
    dims = torch.arange(nb_waves // 2, device=device).float()
    omegas = max_seq_len * 1. / (wave_b ** (2 * dims / nb_waves)).to(device)
    return SineBasisFunctions(omegas), CosineBasisFunctions(omegas)


def add_gaussian_basis_functions(nb_basis, sigmas, device=None):
    mu, sigma = torch.meshgrid(torch.linspace(0, 1, nb_basis // len(sigmas)),
                               torch.Tensor(sigmas))
    mus = mu.flatten().to(device)
    sigmas = sigma.flatten().to(device)
    return GaussianBasisFunctions(mus, sigmas)


class ContinuousAttention(nn.Module):
    """Generic ContinuousAttention implementation based on ContinuousSparsemax.

       1. Use `query` and `keys` to compute scores (via an encoder)
       2. Map to a probability distribution
       3. Get the final context vector

    TODO: remove gpu_id from here. Compute everything in cpu and register the
        used tensors as buffers, so we don't need to worry about devices. This
        will require a different strategy for the dynamic_nb_basis option when
        the max_seq_len is very large.

    Args:
        encoder (ContinuousEncoder): the encoder for getting `mu` and `sigma_sq`
        dropout (float): dropout rate (default: 0)
        nb_waves (int): number of sine and cosine waves (default: 16)
        freq (int): frequency param for sine and cosine waves (default: 10000)
        max_seq_len (int): hypothetical maximum sequence length (default: 3000)
        use_power_basis (bool): whether to use power basis functions
        use_wave_basis (bool): whether to use sine/cosine basis functions
        use_gaussian_basis (bool): whether to use gaussian basis functions
        dynamic_nb_basis (bool): whether to use a dynamic nb of basis functions
            where nb_basis = seq_len. If True, the offline computations will be
            saved in cpu memory, and therefore it will impact the runtime
            performance due to memory transfer between cpu and gpu
        consider_pad (bool): whether to consider "pad" positions and insert safe
            margins into the computation of the value function.
        max_activation (str): which prob. density mapping to use:
            sparsemax (default) or softmax (works only with gaussians for now)
        gpu_id (int): gpu id (default: None)
        seq_lens (Iterable): a set containing the sequence lengths of your data.
            Useful to reduce memory usage (will not impact runtime performance).
    """

    def __init__(
        self,
        encoder,
        dropout=0.,
        nb_waves=16,
        wave_b=10000,
        max_seq_len=3000,
        use_power_basis=False,
        use_wave_basis=False,
        use_gaussian_basis=True,
        dynamic_nb_basis=False,
        consider_pad=False,
        max_activation='sparsemax',
        gpu_id=None,
        seq_lens=None
    ):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(p=dropout)
        self.nb_waves = nb_waves
        self.wave_b = wave_b
        self.max_seq_len = max_seq_len
        self.use_power_basis = use_power_basis
        self.use_wave_basis = use_wave_basis
        self.use_gaussian_basis = use_gaussian_basis
        self.dynamic_nb_basis = dynamic_nb_basis
        self.consider_pad = consider_pad
        self.gpu_id = gpu_id

        if not any([use_gaussian_basis, use_power_basis, use_wave_basis]):
            raise Exception('You should use at least one basis function.')

        # stored variables (useful for later)
        self.mu = None
        self.sigma_sq = None

        # use basis functions in `psi` to define continuous transformation
        # psi = None for now
        if max_activation == 'sparsemax':
            self.continuous_max_activation = ContinuousSparsemax(psi=None)
        elif max_activation == 'softmax':
            self.continuous_max_activation = ContinuousSoftmax(psi=None)

        # compute G offline for each length up to `max_seq_len`
        self.Gs = [] if seq_lens is None else {}
        self.psis = [] if seq_lens is None else {}
        lens = range(1, self.max_seq_len + 1) if seq_lens is None else seq_lens
        for length in lens:
            # get the basis functions for this length
            psi = self.create_psi(length)

            if self.consider_pad and length > 1:
                # insert positions before 0 and after 1 as safe margins for
                # "pad" values (cases where the supp goes beyond [0, 1])
                pad_margin = .5
                if length % 2:
                    shift = 1. / length
                    positions = torch.linspace(0 - pad_margin + shift,
                                               1 + pad_margin - shift,
                                               2 * length - 1)
                else:
                    shift = 1. / 2 * length
                    positions = torch.linspace(0 - pad_margin + shift,
                                               1 + pad_margin - shift,
                                               2 * length)
            else:
                shift = 1 / float(2 * length)
                positions = torch.linspace(shift, 1 - shift, length)

            positions = positions.unsqueeze(1).to(self.gpu_id)

            # stack basis functions for each interval
            all_basis = [basis_function.evaluate(positions)
                         for basis_function in psi]
            F = torch.cat(all_basis, dim=-1).t().to(self.gpu_id)
            nb_basis = sum([len(b) for b in psi])
            assert F.size(0) == nb_basis

            # compute G with a ridge penalty
            penalty = 0.01
            I = torch.eye(nb_basis).to(self.gpu_id)
            G = F.t().matmul((F.matmul(F.t()) + penalty * I).inverse())

            # filter out rows associated with "pad" positions
            if self.consider_pad and length > 1:
                if length % 2:
                    G = G[((length - 1) // 2):(-(length - 1) // 2), :]
                else:
                    G = G[(length // 2):-(length // 2), :]
            assert G.size(0) == length

            if isinstance(self.psis, dict):
                self.psis[length-1] = psi
                self.Gs[length-1] = G
            else:
                self.psis.append(psi)
                if self.dynamic_nb_basis:
                    self.Gs.append(G.cpu())  # corner case for large sequences
                else:
                    self.Gs.append(G)

    def create_psi(self, length):
        psi = []
        if self.use_power_basis:
            psi.append(
                add_power_basis_functions(min_d=0, max_d=2, device=self.gpu_id)
            )
        if self.use_wave_basis:
            nb_waves = length if self.dynamic_nb_basis else self.nb_waves
            nb_waves = max(2, nb_waves)
            psi.extend(
                add_wave_basis_functions(nb_waves,
                                         self.wave_b,
                                         self.max_seq_len,
                                         device=self.gpu_id)
            )
        if self.use_gaussian_basis:
            nb_waves = length if self.dynamic_nb_basis else self.nb_waves
            nb_waves = max(2, nb_waves)
            psi.append(
                add_gaussian_basis_functions(nb_waves,
                                             sigmas=[.1, .5],
                                             # sigmas=[.03, .1, .3],
                                             device=self.gpu_id)
            )
        return psi

    def value_function(self, values, mask=None):
        # Approximate B * F = values via multivariate regression.
        # Use a ridge penalty. The solution is B = values * G
        seq_len = values.size(1)
        G = self.Gs[seq_len - 1].to(values.device)
        B = values.transpose(-1, -2).matmul(G)
        return B

    def score_function(self, query, keys, mask=None):
        self.mu, self.sigma_sq, disc_p_attn = self.encoder(
            query, keys, mask=mask
        )
        self.sigma_sq = torch.clamp(self.sigma_sq, min=1e-6)
        # concat time dimension with batch dimension to exploit the parallelism
        # in the ContinuousSparsemax computation
        self.mu = self.mu.view(-1)
        self.sigma_sq = self.sigma_sq.view(-1)
        theta = torch.zeros(self.mu.size(0), 2, device=query.device)
        theta[:, 0] = self.mu / self.sigma_sq
        theta[:, 1] = -1. / (2. * self.sigma_sq)
        return theta, disc_p_attn

    def forward(self, query, keys, values, mask=None):
        """
        Compute attention vector. Legend:
        bs: batch size
        t_ts = target time steps
        s_ts = source time steps
        hdim = hidden dimensionality

        Args:
            query (torch.Tensor): shape of (bs, t_ts, hdim)
            keys (torch.Tensor): shape of (bs, s_ts, hdim)
            values (torch.Tensor): shape of (bs, s_ts, hdim)
            mask (torch.ByteTensor): shape of (bs, s_ts)

        Returns:
            c: torch.Tensor with shape of (bs, t_ts, hdim)
            r: torch.Tensor with shape of (bs, t_ts, nb_basis)
        """
        batch_size = keys.size(0)
        seq_len = keys.size(1)

        # make up a dummy mask
        if mask is None:
            mask = torch.ones(batch_size, seq_len, device=query.device)

        # get `mu` and `sigma` as the canonical parameters `theta`
        # (bs, ts, hdim) -> (bs, 2)
        theta, disc_p_attn = self.score_function(query, keys, mask=mask)

        # map to a probability density over basis functions
        # (bs, 2) -> (bs, nb_basis)
        self.continuous_max_activation.psi = self.psis[seq_len - 1]
        r = self.continuous_max_activation(theta)

        # create a time dimension
        # (bs, nb_basis) -> (bs, 1, nb_basis)
        r = r.unsqueeze(1)

        # apply dropout (default:0 - like in Transformer arch)
        r = self.dropout(r)

        # compute B using a multivariate regression
        # (bs, ts, hdim) -> (bs, hdim, nb_basis)
        B = self.value_function(values, mask=mask)

        # same B for all query vectors
        B = B.repeat(query.shape[1], 1, 1)

        # (bs, hdim, nb_basis) * (bs, nb_basis, 1) -> (bs, hdim, 1)
        # get the context vector
        c = torch.matmul(B, r.transpose(-1, -2))

        # put time dimension back in the correct place
        # (bs, hdim, 1) -> (bs, 1, hdim)
        c = c.transpose(-1, -2)

        # in case attention probabilities from a discrete attention is passed
        if disc_p_attn is not None:
            # compute discrete context vector
            disc_c = torch.matmul(disc_p_attn, values)
            # merge with continuous context vector
            c = c + disc_c

        # split c and r for each query
        c = c.view(query.size(0), query.size(1), -1)
        r = r.view(query.size(0), query.size(1), -1)

        return c, r
