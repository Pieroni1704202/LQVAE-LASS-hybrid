import functools
from typing import Any, Optional, Tuple
import numpy as np
import sparse
from interfaces import Likelihood
import torch

class SparseLikelihood(Likelihood):
    def __init__(self, sum_dist_path: str, device: torch.device, lambda_coeff: float = 1.0):
        self._device = torch.device(device)
        self._lambda_coeff = lambda_coeff
        self._freqs = self._normalize_matrix(sum_dist_path)

    def get_device(self) -> torch.device:
        return self._device

    def get_tokens_count(self) -> int:
        return self._freqs.shape[0]

    @functools.lru_cache(512)
    def get_log_likelihood(self, token_idx: int) -> Tuple[torch.LongTensor, torch.Tensor]:
        sparse_nll = self._freqs[:, :, token_idx].tocoo()
        nll_coords = torch.tensor(sparse_nll.coords, device=self.get_device(), dtype=torch.long)
        nll_data = torch.tensor(sparse_nll.data, device=self.get_device(), dtype=torch.float)
        return nll_coords, nll_data

    def _normalize_matrix(self, sum_dist_path: str):
        sum_dist = sparse.load_npz(str(sum_dist_path))
        integrals = sum_dist.sum(axis=-1, keepdims=True)
        I, J, _ = integrals.coords
        integrals = sparse.COO(
            integrals.coords, integrals.data, shape=integrals.shape, fill_value=1
        )
        log_data = np.log(sum_dist / integrals) * self._lambda_coeff
        return sparse.GCXS.from_coo(log_data, compressed_axes=[2])

