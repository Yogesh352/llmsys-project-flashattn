import time
import pytest
import minitorch
from minitorch.cuda_kernel_ops import CudaKernelOps
import math
from torch.nn import functional as F

import numpy as np
import torch
import torch.nn as nn
import numba
import json
import pandas as pd

def speed_test_multihead_attention_flash_attention(
    batch_size, queries_len, n_embd, num_heads, p_dropout, backend
):
    np.random.seed(10)
    torch.manual_seed(10)

    data = np.random.rand(batch_size, queries_len, n_embd)
    X = minitorch.tensor_from_numpy(data, backend, True)
    X_ = torch.tensor(data, dtype=torch.float32, requires_grad=True)

    layer_ = torch.nn.MultiheadAttention(
        n_embd, num_heads, p_dropout, bias=False, batch_first=True, dtype=torch.float32
    )

    layer = minitorch.MultiHeadAttention(
        n_embd, num_heads, False, p_dropout, bias=False, backend=backend, use_fused_kernel=False, use_flash_attention=True
    )

    # Set weights of minitorch layer to torch weights
    w_qkv = layer_.in_proj_weight.detach().numpy().T.copy()  # (n_embd, 3*n_embd)
    w_q_, w_k_, w_v_ = [
        w.copy() for w in np.split(w_qkv, 3, -1)
    ]  # 3 * (n_embd, n_embd)
    w_out_ = layer_.out_proj.weight.detach().numpy().T.copy()

    w_q = minitorch.tensor_from_numpy((w_q_), backend=backend, requires_grad=True)
    w_k = minitorch.tensor_from_numpy((w_k_), backend=backend, requires_grad=True)
    w_v = minitorch.tensor_from_numpy((w_v_), backend=backend, requires_grad=True)
    w_out = minitorch.tensor_from_numpy((w_out_), backend=backend, requires_grad=True)

    layer.q_projection.weights.value = w_q
    layer.k_projection.weights.value = w_k
    layer.v_projection.weights.value = w_v
    layer.out_projection.weights.value = w_out

    # Measure execution time and memory utilisation for minitorch implementation
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    start_time = time.time()
    result = layer(X)
    minitorch_time = time.time() - start_time

    minitorch_memory = torch.cuda.max_memory_allocated() / 1024**2


    # Measure execution time and memory utilisation for flashattn implementation
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    start_time = time.time()
    result_, _ = layer_(X_, X_, X_)
    flashattn_time = time.time() - start_time

    flashattn_memory = torch.cuda.max_memory_allocated() / 1024**2

    # Validate correctness
    np.testing.assert_allclose(
        result.to_numpy(), result_.detach().numpy(), atol=1e-5, rtol=1e-5
    )

    print(f"Minitorch | FlashAttention (Fused Kernel + Tiling): {minitorch_time:.4f} seconds | {flashattn_time:.4f} seconds | {minitorch_memory:.6f} GB | {flashattn_memory:.6f} GB")

    return minitorch_time, flashattn_time, minitorch_memory, flashattn_memory

def main():
    batch_size = [2**i for i in range(7)]
    queries_len = [2**i for i in range(5, 14)]
    n_embd = [2**i for i in range(6, 9)]
    num_heads = [2**i for i in range(1)]
    p_dropout = 0.0
    backend = minitorch.TensorBackend(CudaKernelOps)

    df = pd.DataFrame(columns=['batch_size', 'N', 'd', 'nh', 'p_dropout', 'original_time', 'flashattn_time', 'original_memory', 'flashattn_memory'])

    for batch in batch_size:
        for N in queries_len:
            for d in n_embd:
                for nh in num_heads:
                    minitorch_time, flashattn_time, minitorch_memory, flashattn_memory = speed_test_multihead_attention_flash_attention(
                        batch, N, d, nh, p_dropout, backend
                    )

                    df = pd.concat([pd.DataFrame([[batch, N, d, nh, p_dropout, minitorch_time, flashattn_time, minitorch_memory, flashattn_memory]], columns=df.columns), df.dropna(axis=1, how="all")], ignore_index=True)
    
    df.to_csv('../speed_test.csv', index=False)

if __name__ == "__main__":
    main()