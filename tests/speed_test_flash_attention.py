import time
import minitorch
from minitorch.cuda_kernel_ops import CudaKernelOps

import numpy as np
import torch
import pandas as pd
import cProfile, pstats

def speed_test_multihead_attention_flash_attention(
    batch_size, queries_len, n_embd, num_heads, p_dropout, backend, causal=False, repeat=1
):
    np.random.seed(10)
    torch.manual_seed(10)

    data = np.random.rand(batch_size, queries_len, n_embd)
    X = minitorch.tensor_from_numpy(data, backend, True)
    X_ = torch.tensor(data, dtype=torch.float32, requires_grad=True)

    layer_ = torch.nn.MultiheadAttention(
        n_embd, num_heads, p_dropout, bias=False, batch_first=True, dtype=torch.float32
    )

    flashattn = minitorch.MultiHeadAttention(
        n_embd, num_heads, causal, p_dropout, bias=False, backend=backend, use_fused_kernel=False, use_flash_attention=True
    )

    layer = minitorch.MultiHeadAttention(
        n_embd, num_heads, causal, p_dropout, bias=False, backend=backend, use_fused_kernel=False, use_flash_attention=False
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

    flashattn.q_projection.weights.value = w_q
    flashattn.k_projection.weights.value = w_k
    flashattn.v_projection.weights.value = w_v
    flashattn.out_projection.weights.value = w_out

    M = torch.triu(-float("inf") * torch.ones(queries_len, queries_len), 1)

    result_, _ = layer_(X_, X_, X_, attn_mask=M) if causal else layer_(X_, X_, X_)

    # Measure execution time for minitorch implementation
    start_time = time.time()
    for _ in range(repeat):
        result = layer(X)
    minitorch_time = time.time() - start_time


    # Measure execution time for flashattn implementation
    start_time = time.time()
    for _ in range(repeat):
        flashattn_result = flashattn(X)
    flashattn_time = time.time() - start_time

    # Validate correctness
    np.testing.assert_allclose(
        flashattn_result.to_numpy(), result_.detach().numpy(), atol=1e-5, rtol=1e-5
    )

    np.testing.assert_allclose(
        result.to_numpy(), result_.detach().numpy(), atol=1e-5, rtol=1e-5
    )

    # print(f"Minitorch | FlashAttention (Fused Kernel + Tiling): {minitorch_time:.4f} seconds | {flashattn_time:.4f} seconds")
    print(f"batch: {batch_size}, N: {queries_len} n_embd: {n_embd} nh: {num_heads} | {minitorch_time:.4f} seconds | {flashattn_time:.4f} seconds")

    return minitorch_time / repeat, flashattn_time / repeat

def profile(batch_size, queries_len, n_embd, num_heads, p_dropout, backend):

    np.random.seed(10)
    torch.manual_seed(10)

    data = np.random.rand(batch_size, queries_len, n_embd)
    X = minitorch.tensor_from_numpy(data, backend, True)
    X_ = torch.tensor(data, dtype=torch.float32, requires_grad=True)

    layer_ = torch.nn.MultiheadAttention(
        n_embd, num_heads, p_dropout, bias=False, batch_first=True, dtype=torch.float32
    )

    flashattn = minitorch.MultiHeadAttention(
        n_embd, num_heads, False, p_dropout, bias=False, backend=backend, use_fused_kernel=False, use_flash_attention=True
    )

    layer = minitorch.MultiHeadAttention(
        n_embd, num_heads, False, p_dropout, bias=False, backend=backend, use_fused_kernel=False
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

    flashattn.q_projection.weights.value = w_q
    flashattn.k_projection.weights.value = w_k
    flashattn.v_projection.weights.value = w_v
    flashattn.out_projection.weights.value = w_out


    profiler = cProfile.Profile()
    profiler.enable()
    #####
    result = layer(X)
    #####
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()

    profiler = cProfile.Profile()
    profiler.enable()
    #####
    result = flashattn(X)
    #####
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()


def main():
    batch_size = [2**i for i in range(6, 7)]
    queries_len = [2**i for i in range(7, 13)]
    n_embd = [2**i for i in range(6, 12)]
    num_heads = [2**i for i in range(1, 5)]
    p_dropout = 0.0
    backend = minitorch.TensorBackend(CudaKernelOps)

    df = pd.DataFrame(columns=['batch_size', 'N', 'd', 'nh', 'p_dropout', 'causal', 'original_time', 'flashattn_time'])

    # profile(batch_size[-1], queries_len[-1], n_embd[-1], num_heads[-1], p_dropout, backend)

    for batch in batch_size:
        for N in queries_len:
            for d in n_embd:
                for nh in num_heads:
                    minitorch_time, flashattn_time = speed_test_multihead_attention_flash_attention(
                        batch, N, d, nh, p_dropout, backend, causal=False
                    )

                    df = pd.concat([pd.DataFrame([[batch, N, d, nh, p_dropout, False, minitorch_time, flashattn_time]], columns=df.columns), df.dropna(axis=1, how="all")], ignore_index=True)

                    minitorch_time, flashattn_time = speed_test_multihead_attention_flash_attention(
                        batch, N, d, nh, p_dropout, backend, causal=True
                    )

                    df = pd.concat([pd.DataFrame([[batch, N, d, nh, p_dropout, True, minitorch_time, flashattn_time]], columns=df.columns), df.dropna(axis=1, how="all")], ignore_index=True)
    
            df.to_csv('../speed_test.csv', index=False)

if __name__ == "__main__":
    main()