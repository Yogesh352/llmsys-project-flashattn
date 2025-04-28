import pytest
import minitorch
from minitorch.cuda_kernel_ops import CudaKernelOps
import math
from torch.nn import functional as F

import numpy as np
import torch
import torch.nn as nn
import numba

np.random.seed(3)

datatype = np.float32

_BACKENDS = [
    pytest.param(
        minitorch.TensorBackend(CudaKernelOps),
        marks=pytest.mark.skipif(not numba.cuda.is_available(), reason="No GPU"),
    )
]


@pytest.mark.parametrize("batch_size", [2**i for i in range(6, 7)])
@pytest.mark.parametrize("queries_len", [2**i for i in range(7, 13)])
@pytest.mark.parametrize("n_embd", [2**i for i in range(6, 12)])
@pytest.mark.parametrize("num_heads", [2**i for i in range(1, 5)])
@pytest.mark.parametrize("p_dropout", [0.0])
@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_multihead_attention_flash_attention(
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
        n_embd,
        num_heads,
        False,
        p_dropout,
        bias=False,
        backend=backend,
        use_fused_kernel=False,
        use_flash_attention=False,
    )

    # Set weights of minitorch layer to torch weights
    w_qkv = layer_.in_proj_weight.detach().numpy().T.copy()
    w_q_, w_k_, w_v_ = [
        w.copy() for w in np.split(w_qkv, 3, -1)
    ]
    w_out_ = layer_.out_proj.weight.detach().numpy().T.copy()

    w_q = minitorch.tensor_from_numpy((w_q_), backend=backend, requires_grad=True)
    w_k = minitorch.tensor_from_numpy((w_k_), backend=backend, requires_grad=True)
    w_v = minitorch.tensor_from_numpy((w_v_), backend=backend, requires_grad=True)
    w_out = minitorch.tensor_from_numpy((w_out_), backend=backend, requires_grad=True)

    layer.q_projection.weights.value = w_q
    layer.k_projection.weights.value = w_k
    layer.v_projection.weights.value = w_v
    layer.out_projection.weights.value = w_out

    result = layer(X)
    result_, _ = layer_(X_, X_, X_)

    np.testing.assert_allclose(
        result.to_numpy(), result_.detach().numpy(), atol=1e-5, rtol=1e-5
    )

    # Check backward
    result.sum().backward()
    result_.sum().backward()

    np.testing.assert_allclose(
        X.grad.to_numpy(), X_.grad.detach().numpy(), atol=1e-5, rtol=1e-5
    )

    np.testing.assert_allclose(
        layer.out_projection.weights.value.grad.to_numpy(),
        layer_.out_proj.weight.grad.detach().numpy().T,
        atol=1e-5,
        rtol=1e-5,
    )

    # Since the torch W_Q, W_K, W_V is all one matrix, we can't compare
    assert (
        (layer.q_projection.weights.value.grad is not None)
        and (layer.k_projection.weights.value.grad is not None)
        and (layer.v_projection.weights.value.grad is not None)
    )



@pytest.mark.parametrize("batch_size", [2**i for i in range(6, 7)])
@pytest.mark.parametrize("queries_len", [2**i for i in range(11, 13)])
@pytest.mark.parametrize("n_embd", [2**i for i in range(6, 12)])
@pytest.mark.parametrize("num_heads", [2**i for i in range(1, 5)])
@pytest.mark.parametrize("p_dropout", [0.0])
@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_multihead_attention_flash_attention_is_causal(
    batch_size, queries_len, n_embd, num_heads, p_dropout, backend
):
    np.random.seed(10)
    torch.manual_seed(10)

    data = np.random.rand(batch_size, queries_len, n_embd)
    X = minitorch.tensor_from_numpy(data, backend, True)
    X_ = torch.tensor(data, dtype=torch.float32, requires_grad=True)

    layer_ = torch.nn.MultiheadAttention(
        n_embd,
        num_heads,
        p_dropout,
        bias=False,
        batch_first=True,
        dtype=torch.float32
    )

    layer = minitorch.MultiHeadAttention(
        n_embd,
        num_heads,
        True,
        p_dropout,
        bias=False,
        backend=backend,
        use_fused_kernel=False,
        use_flash_attention=True,
    )

    # Set weights of minitorch layer to torch weights
    w_qkv = layer_.in_proj_weight.detach().numpy().T.copy()  
    w_q_, w_k_, w_v_ = [
        w.copy() for w in np.split(w_qkv, 3, -1)
    ] 
    w_out_ = layer_.out_proj.weight.detach().numpy().T.copy()

    w_q = minitorch.tensor_from_numpy((w_q_), backend=backend, requires_grad=True)
    w_k = minitorch.tensor_from_numpy((w_k_), backend=backend, requires_grad=True)
    w_v = minitorch.tensor_from_numpy((w_v_), backend=backend, requires_grad=True)
    w_out = minitorch.tensor_from_numpy((w_out_), backend=backend, requires_grad=True)

    layer.q_projection.weights.value = w_q
    layer.k_projection.weights.value = w_k
    layer.v_projection.weights.value = w_v
    layer.out_projection.weights.value = w_out

    # Casual mask
    M = torch.triu(-float("inf") * torch.ones(queries_len, queries_len), 1)

    result = layer(X)
    result_, _ = layer_(X_, X_, X_, attn_mask=M)

    np.testing.assert_allclose(
        result.to_numpy(), result_.detach().numpy(), atol=1e-5, rtol=1e-5
    )

    # Check backward
    result.sum().backward()
    result_.sum().backward()

    np.testing.assert_allclose(
        X.grad.to_numpy(), X_.grad.detach().numpy(), atol=1e-5, rtol=1e-5
    )

    np.testing.assert_allclose(
        layer.out_projection.weights.value.grad.to_numpy(),
        layer_.out_proj.weight.grad.detach().numpy().T,
        atol=1e-5,
        rtol=1e-5,
    )

    # Since the torch W_Q, W_K, W_V is all one matrix, we can't compare
    assert (
        (layer.q_projection.weights.value.grad is not None)
        and (layer.k_projection.weights.value.grad is not None)
        and (layer.v_projection.weights.value.grad is not None)
    )
    