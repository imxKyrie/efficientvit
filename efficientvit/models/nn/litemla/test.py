import torch
import torch.nn.functional as F
import pytest
import triton

import litemla.ops

B, H, D, S = 3, 24, 32, 368
torch.manual_seed(0)
EPS = 1e-5
q = torch.randn(B, H, D, S, dtype=torch.half, device="cuda")
k = torch.randn(B, H, D, S, dtype=torch.half, device="cuda")
v = torch.randn(B, H, D, S, dtype=torch.half, device="cuda")
for stages in range(3, 7):
    t = triton.testing.do_bench(lambda: litemla.ops.litemla_attn(q, k, v, stages=stages, eps=EPS), quantiles=[0.5])
    print(f"cuda: ", t)
t = triton.testing.do_bench(lambda: litemla.ops.litemla_attn_reference(q, k, v, eps=EPS), quantiles=[0.5])
print(f"torch: ", t)

B, H, D, S = 3, 12, 32, 1472
torch.manual_seed(0)
EPS = 1e-5
q = torch.randn(B, H, D, S, dtype=torch.half, device="cuda")
k = torch.randn(B, H, D, S, dtype=torch.half, device="cuda")
v = torch.randn(B, H, D, S, dtype=torch.half, device="cuda")
for i in range(3, 7):
    t = triton.testing.do_bench(lambda: litemla.ops.litemla_attn(q, k, v, stages=i, eps=EPS), quantiles=[0.5])
    print(f"cuda: ", t)
t = triton.testing.do_bench(lambda: litemla.ops.litemla_attn_reference(q, k, v, eps=EPS), quantiles=[0.5])
print(f"torch: ", t)


@pytest.mark.parametrize("stages", [3, 4, 5, 6])
@pytest.mark.parametrize(
    "batch_size, num_heads, seq_len, head_dim",
    [
        (3, 24, 368, 32),
        (3, 12, 1472, 32),
    ],
)
def test_fused_litemla(batch_size, num_heads, seq_len, head_dim, stages):
    B, H, D, S = batch_size, num_heads, head_dim, seq_len
    # torch.manual_seed(0)
    EPS = 1e-5
    # TODO: support (B, H, S, D) layout
    q = torch.randn(B, H, D, S, dtype=torch.half, device="cuda")
    k = torch.randn(B, H, D, S, dtype=torch.half, device="cuda")
    v = torch.randn(B, H, D, S, dtype=torch.half, device="cuda")
    # TODO: fuse pad into kernel
    actual = litemla.ops.litemla_attn(q, k, v, stages, EPS)
    expected = litemla.ops.litemla_attn_reference(q, k, v, stages, EPS)
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
