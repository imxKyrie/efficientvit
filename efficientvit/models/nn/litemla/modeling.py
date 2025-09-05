import functools
import torch
import triton
import litemla.ops as litemla_ops


@torch.library.custom_op("evit::litemla", mutates_args=())
def litemla(qkv: torch.Tensor, dim: int, eps: float) -> torch.Tensor:
    B, _, H, W = qkv.size()

    if qkv.dtype != torch.float32:
        qkv = qkv.float()

    qkv = torch.reshape(
        qkv,
        (
            B,
            -1,
            3 * dim,
            H * W,
        ),
    )
    q, k, v = (
        qkv[:, :, 0:dim],
        qkv[:, :, dim : 2 * dim],
        qkv[:, :, 2 * dim :],
    )
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    return litemla_ops.litemla_attn(q, k, v, stages=3, eps=eps)

    # # # lightweight linear attention
    # q = torch.relu(q)
    # k = torch.relu(k)

    # # linear matmul
    # trans_k = k.transpose(-1, -2)

    # v = torch.nn.functional.pad(v, (0, 0, 0, 1), mode="constant", value=1)
    # vk = torch.matmul(v, trans_k)
    # out = torch.matmul(vk, q)
    # if out.dtype == torch.bfloat16:
    #     out = out.float()
    # out = out[:, :, :-1] / (out[:, :, -1:] + eps)

    # out = torch.reshape(out, (B, -1, H, W))
    # return out.to(qkv.dtype)


@torch.library.register_fake("evit::litemla")
def _litemla_fake(qkv: torch.Tensor, dim: int, eps: float) -> torch.Tensor:
    B, QKV, H, W = qkv.size()
    return torch.empty(size=(B, QKV // 3, H, W), dtype=qkv.dtype, device=qkv.device)


def efficientvit_backbone_b1():
    from efficientvit.models.efficientvit import efficientvit_backbone_b1
    from efficientvit.models.nn import LiteMLA

    def litemla_forward(self: LiteMLA, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))
        qkv = torch.cat(multi_scale_qkv, dim=1)
        out = litemla(qkv, self.dim, self.eps)
        out = self.proj(out)
        return out

    mod = efficientvit_backbone_b1()
    for child in list(mod.modules()):
        if isinstance(child, LiteMLA):
            child.forward = functools.partial(litemla_forward, child)
    return mod


if __name__ == "__main__":
    mod = efficientvit_backbone_b1().eval().cuda()
    from torch.fx.experimental.optimization import fuse

    with torch.inference_mode():
        mod = fuse(mod).half().to(memory_format=torch.channels_last)
        x = torch.rand(3, 3, 512, 768, dtype=torch.float16, device="cuda")
        x = x.to(memory_format=torch.channels_last)
        compiled = torch.compile(mod, fullgraph=True, mode="max-autotune")
        compiled(x)
        # breakpoint()
        duration_ms = triton.testing.do_bench(lambda: compiled(x))
        print(f"Duration: {duration_ms:.3f}ms")
        # breakpoint()
