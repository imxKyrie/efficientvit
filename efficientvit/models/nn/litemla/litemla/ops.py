import torch
import os
import re
import pathlib


def find_dso():
    root = pathlib.Path(__file__).parent
    build_tree_dir = root.parent / "build"
    site_dir = root.parent
    search_paths = [
        build_tree_dir,
        site_dir,
    ]
    while search_paths:
        for dirname, subdirs, filenames in os.walk(search_paths.pop()):
            for filename in filenames:
                if re.match(r"litemla_ops_cuda\..*so", filename):
                    return pathlib.Path(dirname) / filename
            search_paths.extend(subdirs)


_DSO_PATH = find_dso()

if _DSO_PATH is None:
    raise ImportError("Cannot find compiled extension")
else:
    torch.ops.load_library(_DSO_PATH)

litemla_attn = torch.ops.efficientvit.litemla_attn

def litemla_attn_reference(q, k, v, stages=3, eps=1e-5) -> torch.Tensor:
    eps = 1e-5
    # TODO: fuse ReLU into kernel
    q = torch.relu(q)
    k = torch.relu(k)

    # linear matmul
    trans_k = k.transpose(-1, -2)

    v = torch.nn.functional.pad(v, (0, 0, 0, 1), mode="constant", value=1)
    out = torch.matmul(v, trans_k)
    out = torch.matmul(out, q)
    if out.dtype in [torch.float16, torch.bfloat16]:
        out = out.float()
    out = out[:, :, :-1] / (out[:, :, -1:] + eps)
    return out.to(q.dtype)

torch.library.register_kernel("efficientvit::litemla_attn", "default")