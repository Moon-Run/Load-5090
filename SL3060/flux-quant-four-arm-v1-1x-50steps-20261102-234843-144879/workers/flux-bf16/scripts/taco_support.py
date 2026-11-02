"""Mixed-wire forward reference; backward remains the common BF16 approximation."""
import json
from pathlib import Path
import torch
import torch.distributed as dist
from reference import roundtrip
from routing import is_native, is_fused_v2, is_flux_tensor, selection_entry, definition

def masked_roundtrip(piece, mask, src, dst, m, n, world):
    coded, _ = roundtrip(piece)
    if mask is None:
        return coded
    tm, nt = (m // 128, (n + 127) // 128)
    selected = mask[src].reshape(tm, nt)[dst * (tm // world):(dst + 1) * (tm // world)]
    selected = selected.repeat_interleave(128, 0).repeat_interleave(128, 1)[:, :n].to(piece.device).bool()
    return torch.where(selected, coded, piece)

class TacoForward:

    def __init__(self, group, m, n, k_local, policy, role):
        if is_native(policy):
            raise ValueError('Native scope cannot construct a Flux candidate')
        if is_flux_tensor(policy):
            from flux_tensor_taco import FluxTensorTaco
            self.role, self.plan_sha256 = (role, None)
            self.codec = FluxTensorTaco(group, m, n, k_local)
            self.owns_output = True
            self.workspace_info = self.codec.workspace_info
            return
        raise RuntimeError('Basic worker only supports the local GemmOnly codec')

    def forward(self, input_, weight, reduce_scatter_option=None):
        if self.owns_output:
            return self.codec.forward(input_, weight)
        return self.codec.forward(input_.detach(), weight.detach())

@torch.no_grad()
def check_taco_output(partial, actual, group, mask=None):
    if mask is not None:
        raise ValueError('Unquantized Flux must not receive a mask')
    reference = torch.empty_like(actual)
    dist.reduce_scatter_tensor(reference, partial.contiguous(), group=group)
    denominator = max(reference.float().norm().item(), 1e-20)
    delta = actual.float() - reference.float()
    relative = delta.norm().item()/denominator
    error = dict(max_abs=delta.abs().max().item(), relative_l2=relative,
                 protocol_relative_l2=relative, budget_vs_bf16=.05, budget_vs_codec=.01,
                 reference='unquantized_torch_matmul_nccl_reduce_scatter', quantization='none')
    error['passed'] = bool(torch.isfinite(actual).all()) and relative <= .01
    assert error['passed'], error
    return error
