from pathlib import Path
import json
import torch
import torch.distributed as dist

class TacoForward:
    def __init__(self, group, m, n, k_local, policy, role):
        from flux_tensor_taco import FluxTensorTaco
        self.role, self.plan_sha256 = role, None
        self.codec = FluxTensorTaco(group, m, n, k_local)
        self.owns_output = True
        self.workspace_info = self.codec.workspace_info

    def forward(self, input_, weight, reduce_scatter_option=None):
        return self.codec.forward(input_, weight)

@torch.no_grad()
def check_taco_output(partial, actual, group, mask=None):
    if mask is not None:
        raise ValueError('No selective quantization in this experiment')
    arm = json.loads((Path(__file__).parent/'config.json').read_text())['arm']
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    bf16 = torch.empty_like(actual)
    dist.reduce_scatter_tensor(bf16, partial.contiguous(), group=group)
    if arm == 'flux-bf16':
        coded = bf16
    else:
        from megatron_taco import reference_roundtrip
        parts = [torch.empty_like(partial) for _ in range(world)]
        dist.all_gather(parts, partial, group=group)
        rows = partial.shape[0]//world
        coded = torch.zeros_like(actual)
        for source in range(world):
            value = parts[source][rank*rows:(rank+1)*rows]
            if source != rank:
                value = reference_roundtrip(value)
            coded = (coded+value).to(torch.bfloat16)
    denominator = max(bf16.float().norm().item(), 1e-20)
    error = dict(max_abs=(actual.float()-bf16.float()).abs().max().item(),
        relative_l2=(actual.float()-bf16.float()).norm().item()/denominator,
        protocol_relative_l2=(actual.float()-coded.float()).norm().item()/denominator,
        budget_vs_bf16=.05, budget_vs_codec=.01,
        reference='torch_bf16_matmul_nccl_rs_and_independent_block_int8_source_fold',
        quantization='none' if arm == 'flux-bf16' else 'all_remote_block_int8')
    error['passed'] = bool(torch.isfinite(actual).all()) and error['relative_l2'] <= .05 and error['protocol_relative_l2'] <= .01
    assert error['passed'], error
    return error
