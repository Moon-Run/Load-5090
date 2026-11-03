PROTOCOL = 'rtx5090-open-compress-int8-six-arm-v2'
POLICIES = ('flux-bf16', 'flux-int8', 'flux-int8-interleaved', 'megatron-bf16', 'megatron-int8', 'flux-int8-remote-first')
MEGATRON_ARMS = ('megatron-bf16', 'megatron-int8')
SHAPES = {'attn': [2048, 4096, 4096], 'mlp': [2048, 4096, 16384]}
DEFINITIONS = {'flux-bf16': {'policy': 'flux-bf16', 'family': 'flux_tensor', 'scope': 'attn_mlp', 'roles': ['attn', 'mlp'], 'implementation': 'rtx5090-open-compress-flux-bf16', 'workspace': 'tensor', 'output_clone': False, 'base': 'original', 'order': 'original', 'arrival': False, 'interleaved': False, 'quantization': 'none', 'experiment': {'protocol': 'rtx5090-open-compress-int8-six-arm-v2', 'producer': 'flux-gemmonly-sm89-v2-sm120', 'gemm_schedule': 'StreamkSK', 'quantization': 'none', 'interleaved': False, 'remote_first': False, 'arrival': False, 'selective': False, 'calibration_used': False, 'cta_order': 'original-Flux-GemmOnly', 'may_change_streamk_reduction_assignment': False, 'collective': 'NCCL-BF16-ReduceScatter', 'receiver': 'NCCL', 'owns_output': True, 'owns_configuration': True, 'taco_runtime': False, 'original_peer_overlap_preserved': False, 'native_row_forward': False, 'native_linear_backward': False, 'native_rs_backward': False}}, 'flux-int8': {'policy': 'flux-int8', 'family': 'flux_tensor', 'scope': 'attn_mlp', 'roles': ['attn', 'mlp'], 'implementation': 'rtx5090-open-compress-flux-int8', 'workspace': 'tensor', 'output_clone': False, 'base': 'original', 'order': 'original', 'arrival': False, 'interleaved': False, 'quantization': 'all_remote', 'experiment': {'protocol': 'rtx5090-open-compress-int8-six-arm-v2', 'producer': 'flux-gemmonly-sm89-v2-sm120', 'gemm_schedule': 'StreamkSK', 'quantization': 'all_remote_block_int8_128_fp32_scale', 'interleaved': False, 'remote_first': False, 'arrival': False, 'selective': False, 'calibration_used': False, 'cta_order': 'original-Flux-GemmOnly', 'may_change_streamk_reduction_assignment': False, 'collective': 'NCCL-byte-AllToAll', 'receiver': 'block-int8-decode-source-ordered-bf16-fold', 'owns_output': True, 'owns_configuration': True, 'taco_runtime': False, 'original_peer_overlap_preserved': False, 'native_row_forward': False, 'native_linear_backward': False, 'native_rs_backward': False}}, 'flux-int8-interleaved': {'policy': 'flux-int8-interleaved', 'family': 'flux_tensor', 'scope': 'attn_mlp', 'roles': ['attn', 'mlp'], 'implementation': 'rtx5090-open-compress-flux-int8-interleaved', 'workspace': 'tensor', 'output_clone': False, 'base': 'original', 'order': 'interleaved', 'arrival': False, 'interleaved': True, 'quantization': 'all_remote', 'experiment': {'protocol': 'rtx5090-open-compress-int8-six-arm-v2', 'producer': 'flux-gemmonly-sm89-v2-sm120', 'gemm_schedule': 'StreamkSK', 'quantization': 'all_remote_block_int8_128_fp32_scale', 'interleaved': True, 'remote_first': False, 'arrival': False, 'selective': False, 'calibration_used': False, 'cta_order': 'rank-rotated-destination-interleaved', 'may_change_streamk_reduction_assignment': True, 'collective': 'NCCL-byte-AllToAll', 'receiver': 'block-int8-decode-source-ordered-bf16-fold', 'owns_output': True, 'owns_configuration': True, 'taco_runtime': False, 'original_peer_overlap_preserved': False, 'native_row_forward': False, 'native_linear_backward': False, 'native_rs_backward': False}}, 'megatron-bf16': {'policy': 'megatron-bf16', 'family': 'megatron', 'scope': 'attn_mlp', 'roles': ['attn', 'mlp'], 'implementation': 'rtx5090-open-compress-megatron-bf16', 'workspace': 'tensor', 'output_clone': False, 'base': 'original', 'order': 'original', 'arrival': False, 'interleaved': False, 'quantization': 'none', 'experiment': {'protocol': 'rtx5090-open-compress-int8-six-arm-v2', 'producer': 'native-megatron-linear', 'gemm_schedule': 'original-Megatron', 'quantization': 'none', 'interleaved': False, 'remote_first': False, 'arrival': False, 'selective': False, 'calibration_used': False, 'cta_order': 'original-Megatron', 'may_change_streamk_reduction_assignment': False, 'collective': 'NCCL-BF16-ReduceScatter', 'receiver': 'NCCL', 'owns_output': True, 'owns_configuration': True, 'taco_runtime': False, 'original_peer_overlap_preserved': False, 'native_row_forward': True, 'native_linear_backward': True, 'native_rs_backward': True}}, 'megatron-int8': {'policy': 'megatron-int8', 'family': 'megatron', 'scope': 'attn_mlp', 'roles': ['attn', 'mlp'], 'implementation': 'rtx5090-open-compress-megatron-int8', 'workspace': 'tensor', 'output_clone': False, 'base': 'original', 'order': 'original', 'arrival': False, 'interleaved': False, 'quantization': 'all_remote', 'experiment': {'protocol': 'rtx5090-open-compress-int8-six-arm-v2', 'producer': 'native-megatron-linear', 'gemm_schedule': 'original-Megatron', 'quantization': 'all_remote_block_int8_128_fp32_scale', 'interleaved': False, 'remote_first': False, 'arrival': False, 'selective': False, 'calibration_used': False, 'cta_order': 'original-Megatron', 'may_change_streamk_reduction_assignment': False, 'collective': 'NCCL-byte-AllToAll', 'receiver': 'block-int8-decode-source-ordered-bf16-fold', 'owns_output': True, 'owns_configuration': True, 'taco_runtime': False, 'original_peer_overlap_preserved': False, 'native_row_forward': True, 'native_linear_backward': True, 'native_rs_backward': True}}, 'flux-int8-remote-first': {'policy': 'flux-int8-remote-first', 'family': 'flux_tensor', 'scope': 'attn_mlp', 'roles': ['attn', 'mlp'], 'implementation': 'rtx5090-open-compress-flux-int8-remote-first', 'workspace': 'tensor', 'output_clone': False, 'base': 'original', 'order': 'remote-first', 'arrival': False, 'interleaved': False, 'quantization': 'all_remote', 'experiment': {'protocol': 'rtx5090-open-compress-int8-six-arm-v2', 'producer': 'flux-gemmonly-sm89-v2-sm120', 'gemm_schedule': 'StreamkSK', 'quantization': 'all_remote_block_int8_128_fp32_scale', 'interleaved': False, 'remote_first': True, 'arrival': False, 'selective': False, 'calibration_used': False, 'cta_order': 'remote-first-local-last', 'may_change_streamk_reduction_assignment': True, 'collective': 'NCCL-byte-AllToAll', 'receiver': 'block-int8-decode-source-ordered-bf16-fold', 'owns_output': True, 'owns_configuration': True, 'taco_runtime': False, 'original_peer_overlap_preserved': False, 'native_row_forward': False, 'native_linear_backward': False, 'native_rs_backward': False}}}

def require(ok, message):
    if not ok: raise ValueError(message)

def definition(policy):
    require(policy in POLICIES, 'Unknown compression policy')
    return DEFINITIONS[policy]

def is_native(policy):
    definition(policy)
    return policy in MEGATRON_ARMS

def is_flux_tensor(policy):
    definition(policy)
    return policy not in MEGATRON_ARMS

def is_tensor(policy):
    definition(policy)
    return True

def is_fused_v2(policy):
    definition(policy)
    return False

def is_original(policy):
    definition(policy)
    return True

def needs_mapping(policy):
    definition(policy)
    return policy in ('flux-int8-interleaved', 'flux-int8-remote-first')

def module_role(name):
    for role, suffix in (('attn', '.self_attention.linear_proj'), ('mlp', '.mlp.linear_fc2')):
        if name.endswith(suffix): return role
    raise ValueError('Unexpected target projection: '+name)

def enabled(policy, name):
    definition(policy)
    module_role(name)
    return True

def route(policy, name):
    enabled(policy, name)
    return ('megatron_native' if policy == 'megatron-bf16' else
            'megatron_int8' if policy == 'megatron-int8' else 'flux_tensor')

def codec_placement(policy):
    definition(policy)
    return 'tensor'

def module_routes(observed, policy, layers=32, model=None):
    require(layers == 32, 'Expected 32 layers')
    counts, routes = dict.fromkeys(SHAPES, 0), {}
    for name, shape in observed.items():
        role = module_role(name)
        require([shape[k] for k in ('M', 'N', 'K_global')] == SHAPES[role]
                and shape['K_local'] == SHAPES[role][2]//8, 'Wrong projection shape')
        counts[role] += 1
        routes[name] = route(policy, name)
    require(counts == {'attn': 32, 'mlp': 32}, 'All 64 projections required')
    return routes

def role_plans(plan, policy):
    definition(policy)
    require(plan['protocol'] == PROTOCOL and plan['calibration_used'] is False
            and set(plan) == {'protocol', 'calibration_used', 'roles'}, 'Calibration forbidden')
    return plan['roles']

def selection_entry(*args):
    raise ValueError('This experiment forbids selection masks and arrival tables')
