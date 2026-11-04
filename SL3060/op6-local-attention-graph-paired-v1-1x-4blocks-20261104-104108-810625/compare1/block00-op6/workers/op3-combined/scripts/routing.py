"""Standalone OP1 ABC routing; sparse selection is generated from output shape."""
PROTOCOL = 'rtx5090-phase6-op3-combined-v2'
POLICIES = ('op3-combined',)
DEFINITIONS = {'op3-combined': {'policy': 'op3-combined', 'family': 'flux_tensor', 'scope': 'attn_mlp', 'roles': ['attn', 'mlp'], 'implementation': 'rtx5090-op3-combined-v1', 'workspace': 'tensor', 'output_clone': False, 'base': 'original', 'order': 'interleaved', 'arrival': False, 'interleaved': True, 'quantization': 'all_remote', 'experiment': {'protocol': 'rtx5090-phase6-op3-combined-v2', 'original_arm': 'flux-basic', 'producer': 'flux-gemmonly-sm89-v2', 'gemm_schedule': 'StreamkSK-with-CTA-interleave', 'may_change_streamk_reduction_assignment': True, 'quantization': 'all_remote', 'interleaved': True, 'arrival': False, 'selective': False, 'calibration_used': False, 'cta_order': 'rank-rotated-destination-interleaved', 'decoder': 'r-packed-source-prefetch', 'collective': 'NCCL-byte-AllToAll', 'chunks': 1, 'owns_configuration': True, 'owns_output': True, 'packet_layout': 'H128-AoS-136', 'original_peer_overlap_preserved': False, 'arrival_scope': 'whole-exchange', 'foundation': 'OP2-A-MA89-MA11', 'optimizations': {'ma8': 'retain-owned-forward-column-input-v2', 'ma9': 'ordered-fp32-gradient-zero-count', 'ma11': 'pytorch28-exact-causal-softmax'}, 'repeated_ma89_installation': False, 'encoder': 'warp-owned-four-groups', 'r_decoder': True, 'cr3_consumer_fusion': True, 'consumer_rounding': 'BF16-projection-plus-bias-then-residual'}}}
SHAPES = {'attn': [2048, 4096, 4096], 'mlp': [2048, 4096, 16384]}

def require(ok, message):
    if not ok:
        raise ValueError(message)

def definition(policy):
    require(policy in POLICIES, 'Unknown OP3 policy')
    return DEFINITIONS[policy]

def is_native(policy):
    definition(policy)
    return False

def is_flux_tensor(policy):
    definition(policy)
    return True

def is_tensor(policy):
    return is_flux_tensor(policy)

def is_fused_v2(policy):
    return not is_flux_tensor(policy)

def is_original(policy):
    definition(policy)
    return True

def needs_mapping(policy):
    return is_fused_v2(policy)

def module_role(name):
    for role, suffix in (('attn', '.self_attention.linear_proj'), ('mlp', '.mlp.linear_fc2')):
        if name.endswith(suffix):
            return role
    raise ValueError('Unexpected target projection: ' + name)

def enabled(policy, name):
    definition(policy)
    module_role(name)
    return True

def route(policy, name):
    enabled(policy, name)
    return 'flux_tensor' if is_flux_tensor(policy) else 'candidate'

def codec_placement(policy):
    return 'tensor' if is_flux_tensor(policy) else 'fused'

def module_routes(observed, policy, layers=32, model=None):
    require(layers == 32, 'Expected 32 layers')
    counts, routes = (dict.fromkeys(SHAPES, 0), {})
    for name, shape in observed.items():
        role = module_role(name)
        require([shape[k] for k in ('M', 'N', 'K_global')] == SHAPES[role] and shape['K_local'] == SHAPES[role][2] // 8, 'Wrong projection shape')
        counts[role] += 1
        routes[name] = route(policy, name)
    require(counts == {'attn': 32, 'mlp': 32}, 'All 64 projections required')
    return routes

def role_plans(plan, policy):
    definition(policy)
    require(plan['protocol'] == PROTOCOL and plan['calibration_used'] is False and (set(plan) == {'protocol', 'calibration_used', 'roles'}), 'No calibration inputs permitted')
    return plan['roles']

def selection_entry(*args):
    raise ValueError('OP3 always quantizes all remote groups; no selection table')
