TRANSPORT = 'rtx5090-nccl-staged-v2'
TRANSPORT_ENV = {'NCCL_P2P_DISABLE': '1', 'NCCL_SHM_DISABLE': '0', 'NCCL_NVLS_ENABLE': '0'}
"""RTX 5090 TP8 NCCL staging worker gate; peer access is diagnostic."""
import os
import socket

def check_topology(world):
    import torch
    if world != 8 or torch.cuda.device_count() != world:
        raise RuntimeError('Expose exactly eight RTX 5090 devices for TP8 DP1')
    devices = []
    for index in range(world):
        prop = torch.cuda.get_device_properties(index)
        if ('5090' not in prop.name or (prop.major, prop.minor) != (12, 0)
                or prop.multi_processor_count != 170):
            raise RuntimeError('Expected full RTX 5090 SM120 with 170 SMs: ' + prop.name)
        devices.append(dict(name=prop.name, uuid=str(prop.uuid), sm_count=prop.multi_processor_count,
                            capability=[prop.major, prop.minor]))
    if len({device['uuid'] for device in devices}) != world:
        raise RuntimeError('Eight distinct GPUs are required')
    peers = [[i == j or torch.cuda.can_device_access_peer(i, j) for j in range(world)]
             for i in range(world)]
    if any(os.environ.get(key) != value for key, value in TRANSPORT_ENV.items()):
        raise RuntimeError('NCCL staging transport settings changed')
    if os.environ.get('RTX5090_TRANSPORT') != TRANSPORT:
        raise RuntimeError('Wrong RTX5090 transport protocol')
    return dict(host=socket.gethostname(), devices=devices, local_rank=int(os.environ['LOCAL_RANK']),
                peer_access=peers, transport=TRANSPORT)
