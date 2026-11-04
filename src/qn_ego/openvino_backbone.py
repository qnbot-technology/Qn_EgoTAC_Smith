"""Explicit, receipt-checked OpenVINO adapter; temporal model and MANO stay in Torch."""
import json
from pathlib import Path

import numpy as np
import torch

from qn_smith.artifacts import sha256


def compile_backbone(model_path, device, threads):
    import openvino as ov
    core = ov.Core()
    if device not in core.available_devices:
        raise RuntimeError(f'OpenVINO device {device!r} unavailable: {core.available_devices}')
    options = {'PERFORMANCE_HINT': 'LATENCY', 'INFERENCE_PRECISION_HINT': 'f32'}
    if device == 'CPU':
        options['INFERENCE_NUM_THREADS'] = str(threads)
        options['NUM_STREAMS'] = '1'
    return core.compile_model(str(model_path), device, options)


class OpenVINOBackbone(torch.nn.Module):
    def __init__(self, path, checkpoint, device='CPU', threads=4):
        super().__init__()
        path = Path(path).resolve()
        self.receipt = json.loads(path.with_name('validation.json').read_text())
        if not self.receipt['numeric_passed'] or self.receipt['checkpoint_sha256'] != sha256(checkpoint):
            raise ValueError('Unvalidated or different OpenVINO source model')
        for rec in self.receipt['model_files']:
            p = path.parent / rec['path']
            if p.is_symlink() or sha256(p) != rec['sha256']:
                raise ValueError('OpenVINO model changed: '+str(p))
        devices = self.receipt['benchmarks']
        if device not in devices or not devices[device].get('numeric_passed', False):
            raise ValueError('Selected OpenVINO device has no successful numeric validation')
        self.compiled = compile_backbone(path, device, threads)
        self.request = self.compiled.create_infer_request()

    def forward(self, image):
        if image.device.type != 'cpu' or tuple(image.shape[1:]) != (3, 256, 192):
            raise ValueError('OpenVINO backbone expects CPU N×3×256×192 normalized HaWoR crops')
        # Fixed batch=1 limits Intel shared-memory usage; sequence order is retained.
        output = []
        for x in image.detach():
            self.request.infer({0: np.ascontiguousarray(x[None].numpy(), dtype=np.float32)})
            output.append(torch.from_numpy(self.request.get_output_tensor(0).data.copy()))
        return torch.cat(output)
