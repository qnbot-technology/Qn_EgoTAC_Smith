#!/usr/bin/env python3
"""Import the exact standalone inference closure before expensive stages."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from qn_smith.artifacts import write_json

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--mode',choices=['sam','mano'],required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--backend',choices=['torch','openvino'],default='torch')
p.add_argument('--openvino-dir',type=Path)
p.add_argument('--checkpoint',type=Path)
p.add_argument('--sam-model',choices=['sam3','efficientsam3'],default='sam3')
a=p.parse_args()
if a.backend=='openvino':
    if a.mode!='sam' or a.openvino_dir is None or a.checkpoint is None:
        raise ValueError('SAM3 GPU preflight requires explicit model directory and checkpoint')
    from qn_ego.sam3_assets import validate_sam3_assets,validate_efficientsam3_assets
    from qn_smith.artifacts import sha256
    import openvino as ov
    validate=validate_sam3_assets if a.sam_model=='sam3' else validate_efficientsam3_assets
    receipt=validate(a.openvino_dir,a.checkpoint)
    core=ov.Core()
    if 'GPU' not in core.available_devices:raise RuntimeError('Intel GPU unavailable; no CPU fallback')
    write_json(a.output,dict(passed=True,mode='sam',backend='openvino',device='GPU',
        device_name=core.get_property('GPU','FULL_DEVICE_NAME'),
        gpu_available=True,device_discovery_passed=True,execution_probe_passed=False,
        validation_sha256=sha256(a.openvino_dir/'validation.json'),
        numeric_validation_passed=receipt['numeric_passed'],cpu_neural_inference=False,
        sam_model=a.sam_model,model_status='experimental' if a.sam_model=='efficientsam3' else 'validated',
        cpu_fallback=False,python=sys.version,interpreter=sys.executable,project_root=str(ROOT)))
    print('SAM3 GPU assets and device discovery passed; inference verified by separate numeric receipt',flush=True)
    sys.exit(0)
if a.mode=='sam' and a.sam_model!='sam3':
    raise ValueError('EfficientSAM3 currently requires OpenVINO GPU; no torch/CPU fallback')
import torch
device=os.environ.get('QN_EGO_DEVICE','cuda')
if device=='cuda' and not torch.cuda.is_available():
    raise RuntimeError('Requested CUDA unavailable; explicitly select --device cpu for full CPU inference or --visual flow for diagnostics')
probe = torch.eye(4, device=device)
if not torch.allclose(probe @ probe, probe):
    raise RuntimeError(device+' execution probe failed')
if device=='cuda': torch.cuda.synchronize()
if a.mode=='sam':
    sys.path.insert(0,str(ROOT/'third_party/sam3'))
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
else:
    sys.path.insert(0,str(ROOT/'third_party/egosmith/src'))
    from hawor_loader import load_hawor
    from lib.pipeline.hands.mano_runtime import run_mano,run_mano_left
write_json(a.output,dict(passed=True,mode=a.mode,torch=torch.__version__,
    cuda_available=torch.cuda.is_available(),device=torch.cuda.get_device_name(0) if device=='cuda' else 'cpu',python=sys.version,
    device_capability=list(torch.cuda.get_device_capability(0)) if device=='cuda' else None,
    torch_cuda=torch.version.cuda,cuda_execution_passed=device=='cuda',execution_probe_passed=True,
    interpreter=sys.executable,project_root=str(ROOT)))
print('Standalone model runtime passed:',a.mode,flush=True)
