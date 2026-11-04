"""Verify the exact GPU-validated SAM3 files without compiling or loading Torch."""
import json
from pathlib import Path

from qn_smith.artifacts import sha256


def validate_efficientsam3_benchmark(directory, source_sha256, diagnostic=False):
    """Bind an experimental benchmark to its audited IR; never authorizes full."""
    directory = Path(directory)
    conversion = json.loads((directory / 'conversion.json').read_text())
    receipt = json.loads((directory / 'validation.json').read_text())
    if conversion['source_sha256'] != source_sha256:
        raise ValueError('EfficientSAM3 comparison input changed')
    if receipt.get('image_activation_scale_factor') != conversion.get('image_activation_scale_factor'):
        raise ValueError('EfficientSAM3 activation scale changed after validation')
    required = {'image.xml', 'image.bin', 'grounding.xml', 'grounding.bin'}
    if set(receipt.get('ir_sha256', {})) != required:
        raise ValueError('Incomplete EfficientSAM3 validation hashes')
    for name in required:
        path = directory / name
        if path.is_symlink() or sha256(path) != receipt['ir_sha256'][name]:
            raise ValueError('EfficientSAM3 IR changed: ' + name)
    for name in ('image', 'grounding'):
        devices = receipt.get('execution_devices', {}).get(name, [])
        if not devices or any(not str(d).startswith('GPU') for d in devices):
            raise ValueError('EfficientSAM3 benchmark requires GPU validation execution')
    if receipt.get('numeric_passed') is not True and not diagnostic:
        raise ValueError('Unvalidated EV-M; only explicit diagnostic benchmark allowed')
    return receipt


def validate_efficientsam3_assets(directory, checkpoint):
    """Explicit model selection permits experimental use, retaining failed QA."""
    directory = Path(directory).resolve()
    conversion = json.loads((directory/'conversion.json').read_text())
    expected = '086b04b2e7da7cc98aa4621b70c7291608aa9d187357b98d03bd4d6533ed5a17'
    if conversion.get('schema') != 'qn_ego.efficientsam3_openvino.v1':
        raise ValueError('Not an EfficientSAM3 export')
    if (conversion.get('checkpoint_sha256') != expected or Path(checkpoint).is_symlink()
            or sha256(checkpoint) != expected):
        raise ValueError('EfficientSAM3 checkpoint mismatch')
    if (conversion.get('prompt') != 'gloved hand' or conversion.get('resolution') != 1008
            or conversion.get('confidence_threshold') != .35):
        raise ValueError('Unsupported EfficientSAM3 conversion configuration')
    receipt = validate_efficientsam3_benchmark(directory, conversion['source_sha256'], diagnostic=True)
    if receipt.get('precision') != 'f16':
        raise ValueError('Unsupported EfficientSAM3 precision')
    return dict(receipt, model_status='experimental', checkpoint_sha256=expected,
        user_selected_experimental_model=True)


def validate_sam3_assets(directory, checkpoint):
    directory=Path(directory).resolve()
    receipt=json.loads((directory/'validation.json').read_text())
    if receipt.get('numeric_passed') is not True or receipt.get('device')!='GPU':
        raise ValueError('SAM3 OpenVINO GPU has no successful comparison receipt')
    if receipt.get('precision') not in ('f32','f16'):
        raise ValueError('Unsupported SAM3 inference precision')
    for name in ('image','grounding'):
        devices=receipt.get('execution_devices',{}).get(name,[])
        if not devices or any(not str(d).startswith('GPU') for d in devices):
            raise ValueError('SAM3 receipt does not prove GPU execution: '+name)
    if receipt['checkpoint_sha256']!=sha256(checkpoint):
        raise ValueError('SAM3 checkpoint mismatch')
    required={'image.xml','image.bin','grounding.xml','grounding.bin'}
    records=receipt['model_files']
    if len(records)!=4 or {r['path'] for r in records}!=required:
        raise ValueError('Incomplete SAM3 IR manifest')
    for r in records:
        path=directory/r['path']
        if path.is_symlink() or path.stat().st_size!=r['bytes'] or sha256(path)!=r['sha256']:
            raise ValueError('SAM3 IR changed: '+str(path))
    conversion=json.loads((directory/'conversion.json').read_text())
    for key in ('checkpoint_sha256','prompt','resolution','confidence_threshold'):
        if receipt[key]!=conversion[key]:
            raise ValueError('SAM3 conversion config changed')
    return receipt
