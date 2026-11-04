"""One model choice shared by CLI, API and queue; no implicit fallback."""
from pathlib import Path

SAM_MODELS = ('sam3', 'efficientsam3')


def resolve_sam(model, backend, device, directory, config, root):
    if model not in SAM_MODELS or backend not in ('torch', 'openvino'):
        raise ValueError('Unsupported SAM model/backend')
    if model == 'efficientsam3' and backend != 'openvino':
        raise ValueError('EfficientSAM3 currently requires --sam-backend openvino --device cpu; neural inference is GPU only')
    if backend == 'openvino' and device != 'cpu':
        raise ValueError('OpenVINO SAM requires --device cpu for tensor glue; neural inference is GPU only')
    root = Path(root)
    checkpoint = (config['sam_checkpoint'] if model == 'sam3' else
        config.get('efficient_sam_checkpoint', root/'weights/efficientsam3/efficientsam3_efficientvit.pt'))
    if backend == 'openvino' and directory is None:
        directory = root/'weights/openvino'/('sam3-gloved-hand-v7-f16' if model == 'sam3'
            else 'efficientsam3-evm-gloved-hand-v3-scale8')
    return dict(model=model, checkpoint=Path(checkpoint).resolve(),
        directory=Path(directory).resolve() if directory is not None else None,
        experimental=model == 'efficientsam3')
