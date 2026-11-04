#!/usr/bin/env python3
"""Export and compare real HaWoR crops. No quantization or automatic backend switch."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts'),
               str(ROOT/'third_party/egosmith/src'), str(ROOT/'third_party/egosmith')]
os.environ['QN_EGO_MANO_ROOT'] = str(ROOT/'weights/mano')
os.environ['QN_EGO_DEVICE'] = 'cpu'
os.environ['HF_HUB_OFFLINE'] = '1'


def main():
    import numpy as np
    import torch
    import openvino as ov
    from qn_smith.artifacts import sha256, write_json, now
    from qn_ego.openvino_backbone import compile_backbone
    from hawor_loader import load_hawor
    from lib.pipeline.io.frame_source import ImageFolderFrameSource
    from lib.pipeline.hands.track_dataset import TrackDatasetEval

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, required=True, help='Prepared visual/input/episode')
    ap.add_argument('--crops', type=Path, required=True, help='Same-source SAM crop NPZ')
    ap.add_argument('--output', type=Path, required=True, help='New weights/openvino directory')
    ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(a.threads)
    info = json.loads((a.input/'input.json').read_text())
    frames = ImageFolderFrameSource([a.input/'images'/f'{i:06d}.jpg' for i in range(info['rows'])], use_turbojpeg=False)
    tensors, samples = [], []
    with np.load(a.crops) as c:
        if str(c['input_sha256']) != info['input_sha256']:
            raise ValueError('Crop source mismatch')
        for side in ('left','right'):
            rows = np.flatnonzero(c[side+'_valid'])
            if not len(rows): continue
            rows = rows[np.unique(np.linspace(0, len(rows)-1, 2).astype(int))]
            ds = TrackDatasetEval(frames, rows, c[side+'_boxes'][rows], img_focal=float(info['K'][0][0]),
                img_center=np.array(info['K'])[:2,2].tolist(), dilate=1.2, do_flip=side=='left')
            for i, row in enumerate(rows):
                tensors.append(ds[i]['img'][None, :, :, 32:-32].contiguous())
                samples.append(dict(side=side, source_row=int(row)))
    if len(tensors) < 2:
        raise ValueError('Need at least two real hand crops for comparison')
    checkpoint = ROOT/'weights/hawor/checkpoints/hawor.ckpt'
    record = dict(schema='qn_ego.openvino_backbone.v1', started_at=now(), status='running',
        argv=sys.argv, torch=torch.__version__, openvino=ov.__version__, threads=a.threads,
        checkpoint_sha256=sha256(checkpoint), source_sha256=info['input_sha256'],
        input_json_sha256=sha256(a.input/'input.json'), crops_sha256=sha256(a.crops),
        samples=samples, input_shape=[1,3,256,192], storage_precision='FP32',
        inference_precision_hint='f32', training_ready=False,
        scope='HaWoR spatial backbone only; not SAM3 or complete pipeline speed/quality acceptance')
    write_json(a.output/'validation.json', record)
    print('LOAD TORCH', flush=True)
    model, _ = load_hawor(str(checkpoint))
    # Upstream ViT.train() does not return self, so its inherited eval() returns
    # None. Preserve the module reference instead of chaining eval().
    backbone = model.backbone
    backbone.eval()
    del model; gc.collect()
    references, torch_times = [], []
    with torch.inference_mode():
        backbone(tensors[0])  # warmup excluded
        for x in tensors:
            tic = time.perf_counter(); references.append(backbone(x).numpy().copy())
            torch_times.append(time.perf_counter()-tic)
        print('EXPORT OPENVINO', flush=True)
        tic = time.perf_counter()
        converted = ov.convert_model(backbone, example_input=tensors[0], input=[1,3,256,192])
        ov.save_model(converted, a.output/'model.xml', compress_to_fp16=False)
        record['export_s'] = time.perf_counter()-tic
    del backbone, converted; gc.collect()
    np.savez_compressed(a.output/'comparison-inputs.npz', inputs=np.concatenate([x.numpy() for x in tensors]),
                        torch_features=np.concatenate(references), torch_seconds=np.array(torch_times))
    record['model_files'] = [dict(path=p.name, bytes=p.stat().st_size, sha256=sha256(p))
                            for p in [a.output/'model.xml', a.output/'model.bin']]
    record['available_devices'] = ov.Core().available_devices
    record['torch_median_s'] = float(np.median(torch_times)); record['benchmarks'] = {}
    write_json(a.output/'validation.json', record)
    for device in record['available_devices']:
        if device not in ('CPU', 'GPU'): continue
        try:
            print('BENCHMARK', device, flush=True)
            tic = time.perf_counter(); compiled = compile_backbone(a.output/'model.xml', device, a.threads)
            compile_s = time.perf_counter()-tic
            req = compiled.create_infer_request(); req.infer({0:tensors[0].numpy()})
            errors, durations = [], []
            for x, ref in zip(tensors, references):
                tic = time.perf_counter(); req.infer({0:x.numpy()}); durations.append(time.perf_counter()-tic)
                pred = req.get_output_tensor(0).data.copy()
                rel_l2 = float(np.linalg.norm(pred-ref)/max(np.linalg.norm(ref), 1e-12))
                errors.append(dict(max_abs=float(np.max(np.abs(pred-ref))), relative_l2=rel_l2,
                    passed=bool(np.isfinite(pred).all() and rel_l2<1e-3 and np.max(np.abs(pred-ref))<.02)))
            record['benchmarks'][device] = dict(compile_s=compile_s, seconds=durations,
                median_s=float(np.median(durations)), speedup=record['torch_median_s']/float(np.median(durations)),
                comparisons=errors, numeric_passed=all(x['passed'] for x in errors))
            del req, compiled; gc.collect()
        except Exception as exc:
            record['benchmarks'][device] = dict(numeric_passed=False, error=repr(exc))
        write_json(a.output/'validation.json', record)
    record.update(status='completed', finished_at=now(),
                  numeric_passed=any(v['numeric_passed'] for v in record['benchmarks'].values()))
    write_json(a.output/'validation.json', record)
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__': main()
