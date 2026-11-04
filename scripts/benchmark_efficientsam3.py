#!/usr/bin/env python3
"""Isolated EV-M export and same-frame Intel GPU benchmark; never promotes full."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
EXPECTED_SHA = '086b04b2e7da7cc98aa4621b70c7291608aa9d187357b98d03bd4d6533ed5a17'
UPSTREAM_COMMIT = 'bd0936c788fed8d51fa799437f05abd97b401b06'
RAW_NAMES = ['pred_logits', 'pred_boxes', 'pred_masks', 'presence_logit_dec']

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()

def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def load_model(output):
    import torch
    sys.path.insert(0, str(ROOT / 'third_party/efficientsam3/sam3'))
    from sam3.model_builder import build_efficientsam3_image_model
    checkpoint = ROOT / 'weights/efficientsam3/efficientsam3_efficientvit.pt'
    if sha(checkpoint) != EXPECTED_SHA: raise ValueError('EV-M checkpoint SHA mismatch')
    model = build_efficientsam3_image_model(checkpoint_path=None, load_from_HF=False,
        backbone_type='efficientvit', model_name='b1', text_encoder_type='MobileCLIP-S0',
        text_encoder_context_length=16, enable_inst_interactivity=False, device='cpu',
        bpe_path=str(ROOT / 'third_party/efficientsam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz'))
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if isinstance(state.get('model'), dict): state = state['model']
    cleaned = {k.removeprefix('detector.').replace('student_trunk.', ''): v for k,v in state.items()}
    target = model.state_dict()
    missing = sorted(set(target) - set(cleaned))
    unexpected = sorted(set(cleaned) - set(target))
    mismatched = {k: [list(cleaned[k].shape), list(target[k].shape)]
                  for k in target.keys() & cleaned.keys() if cleaned[k].shape != target[k].shape}
    report = dict(checkpoint_sha256=EXPECTED_SHA, upstream_commit=UPSTREAM_COMMIT,
        configuration=dict(backbone='efficientvit', model_name='b1', text='MobileCLIP-S0', context=16),
        missing=missing, unexpected=unexpected, shape_mismatches=mismatched,
        parameters=sum(x.numel() for x in model.parameters()),
        status='passed' if not (missing or mismatched) else 'failed',
        note='Unexpected non-image keys excluded only after reporting; no missing target keys allowed.')
    write(output / 'checkpoint-audit.json', report)
    if missing or mismatched: raise RuntimeError('Incomplete checkpoint: see checkpoint-audit.json')
    model.load_state_dict({k:cleaned[k] for k in target}, strict=True)
    del state, cleaned, target
    model.eval()
    return model

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase', choices=['reference','image','grounding','validate','benchmark'], required=True)
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--ir', type=Path)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--frames', type=int, default=16)
    ap.add_argument('--model', choices=['efficient','sam3'], default='efficient')
    ap.add_argument('--allow-unvalidated-diagnostic', action='store_true',
                    help='Benchmark finite experimental outputs without passing numeric gate; never enables full')
    a=ap.parse_args(argv)
    if a.allow_unvalidated_diagnostic and (a.phase!='benchmark' or a.model!='efficient'):
        ap.error('Diagnostic override is only valid for the isolated EfficientSAM3 benchmark')
    import numpy as np
    import torch
    from PIL import Image
    torch.set_num_threads(a.threads)
    a.input=a.input.resolve();a.output=a.output.resolve()
    os.environ['HF_HUB_OFFLINE']='1'
    info=json.loads((a.input/'input.json').read_text())
    start=time.monotonic()
    receipt=a.output/(a.phase+'-receipt.json')
    if receipt.exists(): raise FileExistsError(receipt)
    if a.phase in ['reference','benchmark']: a.output.mkdir(parents=True,exist_ok=False)
    print('PHASE',a.phase,'START',time.strftime('%Y-%m-%dT%H:%M:%S%z'),flush=True)
    if a.phase in ['reference','image','grounding']:
        model=load_model(a.output)
    if a.phase=='reference':
        from sam3.model.sam3_image_processor import Sam3Processor
        processor=Sam3Processor(model, device='cpu', confidence_threshold=.35)
        rows=sorted(set([0,info['rows']-1]))
        write(a.output/'conversion.json',dict(schema='qn_ego.efficientsam3_openvino.v1',
            prompt='gloved hand', resolution=1008, confidence_threshold=.35, rows=rows,
            source_sha256=info['input_sha256'],checkpoint_sha256=EXPECTED_SHA,
            upstream_commit=UPSTREAM_COMMIT,precision='f16',production_device='GPU',
            reference_only_cpu=True,cpu_fallback=False,training_ready=False,argv=sys.argv))
        with torch.inference_mode():
            text=model.backbone.forward_text(['gloved hand'],device='cpu')
            np.savez_compressed(a.output/'text-cache.npz',**{k:v.numpy() for k,v in text.items()})
            for row in rows:
                path=a.input/'images'/f'{row:06d}.jpg';im=Image.open(path).convert('RGB')
                tic=time.monotonic();state=processor.set_image(im);b=state['backbone_out'];b.update(text)
                raw=model.forward_grounding(backbone_out=b,find_input=processor.find_stage,
                    find_target=None,geometric_prompt=model._get_dummy_prompt())
                np.savez_compressed(a.output/f'reference-{row:06d}.npz',
                    **{k:raw[k].numpy() for k in RAW_NAMES},height=im.height,width=im.width,
                    image_sha256=sha(path),seconds=time.monotonic()-tic)
                if row==rows[0]:
                    from torchvision.transforms import v2
                    x=processor.transform(v2.functional.to_image(im)).unsqueeze(0)
                    np.save(a.output/'image-input.npy',x.numpy())
                    np.savez_compressed(a.output/'image-features.npz',
                        **{f'x{i}':v.numpy() for i,v in enumerate(b['backbone_fpn']+b['vision_pos_enc'])})
                print('REFERENCE',row,round(time.monotonic()-tic,3),flush=True)
    elif a.phase in ['image','grounding']:
        import openvino as ov
        from qn_ego.sam3_export import ImageEncoder,TextOnlyGrounding,EmptyGeometryEncoder
        if a.phase=='image':
            wrapper=ImageEncoder(model.backbone)
            inputs=(torch.from_numpy(np.load(a.output/'image-input.npy')),)
        else:
            with np.load(a.output/'text-cache.npz') as z: text={k:torch.from_numpy(z[k]) for k in z.files}
            wrapper=TextOnlyGrounding(model,text)
            with np.load(a.output/'image-features.npz') as z: inputs=tuple(torch.from_numpy(z[f'x{i}']) for i in range(6))
        del model;gc.collect();wrapper.eval()
        for mod in wrapper.modules():
            if getattr(mod,'act_ckpt',False): mod.act_ckpt=False
        with torch.inference_mode():
            native=wrapper(*inputs)
            if a.phase=='grounding':
                wrapper.model.geometry_encoder=EmptyGeometryEncoder(wrapper.model.geometry_encoder)
                specialized=wrapper(*inputs)
                diffs=[float((x-y).abs().max()) for x,y in zip(native,specialized)]
                for x,y in zip(native,specialized): torch.testing.assert_close(x,y,rtol=1e-5,atol=1e-5)
                write(a.output/'empty-geometry-equivalence.json',dict(passed=True,max_abs=diffs))
                del specialized
            print('NATIVE', [list(x.shape) for x in native],flush=True)
            traced=torch.jit.trace(wrapper, inputs, strict=False,check_trace=False)
            actual=traced(*inputs)
            diffs=[float((x-y).abs().max()) for x,y in zip(native,actual)]
            for x,y in zip(native,actual): torch.testing.assert_close(x,y,rtol=1e-4,atol=1e-4)
            torch.jit.save(traced,a.output/(a.phase+'-traced.pt'))
            shapes=[(ov.PartialShape(list(x.shape)),ov.Type.f32) for x in inputs]
            del wrapper,native,actual,traced;gc.collect()
            print('OV CONVERT',a.phase,flush=True)
            converted=ov.convert_model(str(a.output/(a.phase+'-traced.pt')),input=shapes)
            if any(p.get_element_type()!=ov.Type.f32 for p in converted.inputs): raise ValueError('Input dtype changed')
            if any(n.get_type_name()=='ConvertPromoteTypes' for n in converted.get_ordered_ops()): raise ValueError('Unresolved dtype')
            ov.save_model(converted,a.output/(a.phase+'.xml'),compress_to_fp16=False)
            write(a.output/(a.phase+'-trace-comparison.json'),dict(passed=True,max_abs=diffs))
    else:
        from qn_ego.openvino_sam3 import UnvalidatedSAM3GPU,OpenVINOSAM3Processor,postprocess
        ir=(a.ir or a.output).resolve()
        numeric_receipt=None
        if a.phase=='benchmark' and a.model=='efficient':
            from qn_ego.sam3_assets import validate_efficientsam3_benchmark
            numeric_receipt=validate_efficientsam3_benchmark(ir,info['input_sha256'],a.allow_unvalidated_diagnostic)
        t0=time.monotonic()
        if a.model=='sam3': proc=OpenVINOSAM3Processor(ir,ROOT/'weights/sam3/sam3.pt')
        else:
            conversion=json.loads((ir/'conversion.json').read_text())
            proc=UnvalidatedSAM3GPU(ir,precision='f16',
                image_activation_scale_factor=conversion.get('image_activation_scale_factor'))
        startup=time.monotonic()-t0
        def predict(row):
            im=Image.open(a.input/'images'/f'{row:06d}.jpg').convert('RGB');t=time.monotonic()
            with torch.inference_mode(): pred=proc.set_text_prompt('gloved hand',proc.set_image(im))
            return pred,time.monotonic()-t,dict(proc.inference_seconds)
        if a.phase=='validate':
            from scipy.optimize import linear_sum_assignment
            rec=json.loads((a.output/'conversion.json').read_text());results=[]
            for row in rec['rows']:
                with np.load(a.output/f'reference-{row:06d}.npz') as z:
                    if str(z['image_sha256'])!=sha(a.input/'images'/f'{row:06d}.jpg'): raise ValueError('Frame changed')
                    ref=postprocess({k:z[k] for k in RAW_NAMES},int(z['height']),int(z['width']),.35)
                pred,seconds,stages=predict(row)
                rm=ref['masks'].numpy()[:,0];pm=pred['masks'].numpy()[:,0]
                iou=np.array([[(r&p).sum()/max(1,(r|p).sum()) for p in pm] for r in rm]).reshape(len(rm),len(pm))
                ri,pi=linear_sum_assignment(-iou);matched=[float(iou[i,j]) for i,j in zip(ri,pi)]
                bm=max([float((ref['boxes'][i]-pred['boxes'][j]).abs().max()) for i,j in zip(ri,pi)] or [1e9])
                sm=max([float(abs(ref['scores'][i]-pred['scores'][j])) for i,j in zip(ri,pi)] or [1e9])
                passed=len(rm)==len(pm)>0 and min(matched,default=0)>=.98 and bm<=2 and sm<=.01
                results.append(dict(row=row,reference_instances=len(rm),gpu_instances=len(pm),mask_ious=matched,box_max_px=bm,score_max_abs=sm,passed=passed,gpu_s=seconds,stages=stages))
                np.savez_compressed(a.output/f'gpu-{row:06d}.npz',**{k:v.numpy() for k,v in pred.items()})
                print('VALIDATION',results[-1],flush=True)
            result=dict(numeric_passed=all(x['passed'] for x in results),comparisons=results,execution_devices=proc.execution_devices,precision='f16',training_ready=False,
                image_activation_scale_factor=conversion.get('image_activation_scale_factor'),
                ir_sha256={name:sha(ir/name) for name in ['image.xml','image.bin','grounding.xml','grounding.bin']})
            write(a.output/'validation.json',result)
            if not result['numeric_passed']: raise RuntimeError('GPU numeric validation failed')
        else:
            warm=[]
            for _ in range(2):
                _,s,st=predict(0);warm.append(dict(seconds=s,stages=st))
            records=[]
            for row in range(min(a.frames,info['rows'])):
                pred,seconds,stages=predict(row)
                np.savez_compressed(a.output/f'prediction-{row:06d}.npz',masks=pred['masks'].numpy(),boxes=pred['boxes'].numpy(),scores=pred['scores'].numpy())
                rec=dict(row=row,seconds=seconds,stages=stages,instances=len(pred['scores']),image_sha256=sha(a.input/'images'/f'{row:06d}.jpg'))
                records.append(rec);print('FRAME',rec,flush=True)
                write(a.output/'progress.json',dict(completed=len(records),total=min(a.frames,info['rows'])))
            seconds=[r['seconds'] for r in records]
            write(a.output/'summary.json',dict(model=a.model,source_sha256=info['input_sha256'],input=str(a.input),ir=str(ir),
                prompt='gloved hand',resolution=1008,threshold=.35,frames=records,startup_s=startup,warmup=warm,
                steady_mean_s=float(np.mean(seconds)),steady_median_s=float(np.median(seconds)),steady_p95_s=float(np.percentile(seconds,95)),
                execution_devices=proc.execution_devices,precision='f16',peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                image_activation_scale_factor=(conversion.get('image_activation_scale_factor') if a.model=='efficient' else None),
                gpu_memory_note='Intel GPU uses shared memory; RSS is process host memory, not dedicated VRAM.',
                cpu_neural_inference=False,cpu_fallback=False,training_ready=False,argv=sys.argv))
            summary=json.loads((a.output/'summary.json').read_text())
            summary.update(diagnostic_only=bool(a.allow_unvalidated_diagnostic),
                numeric_validation_passed=(numeric_receipt['numeric_passed'] if numeric_receipt else True),
                validation_receipt_sha256=sha(ir/'validation.json'),
                production_promoted=False)
            write(a.output/'summary.json',summary)
    write(receipt,dict(status='passed',phase=a.phase,elapsed_s=time.monotonic()-start,argv=sys.argv,peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    print('PHASE',a.phase,'DONE',round(time.monotonic()-start,3),flush=True)

if __name__=='__main__': main()
