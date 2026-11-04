#!/usr/bin/env python3
"""Convert original fixed-prompt SAM3 and validate real frames on Intel GPU.

CPU Torch is used only to create export/reference evidence. Production SAM3
inference uses GPU OpenVINO for all image-dependent learned operations.
Each phase is a separate process to release large export graphs before compiling.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import shutil

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'third_party/sam3'),str(ROOT/'third_party/openvino_sam3')]
os.environ['HF_HUB_OFFLINE']='1'
os.environ['QN_EGO_DEVICE']='cpu'


def main():
    import numpy as np
    import torch
    from PIL import Image
    from qn_smith.artifacts import sha256,write_json,now
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--phase',choices=['all','reference','image','grounding','validate'],default='all')
    ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--precision',choices=['f32','f16'],default='f32')
    ap.add_argument('--reuse-from',type=Path,help='Reuse successful reference/image stages into a NEW output; never overwrite parent')
    ap.add_argument('--validate-export-from',type=Path,help='Copy successful FP32 exports and references into a NEW validation experiment')
    ap.add_argument('--trace-first',action='store_true',help='Trace grounding separately to distinguish Torch and OpenVINO failures')
    ap.add_argument('--empty-geometry',action='store_true',help='Numerically verify and specialize the exactly empty geometric prompt')
    a=ap.parse_args();a.input=a.input.resolve();a.output=a.output.resolve()
    if a.reuse_from and a.validate_export_from:
        ap.error('--reuse-from and --validate-export-from are mutually exclusive')
    if a.empty_geometry and not a.trace_first:
        ap.error('--empty-geometry requires --trace-first for equivalence validation')
    torch.set_num_threads(a.threads)
    os.environ['QN_EGO_THREADS']=str(a.threads)
    checkpoint=ROOT/'weights/sam3/sam3.pt'
    if a.phase=='all':
        if a.output.exists():raise FileExistsError(a.output)
        phases=['reference','image','grounding','validate']
        if a.reuse_from or a.validate_export_from:
            parent=(a.reuse_from or a.validate_export_from).resolve()
            record=json.loads((parent/'conversion.json').read_text())
            info=json.loads((a.input/'input.json').read_text())
            if record['checkpoint_sha256']!=sha256(checkpoint) or record['source_sha256']!=info['input_sha256'] or (not a.validate_export_from and record['precision']!=a.precision):
                raise ValueError('Reuse input/checkpoint/precision mismatch')
            for phase in ['reference','image']+(['grounding'] if a.validate_export_from else []):
                if json.loads((parent/(phase+'-receipt.json')).read_text())['status']!='passed':raise ValueError('Unsuccessful inherited stage')
            names=['reference-receipt.json','image-receipt.json','image.xml','image.bin',
                   'image-input.npy','image-features.npz','text-cache.npz']
            names += [f'reference-{row:06d}.npz' for row in record['rows']]
            if a.validate_export_from:
                names += ['grounding.xml','grounding.bin','grounding-receipt.json']
                names += [name for name in ['empty-geometry-comparison.json'] if (parent/name).exists()]
            records=[dict(path=name,bytes=(parent/name).stat().st_size,sha256=sha256(parent/name)) for name in names]
            a.output.mkdir(parents=True)
            for item in records:
                shutil.copyfile(parent/item['path'],a.output/item['path'])
                if sha256(a.output/item['path'])!=item['sha256']:raise ValueError('Reuse copy mismatch')
            record.update(status='preparing',started_at=now(),input=str(a.input),argv=sys.argv,
                inherited_from=str(parent),inherited_files=records,
                trace_first=record.get('trace_first',False) if a.validate_export_from else a.trace_first,
                empty_geometry=record.get('empty_geometry',False) if a.validate_export_from else a.empty_geometry,
                parent_precision=record['precision'],precision=a.precision,
                validation_only=bool(a.validate_export_from))
            write_json(a.output/'conversion.json',record)
            phases=['validate'] if a.validate_export_from else ['grounding','validate']
        for phase in phases:
            subprocess.run([sys.executable,__file__,'--phase',phase,'--input',str(a.input),
                '--output',str(a.output),'--threads',str(a.threads),'--precision',a.precision,
                *(['--trace-first'] if a.trace_first else []),
                *(['--empty-geometry'] if a.empty_geometry else [])],check=True,cwd=ROOT)
        return
    tic=time.monotonic()
    if a.phase=='reference':
        a.output.mkdir(parents=True,exist_ok=False)
        info=json.loads((a.input/'input.json').read_text())
        rows=sorted(set([0,info['rows']-1]))
        write_json(a.output/'conversion.json',dict(schema='qn_ego.sam3_openvino.v1',status='preparing',
            started_at=now(),checkpoint_sha256=sha256(checkpoint),prompt='gloved hand',resolution=1008,
            confidence_threshold=.35,source_sha256=info['input_sha256'],rows=rows,
            input=str(a.input),precision=a.precision,training_ready=False,argv=sys.argv,
            reference_only_cpu=True,production_device='GPU',cpu_fallback=False))
    record=json.loads((a.output/'conversion.json').read_text())
    if record['checkpoint_sha256']!=sha256(checkpoint):raise ValueError('Source checkpoint changed')
    if (a.output/(a.phase+'-receipt.json')).exists():raise FileExistsError('Preserve previous phase receipt')
    print('SAM3',a.phase,'START',now(),flush=True)
    if a.phase in ['reference','image','grounding']:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        model=build_sam3_image_model(checkpoint_path=str(checkpoint),load_from_HF=False,
            bpe_path=str(ROOT/'third_party/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz'),
            enable_inst_interactivity=False,device='cpu')
        model.eval()
    if a.phase=='reference':
        processor=Sam3Processor(model,confidence_threshold=.35,device='cpu')
        with torch.inference_mode():
            text=model.backbone.forward_text([record['prompt']],device='cpu')
            np.savez_compressed(a.output/'text-cache.npz',**{k:v.numpy() for k,v in text.items()})
            for row in record['rows']:
                path=a.input/'images'/f'{row:06d}.jpg'
                image=Image.open(path).convert('RGB');t0=time.monotonic()
                state=processor.set_image(image)
                b=state['backbone_out'];b.update(text)
                raw=model.forward_grounding(backbone_out=b,find_input=processor.find_stage,
                    find_target=None,geometric_prompt=model._get_dummy_prompt())
                names=['pred_logits','pred_boxes','pred_masks','presence_logit_dec']
                np.savez_compressed(a.output/f'reference-{row:06d}.npz',
                    **{k:raw[k].numpy() for k in names},height=image.height,width=image.width,
                    image_sha256=sha256(path),seconds=time.monotonic()-t0)
                if row==record['rows'][0]:
                    from torchvision.transforms import v2
                    x=processor.transform(v2.functional.to_image(image)).unsqueeze(0)
                    np.save(a.output/'image-input.npy',x.numpy())
                    np.savez_compressed(a.output/'image-features.npz',
                        **{f'x{i}':v.numpy() for i,v in enumerate(b['backbone_fpn']+b['vision_pos_enc'])})
                print('REFERENCE',row,'seconds',round(time.monotonic()-t0,3),flush=True)
    elif a.phase in ['image','grounding']:
        import openvino as ov
        if a.phase=='image':
            from qn_ego.sam3_export import ImageEncoder
            from ov_sam3_helper import patch_vit_rope
            patch_vit_rope(model.backbone.vision_backbone.trunk)
            wrapper=ImageEncoder(model.backbone)
            del model;gc.collect()
            inputs=torch.from_numpy(np.load(a.output/'image-input.npy'))
            shapes=[[1,3,1008,1008]]
        else:
            from qn_ego.sam3_export import TextOnlyGrounding
            with np.load(a.output/'text-cache.npz') as z:text={k:torch.from_numpy(z[k]) for k in z.files}
            wrapper=TextOnlyGrounding(model,text)
            del model;gc.collect()
            with np.load(a.output/'image-features.npz') as z:inputs=tuple(torch.from_numpy(z[f'x{i}']) for i in range(6))
            shapes=[list(x.shape) for x in inputs]
        wrapper.eval()
        if a.phase=='grounding' and a.trace_first:
            # Pixel decoder checkpointing is enabled even in eval upstream. Its
            # Python _NoopSaveInputs cannot be serialized; no gradients are used.
            disabled=[]
            for name,module in wrapper.named_modules():
                if getattr(module,'act_ckpt',False):
                    module.act_ckpt=False;disabled.append(name)
            print('EXPORT disabled activation checkpointing',disabled,flush=True)
        print('CONVERT',a.phase,'shapes',shapes,flush=True)
        with torch.inference_mode():
            if a.phase=='grounding' and a.trace_first:
                import faulthandler
                faulthandler.enable()
                print('GROUNDING native forward START',flush=True)
                native=wrapper(*inputs)
                print('GROUNDING native forward PASSED',[list(x.shape) for x in native],flush=True)
                if a.empty_geometry:
                    from qn_ego.sam3_export import EmptyGeometryEncoder
                    wrapper.model.geometry_encoder=EmptyGeometryEncoder(wrapper.model.geometry_encoder)
                    specialized=wrapper(*inputs)
                    differences=[float(torch.max(torch.abs(x-y))) for x,y in zip(native,specialized)]
                    for x,y in zip(native,specialized):torch.testing.assert_close(x,y,rtol=1e-5,atol=1e-5)
                    write_json(a.output/'empty-geometry-comparison.json',dict(
                        passed=True,max_abs=differences,scope='Same real image features, all four raw outputs'))
                    print('GROUNDING empty geometry equivalence PASSED',differences,flush=True)
                    del specialized
                print('GROUNDING Torch trace START',flush=True)
                traced=torch.jit.trace(wrapper,inputs,strict=False,check_trace=False)
                actual=traced(*inputs)
                differences=[float(torch.max(torch.abs(x-y))) for x,y in zip(native,actual)]
                print('GROUNDING Torch trace PASSED max_abs',differences,flush=True)
                for x,y in zip(native,actual):torch.testing.assert_close(x,y,rtol=1e-4,atol=1e-4)
                torch.jit.save(traced,a.output/'grounding-traced.pt')
                del wrapper,native,actual,traced;gc.collect()
                print('GROUNDING OpenVINO from saved trace START',flush=True)
                # A reloaded TorchScript graph plus shape-only overrides loses
                # input dtypes in OVC 2026.4, leaving ConvertPromoteTypes nodes
                # unsupported by the Intel GPU. All six original FPN/pos inputs
                # are FP32; make that part of the exported contract explicit.
                converted=ov.convert_model(str(a.output/'grounding-traced.pt'),
                    input=[(ov.PartialShape(shape),ov.Type.f32) for shape in shapes])
            else:
                converted=ov.convert_model(wrapper,example_input=inputs,input=shapes)
        for port in converted.inputs:
            if port.get_element_type()!=ov.Type.f32:
                raise ValueError('Exported SAM3 input is not explicitly FP32')
        unresolved=[node.get_friendly_name() for node in converted.get_ordered_ops()
                    if node.get_type_name()=='ConvertPromoteTypes']
        if unresolved:raise ValueError('Unresolved input types in SAM3 export: '+str(unresolved))
        ov.save_model(converted,a.output/(a.phase+'.xml'),compress_to_fp16=False)
        print('SAVED',a.phase,flush=True)
    else:
        import faulthandler
        # Record the Python boundary during slow first-inference initialization.
        # This is diagnostic only; never changes the requested inference device.
        faulthandler.enable()
        faulthandler.dump_traceback_later(60,repeat=True)
        from qn_ego.openvino_sam3 import UnvalidatedSAM3GPU,postprocess
        processor=UnvalidatedSAM3GPU(a.output,precision=a.precision)
        results=[]
        for row in record['rows']:
            print('VALIDATE reference load',row,flush=True)
            path=a.input/'images'/f'{row:06d}.jpg'
            with np.load(a.output/f'reference-{row:06d}.npz') as z:
                if sha256(path)!=str(z['image_sha256']):raise ValueError('Image changed')
                ref=postprocess({k:z[k] for k in ['pred_logits','pred_boxes','pred_masks','presence_logit_dec']},int(z['height']),int(z['width']),.35)
                reference_s=float(z['seconds'])
            image=Image.open(path).convert('RGB')
            print('VALIDATE GPU inference',row,flush=True)
            t0=time.monotonic();state=processor.set_image(image)
            pred=processor.set_text_prompt('gloved hand',state);seconds=time.monotonic()-t0
            print('VALIDATE compare',row,processor.inference_seconds,flush=True)
            # Match instances by mask IoU (do not depend on query order).
            from scipy.optimize import linear_sum_assignment
            rm=ref['masks'].numpy()[:,0];pm=pred['masks'].numpy()[:,0]
            ious=np.zeros((len(rm),len(pm)))
            for i,m in enumerate(rm):
                for j,p in enumerate(pm):ious[i,j]=(m&p).sum()/max(1,(m|p).sum())
            ri,pi=linear_sum_assignment(-ious)
            matched=[float(ious[i,j]) for i,j in zip(ri,pi)]
            boxmax=max([float(np.max(np.abs(ref['boxes'][i].numpy()-pred['boxes'][j].numpy()))) for i,j in zip(ri,pi)] or [1e9])
            scoremax=max([float(abs(ref['scores'][i]-pred['scores'][j])) for i,j in zip(ri,pi)] or [1e9])
            passed=len(rm)==len(pm)>0 and min(matched,default=0)>=.98 and boxmax<=2 and scoremax<=.01
            item=dict(row=row,reference_instances=len(rm),gpu_instances=len(pm),mask_ious=matched,
                box_max_px=boxmax,score_max_abs=scoremax,reference_cpu_s=reference_s,gpu_s=seconds,
                passed=bool(passed))
            results.append(item);print(json.dumps(item),flush=True)
            # Save QA evidence, not just a scalar assertion.
            np.savez_compressed(a.output/f'gpu-{row:06d}.npz',**{k:v.numpy() for k,v in pred.items()})
        validation=dict(record,status='completed',finished_at=now(),numeric_passed=all(r['passed'] for r in results),
            comparisons=results,device='GPU',execution_devices=processor.execution_devices,
            model_files=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha256(p))
                for p in sorted(a.output.glob('*.xml'))+sorted(a.output.glob('*.bin'))],
            precision=a.precision,scope='Same-weight fixed-text SAM3; real-frame segmentation comparison, not geometric truth')
        write_json(a.output/'validation.json',validation)
        faulthandler.cancel_dump_traceback_later()
        if not validation['numeric_passed']:raise RuntimeError('SAM3 GPU numeric comparison failed; deployment forbidden')
    write_json(a.output/(a.phase+'-receipt.json'),dict(phase=a.phase,status='passed',at=now(),elapsed_s=time.monotonic()-tic,argv=sys.argv))
    print('SAM3',a.phase,'PASSED',round(time.monotonic()-tic,3),flush=True)


if __name__=='__main__':main()
