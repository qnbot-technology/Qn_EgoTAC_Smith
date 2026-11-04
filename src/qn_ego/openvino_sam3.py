"""Explicit Intel GPU SAM3. No CPU neural-network inference or fallback.

PIL/Torch tensor resizing and thresholding remain CPU preprocessing/postprocessing.
The fixed text was encoded once at export and is a constant in the grounding IR.
"""
import json
from pathlib import Path
import time

import numpy as np
import torch


def postprocess(raw, height, width, threshold):
    from torch.nn import functional as F
    tensors={k:torch.from_numpy(np.array(v,copy=True)) for k,v in raw.items()}
    scores=(tensors['pred_logits'].sigmoid()*tensors['presence_logit_dec'].sigmoid().unsqueeze(1)).squeeze(-1)
    keep=scores>threshold
    probs=scores[keep];boxes=tensors['pred_boxes'][keep];masks=tensors['pred_masks'][keep]
    xyxy=torch.cat([boxes[:,:2]-boxes[:,2:]/2,boxes[:,:2]+boxes[:,2:]/2],dim=-1)
    xyxy*=torch.tensor([width,height,width,height],dtype=xyxy.dtype)
    if len(masks):
        logits=F.interpolate(masks.unsqueeze(1),size=(height,width),mode='bilinear',align_corners=False).sigmoid()
    else:
        logits=torch.empty((0,1,height,width),dtype=torch.float32)
    return dict(masks=logits>.5,masks_logits=logits,boxes=xyxy,scores=probs)


class UnvalidatedSAM3GPU:
    """Low-level converter/diagnostic runtime; pipeline uses asset-checking subclasses."""
    def __init__(self, directory, precision='f32', image_activation_scale_factor=None):
        import openvino as ov
        from torchvision.transforms import v2
        self.directory=Path(directory).resolve()
        conversion=json.loads((self.directory/'conversion.json').read_text())
        if conversion['resolution']!=1008 or conversion['prompt']!='gloved hand':
            raise ValueError('Unsupported SAM3 prompt/resolution')
        self.prompt=conversion['prompt'];self.threshold=conversion['confidence_threshold']
        core=ov.Core()
        if 'GPU' not in core.available_devices:raise RuntimeError('Intel GPU unavailable; no CPU fallback')
        options={'PERFORMANCE_HINT':'LATENCY','INFERENCE_PRECISION_HINT':precision,'NUM_STREAMS':'1'}
        self.models=[];self.requests=[];self.execution_devices={}
        self.inference_seconds={}
        for name in ['image','grounding']:
            print('COMPILE SAM3 GPU',name,flush=True);tic=time.monotonic()
            stage_options=dict(options)
            if name=='image' and image_activation_scale_factor is not None:
                if not np.isfinite(image_activation_scale_factor) or image_activation_scale_factor<=0:
                    raise ValueError('Activation scale must be finite and positive')
                stage_options['ACTIVATIONS_SCALE_FACTOR']=float(image_activation_scale_factor)
            compiled=core.compile_model(str(self.directory/(name+'.xml')),'GPU',stage_options)
            devices=list(compiled.get_property('EXECUTION_DEVICES'))
            if not devices or any(not str(d).startswith('GPU') for d in devices):
                raise RuntimeError('SAM3 compiled outside requested GPU: '+str(devices))
            self.execution_devices[name]=devices
            self.models.append(compiled);self.requests.append(compiled.create_infer_request())
            print('COMPILED SAM3 GPU',name,round(time.monotonic()-tic,3),'s',flush=True)
        self.transform=v2.Compose([v2.ToDtype(torch.uint8,scale=True),v2.Resize(size=(1008,1008)),
            v2.ToDtype(torch.float32,scale=True),v2.Normalize(mean=[.5]*3,std=[.5]*3)])

    def infer(self,index,inputs):
        tic=time.monotonic();request=self.requests[index]
        name=['image','grounding'][index]
        if name not in self.inference_seconds:
            print('FIRST SAM3 GPU inference START',name,flush=True)
        request.infer({i:np.ascontiguousarray(x) for i,x in enumerate(inputs)})
        outputs=[request.get_output_tensor(i).data.copy() for i in range(len(self.models[index].outputs))]
        if any(not np.isfinite(x).all() for x in outputs):
            raise RuntimeError('Nonfinite GPU '+name+' output; refusing invalid segmentation')
        if name not in self.inference_seconds:
            print('FIRST SAM3 GPU inference DONE',name,round(time.monotonic()-tic,3),'s',flush=True)
        self.inference_seconds[name]=time.monotonic()-tic
        return outputs

    def set_image(self,image,state=None):
        from torchvision.transforms import v2
        state={} if state is None else state
        state['original_height']=image.height;state['original_width']=image.width
        x=self.transform(v2.functional.to_image(image)).unsqueeze(0).numpy()
        state['image_features']=self.infer(0,[x])
        return state

    def set_text_prompt(self,prompt,state):
        if prompt!=self.prompt:raise ValueError('SAM3 IR only accepts the validated fixed prompt')
        result=self.infer(1,state['image_features'])
        raw=dict(zip(['pred_logits','pred_boxes','pred_masks','presence_logit_dec'],result))
        state.update(postprocess(raw,state['original_height'],state['original_width'],self.threshold))
        return {k:state[k] for k in ['masks','masks_logits','boxes','scores']}


class OpenVINOSAM3Processor(UnvalidatedSAM3GPU):
    def __init__(self,directory,checkpoint):
        from qn_ego.sam3_assets import validate_sam3_assets
        receipt=validate_sam3_assets(directory,checkpoint)
        self.receipt=receipt
        super().__init__(directory,precision=receipt['precision'])


class OpenVINOEfficientSAM3Processor(UnvalidatedSAM3GPU):
    """Opt-in full-pipeline candidate; numeric QA status is never rewritten."""
    def __init__(self,directory,checkpoint):
        from qn_ego.sam3_assets import validate_efficientsam3_assets
        import warnings
        self.receipt=validate_efficientsam3_assets(directory,checkpoint)
        warnings.warn('Explicit EfficientSAM3 selection: experimental segmentation; '
            'numeric_validation_passed='+str(self.receipt['numeric_passed']),RuntimeWarning)
        super().__init__(directory,precision=self.receipt['precision'],
            image_activation_scale_factor=self.receipt.get('image_activation_scale_factor'))
