#!/usr/bin/env python3
"""Create an explicitly labelled contiguous diagnostic derivative; never edit raw HDF5."""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--first-row',type=int,required=True);p.add_argument('--frames',type=int,default=16)
a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
before=digest(a.input)
with h5py.File(a.input,'r') as src:
    n=len(src['time/relative_s']);lo=a.first_row;hi=lo+a.frames
    if src.attrs.get('complete')!=1 or not 0<=lo<hi<=n:raise ValueError('Invalid completed input or range')
    with h5py.File(a.output,'x') as dst:
        for k,v in src.attrs.items():dst.attrs[k]=v
        paths=[]
        def copy(name,obj):
            if isinstance(obj,h5py.Group):
                g=dst.require_group(name)
                for k,v in obj.attrs.items():g.attrs[k]=v
            elif isinstance(obj,h5py.Dataset):
                frame_dataset=(name.startswith('time/') or (name.startswith('raw/cameras/head_rgb/') and '/imu/' not in name))
                if frame_dataset and obj.shape and obj.shape[0]==n:
                    d=dst.create_dataset(name,data=obj[lo:hi],dtype=obj.dtype)
                    for k,v in obj.attrs.items():d.attrs[k]=v
                    paths.append(name)
                else:src.copy(obj,dst.require_group(str(Path(name).parent)),name=Path(name).name)
        src.visititems(copy)
        dst.attrs['deployment_diagnostic_derivative']=True
        dst.attrs['source_sha256']=before
        dst.attrs['source_original_path']=str(a.input.resolve())
        dst.attrs['source_first_row_0based']=lo
        dst.attrs['source_last_row_0based']=hi-1
        dst.attrs['diagnostic_note']='Contiguous RGB/time subset; native asynchronous streams and calibrations retained. Not the original recording or full-video acceptance.'
if digest(a.input)!=before:raise RuntimeError('Source changed')
record=dict(source=str(a.input.resolve()),source_sha256=before,output=str(a.output.resolve()),
    output_sha256=digest(a.output),first_row_0based=lo,last_row_0based=hi-1,frames=a.frames,
    frame_datasets_sliced=paths,asynchronous_streams_retained=True,original_unchanged=True,diagnostic_only=True)
a.output.with_suffix('.provenance.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(record,ensure_ascii=False,indent=2))
