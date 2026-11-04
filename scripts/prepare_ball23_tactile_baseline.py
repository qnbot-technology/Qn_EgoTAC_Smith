#!/usr/bin/env python3
"""Offline, fixed channel baselines from native packets; preserve raw and time."""
import argparse
import json
from pathlib import Path
import sys
import subprocess

import h5py
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'src'))
from qn_smith.artifacts import now, sha256, write_json
from qn_smith.tactile_baseline import fit_baseline, apply_baseline, residual_regions
from qn_smith.tactile_protocol import finger_strength, palm_strength, PROTOCOL_SHA256


def prepare(source, root, config, config_path, episode=None):
    ep = root/(episode or source.stem)
    dest = ep/'tactile-baseline-v1'
    dest.mkdir(exist_ok=True)
    report = json.loads((ep/'trajectory-report.json').read_text())
    if sha256(source) != report['source_sha256']:
        raise ValueError('source SHA mismatch')
    if config['method'] != 'estimated_lower_tail':
        raise ValueError('This batch has no operator-confirmed unloaded reference; use estimated display mode')
    kwargs = {k: config[k] for k in ('method','quantile','lower_tail_quantile','min_deadband','noise_multiplier','min_samples')}
    summaries = {}; models = {}; output = {}; traces = {}
    with np.load(ep/'trajectory-and-tactile.npz') as z, h5py.File(source, 'r') as f:
        for k in ('source_row', 'sample_time_s', 'camera_receive_ns'):
            output[k] = z[k]
        for side in ('left','right'):
            g = f[f'raw/tactile_emf/{side}/tactile/native']
            raw = g['values_u8'][:].reshape(-1,460)
            valid = g['valid'][:].ravel().astype(bool)
            ns = g['host_receive_monotonic_ns'][:].ravel()
            if np.any(np.diff(ns) < 0): raise ValueError('nonmonotonic native tactile receive time')
            model = fit_baseline(raw, valid, **kwargs)
            signed, positive = apply_baseline(raw, valid, model['baseline_count'], model['deadband_count'])
            q = np.quantile(raw[valid].astype(float), [0,.1,.2,.5,.9,1],axis=0)
            np.savez_compressed(dest/f'native-residual-{side}.npz', raw_count=raw, valid=valid,
                native_source_index=np.arange(len(raw)), receive_monotonic_ns=ns,
                signed_residual_count=signed, positive_residual_count=positive,
                channel_quantiles=q, channel_quantile_levels=[0,.1,.2,.5,.9,1],
                baseline_count=model['baseline_count'], noise_proxy_count=model['noise_proxy_count'],
                deadband_count=model['deadband_count'], reference_packet_mask=model['reference_packet_mask'])
            ix=z[side+'_tactile_index'];fv=z[side+'_tactile_valid']
            np.testing.assert_array_equal(raw[ix],z[side+'_tactile_values'])
            fs,fp=apply_baseline(raw[ix],fv,model['baseline_count'],model['deadband_count'])
            peaks,means=residual_regions(fp,fv,side)
            raw_f,raw_fm=finger_strength(raw[ix],fv,side);raw_p,raw_pm=palm_strength(raw[ix],fv,side)
            raw_regions=np.c_[raw_f,raw_p]
            for suffix in ('_tactile_index','_tactile_valid','_tactile_delta_ms'):
                output[side+suffix]=z[side+suffix]
            output.update({side+'_tactile_values':raw[ix], side+'_signed_residual_count':fs,
                side+'_positive_residual_count':fp,side+'_baseline_count':model['baseline_count'],
                side+'_noise_proxy_count':model['noise_proxy_count'],side+'_deadband_count':model['deadband_count'],
                side+'_finger_peak_residual_count':peaks[:,:5],side+'_finger_mean_residual_count':means[:,:5],
                side+'_palm_peak_residual_count':peaks[:,5],side+'_palm_mean_residual_count':means[:,5]})
            models[side]={k:v for k,v in model.items() if k!='reference_packet_mask'}
            models[side].update(native_hdf5_group=g.name, native_packets=len(raw),
                reference_indices_file=f'native-residual-{side}.npz',
                selected_receive_ns_range=[int(ns[valid][0]),int(ns[valid][-1])])
            summaries[side]=dict(native_packets=len(raw),native_valid_packets=int(valid.sum()),
                valid_rgb_frames=int(fv.sum()),invalid_rgb_frames=int((~fv).sum()),
                nonzero_baseline_channels=int((model['baseline_count']>0).sum()),
                constant_channels=int((q[0]==q[-1]).sum()),
                constant_positive_channels=int(((q[0]==q[-1])&(q[0]>0)).sum()),
                channel_saturation_fraction=(raw[valid]==255).mean(0),
                channel_constant_mask=q[0]==q[-1],
                constant_channel_interpretation='observed constant during this episode; not proof of a dead sensor',
                raw_region_peak_q10_median_q90=np.nanquantile(raw_regions,[.1,.5,.9],axis=0),
                corrected_region_peak_q10_median_q90=np.nanquantile(peaks,[.1,.5,.9],axis=0),
                corrected_region_zero_fraction=np.mean(peaks[fv]==0,axis=0),
                negative_channel_fraction=np.mean(fs[fv]<0,axis=0))
            traces[side+'_raw_regions']=raw_regions
            traces[side+'_residual_regions']=peaks
    np.savez_compressed(dest/'frame-residual.npz',**output)
    metadata=dict(at=now(),source=str(source),source_sha256=report['source_sha256'],
        argv=sys.argv, config_path=str(config_path),config_sha256=sha256(config_path),config=config,
        module_sha256=sha256(PROJECT/'src/qn_smith/tactile_baseline.py'),
        script_sha256=sha256(__file__),protocol_sha256=PROTOCOL_SHA256,models=models,
        units='raw count difference, not Newtons',
        formula='signed=x-b; positive=max(signed-deadband,0); region=max/mean(positive channels)',
        baseline_estimated=True,no_load_confirmed=False,fixed_over_time=True,
        offline_uses_future_samples=True,raw_hdf5_changed=False,
        limitation='constant contact/preload throughout recording can be subtracted; posture-dependent preload and drift may remain',
        training_ready=False)
    write_json(dest/'baseline-model.json',metadata)
    write_json(dest/'diagnostics.json',dict(at=now(),regions=['thumb','index','middle','ring','pinky','palm'],sides=summaries))
    np.savez_compressed(dest/'region-traces.npz',sample_time_s=output['sample_time_s'],**traces)
    # Existing isolated system NumPy/matplotlib; do not change shared environments.
    plot_argv=[sys.executable,'-I',str(PROJECT/'scripts/plot_tactile_residual_traces.py'),str(dest),source.stem]
    subprocess.run(plot_argv,check=True)
    write_json(dest/'plot-call.json',dict(argv=plot_argv,at=now(),exit_code=0,
        reason='Standalone environment and explicit plotting dependency; no shared project environment'))
    print(source.stem,json.dumps({s:{k:v for k,v in d.items() if k in ('native_packets','nonzero_baseline_channels','constant_positive_channels')} for s,d in summaries.items()}),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,nargs='+',required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--episode', help='Stable source/session name for MCAP; defaults to source stem')
    ap.add_argument('--config',type=Path,default=PROJECT/'configs/tactile-baseline.json')
    a=ap.parse_args();config=json.loads(a.config.read_text())
    for source in a.source: prepare(source.resolve(),a.output.resolve(),config,a.config.resolve(),a.episode)


if __name__=='__main__': main()
