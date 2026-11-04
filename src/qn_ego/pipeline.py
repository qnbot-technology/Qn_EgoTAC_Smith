"""Explicit source-bound stages, isolated subprocesses and resumable receipts."""
import argparse
import atexit
from collections import Counter
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
os.environ['PATH'] = str(ROOT/'runtime/media/bin') + os.pathsep + os.environ.get('PATH','')
SIDES = ('left', 'right')


class EmfDataLossError(RuntimeError):
    """Compact, actionable failure for invalid EMF/native glove packets.

    The child stage keeps its full traceback in the stage log.  This exception
    lets the CLI/platform surface only the packet identity and missing-data
    reason instead of repeating a long Python traceback to the operator.
    """


def _emf_loss_marker(log_path):
    try:
        text = Path(log_path).read_text(errors='replace')
    except OSError:
        return None
    for line in reversed(text.splitlines()):
        marker = 'EMF_DATA_LOSS:'
        if marker in line:
            return line.split(marker, 1)[1].strip()
    return None


def resolve_tag_frame_workers(*, is_mcap, search_mode, requested, threads):
    """Return the actual full-frame worker count recorded in the recipe.

    ROI has causal state and is always serial. Full-frame HDF5 uses a small
    bounded pool because its JPEG rows are already local. MCAP automatic mode
    reserves two CPUs for the recorder and system services, capped at six;
    an explicit positive value can still request eight workers offline.
    """
    requested = int(requested)
    if requested < 0:
        raise ValueError('tag frame workers must be non-negative (0=automatic)')
    if search_mode == 'roi':
        return 1
    if requested:
        return requested
    cpu_budget = max(1, int(threads))
    machine_budget = max(1, (os.cpu_count() or 1) - 2)
    return min(6, cpu_budget, machine_budget) if is_mcap else min(6, cpu_budget)

def read(p):
    return json.loads(Path(p).read_text())

def load_config(path):
    config = read(path)
    if config['schema'] != 'qn_ego.pipeline.v1':
        raise ValueError('Unsupported pipeline configuration')
    return {k: (ROOT / v).resolve() if k != 'schema' else v for k, v in config.items()}

def path_record(p):
    from qn_smith.artifacts import sha256
    p = Path(p)
    return dict(path=str(p), bytes=p.stat().st_size, sha256=sha256(p))

def env_for(gpu, device='cuda', threads=4):
    env = dict(os.environ, PYTHONNOUSERSITE='1', OMP_NUM_THREADS=str(threads),
               OPENBLAS_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               QN_EGO_THREADS=str(threads), QN_EGO_DEVICE=device,
               CUDA_VISIBLE_DEVICES=str(gpu) if device=='cuda' else '',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               PYTHONPATH=os.pathsep.join(str(ROOT/p) for p in ['src','scripts','runtime/python_startup']),
               QN_EGO_MANO_ROOT=str(ROOT/'weights/mano'),
               TORCH_HOME=str(ROOT/'.cache/torch'))
    return env

class Journal:
    def __init__(self, run, resume, env):
        self.run, self.resume, self.env = run, resume, env
        (run/'calls').mkdir(parents=True, exist_ok=True)

    def call(self, name, argv, outputs):
        from qn_smith.artifacts import now, sha256, write_json
        receipt = self.run/'calls'/(name+'.json')
        argv = list(map(str, argv))
        signature = dict(argv=argv, script_sha256=sha256(argv[1]) if len(argv)>1 and Path(argv[1]).is_file() else None)
        if receipt.exists():
            old = read(receipt)
            if old['status'] == 'passed' and self.resume:
                if old['signature'] != signature:
                    raise ValueError('Resume command/code changed: '+name)
                for rec in old['outputs']:
                    if path_record(rec['path']) != rec:
                        raise ValueError('Resume output changed: '+rec['path'])
                print('SKIP verified', name, flush=True)
                return
            raise RuntimeError('Existing unfinished/failed stage: '+name+'; preserve it and use a new output run')
        rec = dict(stage=name, argv=argv, cwd=str(ROOT), signature=signature,
                   started_at=now(), status='running')
        write_json(receipt, rec)
        print('START', name, flush=True)
        tic = time.monotonic()
        try:
            with (self.run/'calls'/(name+'.log')).open('x') as stream:
                audit = self.run/'calls'/(name+'-imports.json')
                env = dict(self.env, QN_EGO_IMPORT_AUDIT=str(audit))
                result = subprocess.run(argv, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            rec['exit_code'] = result.returncode
            if result.returncode:
                log_path = self.run/'calls'/(name+'.log')
                loss = _emf_loss_marker(log_path)
                if loss:
                    raise EmfDataLossError('EMF_DATA_LOSS: ' + loss)
                raise RuntimeError(name+' failed; see '+str(log_path))
            if not audit.exists() or not read(audit)['passed']:
                raise RuntimeError('Independent source import audit failed: '+str(audit))
            rec['outputs'] = [path_record(p) for p in [*outputs,audit]]
            rec.update(status='passed', finished_at=now(), elapsed_s=time.monotonic()-tic)
            write_json(receipt, rec)
        except BaseException as exc:
            rec.update(status='failed', finished_at=now(), elapsed_s=time.monotonic()-tic, error=str(exc))
            write_json(receipt, rec)
            raise
        print('DONE', name, round(rec['elapsed_s'], 2), 's', flush=True)

def verify_weights(deep=False):
    from qn_smith.artifacts import sha256
    path = ROOT/'weights/manifest.json'
    if not path.exists():
        raise FileNotFoundError('Model manifest missing; complete model delivery first')
    records = read(path)
    for rec in records:
        p = ROOT/rec['path']
        if not p.is_file() or p.is_symlink() or p.stat().st_size != rec['bytes']:
            raise ValueError('Missing/incomplete/linked model '+str(p))
        if deep and sha256(p) != rec['sha256']:
            raise ValueError('Model SHA mismatch '+str(p))
    return dict(passed=True, files=len(records), bytes=sum(x['bytes'] for x in records), sha256_checked=deep)

def doctor(config, full, deep):
    from qn_smith.artifacts import sha256
    result = dict(root=str(ROOT), python=sys.executable, dependencies={}, tools={})
    for name in ['numpy', 'scipy', 'cv2', 'h5py', 'PIL', 'matplotlib']:
        spec=importlib.util.find_spec(name)
        result['dependencies'][name]=spec is not None
    for name in ['ffmpeg','ffprobe']:
        result['tools'][name]=shutil.which(name)
    for key in ['humanv2','layout','mount','registration']:
        if not config[key].exists(): raise FileNotFoundError(config[key])
    if full: result['weights']=verify_weights(deep)
    result['passed']=all(result['dependencies'].values()) and all(result['tools'].values())
    if not result['passed']: raise RuntimeError(json.dumps(result,ensure_ascii=False))
    return result

def freeze(run, episode, mode, mapping_subdir='visual/fused-stable-v3'):
    from qn_smith.artifacts import now,sha256,write_json,verify_manifest
    from qn_smith.tactile_display import PROFILES,prepare_display
    import numpy as np
    base=run/'baseline'/episode;prior=base/'tag3-tag16-handmap-residual-v4'
    folder=run/mapping_subdir/episode
    source=prior/'tactile-display.npz';dest=folder/'tactile-display-profile.npz'
    with np.load(source) as z: tactile=prepare_display(dict(z),PROFILES['D'])
    np.savez_compressed(dest,**tactile)
    paths=[folder/'fused-humanv2.npz',folder/'fusion-report.json',dest,
           prior/'bone-projections.npz',base/'camera.json',base/'trajectory-and-tactile.npz',
           base/'trajectory-report.json',prior/'applied-camera-trajectory.npz',source,
           run/'provenance/input.json',run/'provenance/registration.json']
    records=[dict(path=p.relative_to(run).as_posix(),bytes=p.stat().st_size,sha256=sha256(p)) for p in paths]
    write_json(run/'processing-manifest.json',records)
    errors=verify_manifest(run,records)
    write_json(run/'processing-verification.json',dict(passed=not errors,errors=errors,at=now(),
        manifest_sha256=sha256(run/'processing-manifest.json'),visual_mode=mode))
    if errors:raise ValueError(errors)

def run_pipeline(a):
    import h5py
    import numpy as np
    from qn_smith.artifacts import now,sha256,write_json,verify_manifest
    config=load_config(a.config)
    # The 26 mm CAD relation is an isolated MCAP calibration. HDF5 keeps its
    # validated 23 mm layout/mount. MCAP now follows the HDF5 physical tag
    # order by default; an old dataset-specific numbering can only be replayed
    # through the explicit --mcap-right-tag-start override.
    input_suffix = Path(a.input).suffix.lower()
    is_mcap = input_suffix == '.mcap'
    mcap_offset_profile = getattr(a, 'mcap_offset_profile', 'ball23-h5') if is_mcap else None
    if is_mcap and mcap_offset_profile not in ('ball23-h5', 'ball26-new'):
        raise ValueError('mcap offset profile must be ball23-h5 or ball26-new')
    mcap_depth_mode = getattr(a, 'mcap_depth_mode', 'raw') if is_mcap else None
    if is_mcap and mcap_depth_mode not in ('raw', 'metadata'):
        raise ValueError('mcap depth mode must be raw or metadata')
    if is_mcap:
        if mcap_offset_profile == 'ball26-new':
            config['layout'] = (ROOT/'configs/ball26/layout-26mm-mcap-temporary.json').resolve()
            config['mount'] = (ROOT/'configs/ball26/mount-26mm-new.json').resolve()
            config['registration'] = (ROOT/'configs/ball26/registration-26mm-new-template.json').resolve()
        else:
            # The MX HDF5 relation is the reviewed baseline. MCAP may opt in
            # to the newer 26mm CAD candidate for a separately reviewed run.
            config['layout'] = (ROOT/'configs/ball23/layout-23mm-roi-candidate.json').resolve()
            config['mount'] = (ROOT/'configs/ball23/mount.json').resolve()
            config['registration'] = (ROOT/'configs/ball23/registration-template.json').resolve()
    from qn_ego.sam_selection import resolve_sam
    selection=resolve_sam(getattr(a,'sam_model','sam3'),a.sam_backend,a.device,a.sam_openvino_dir,config,ROOT)
    a.sam_model=selection['model'];a.sam_openvino_dir=selection['directory']
    if a.hawor_backend=='openvino' and (a.device!='cpu' or a.openvino_model is None):
        raise ValueError('OpenVINO requires explicit --device cpu --openvino-model PATH')
    if a.sam_backend=='openvino' and (a.device!='cpu' or a.sam_openvino_dir is None):
        raise ValueError('SAM3 OpenVINO GPU requires --device cpu --sam-openvino-dir PATH')
    output=a.output.resolve();input_source=a.input.resolve()
    if output.exists() and any(output.iterdir()) and not a.resume:
        raise ValueError('Use an empty output directory or --resume')
    output.mkdir(parents=True,exist_ok=True)
    lock=(output/'.pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Input/calibration/model identity is frozen before any stage can be reused.
    source_identity=path_record(input_source)
    is_mcap=input_source.suffix.lower()=='.mcap'
    adapter_reference=Path(a.mcap_reference).resolve() if getattr(a, 'mcap_reference', None) else ROOT/'configs/mcap/seven_gesture_reference.json'
    if is_mcap and not adapter_reference.is_file():
        raise FileNotFoundError(f'MCAP HumanV2 reference calibration not found: {adapter_reference}')
    adapter_status=('explicit_recording_reference' if getattr(a, 'mcap_reference', None) else 'bundled_reference_unverified') if is_mcap else None
    # Platform processing runs bind recording-specific calibration beside the
    # supplied reference.json.  Older recordings may still carry a sibling
    # calibration directory, so prefer the bound run input when present and
    # keep the source recording read-only.
    adapter_calibration_dir = (input_source.parent/'calibration') if is_mcap else None
    if is_mcap and not adapter_calibration_dir.is_dir():
        bound_calibration = adapter_reference.parent/'calibration'
        if bound_calibration.is_dir():
            adapter_calibration_dir = bound_calibration
    adapter=dict(format='mcap_qnbot_daq_v1',enabled=is_mcap,reference=path_record(adapter_reference) if is_mcap else None,
                 humanv2_calibration_status=adapter_status,
                 calibration_dir=str(adapter_calibration_dir.resolve()) if is_mcap else None,
                 synchronization='affine board-ms to camera host-ns interval mapping; training_ready=false')
    mcap_right_tag_start = int(getattr(a, 'mcap_right_tag_start', 13)) if is_mcap else 13
    if mcap_right_tag_start < 0:
        raise ValueError('mcap right tag start must be non-negative')
    requested_tag_workers = int(getattr(a, 'tag_frame_workers', 0))
    # Full-frame Tag solves are independent and deterministic because results
    # are consumed in source-row order. Both formats use up to six workers;
    # MCAP automatic mode reserves two CPUs for the live recorder. ROI search
    # remains causal and the processing script forces it back to one.
    tag_frame_workers = resolve_tag_frame_workers(
        is_mcap=is_mcap, search_mode=a.mcap_tag_search,
        requested=requested_tag_workers, threads=a.threads)
    mcap_cache_arg = Path(a.mcap_cache).resolve() if is_mcap and getattr(a, 'mcap_cache', None) else None
    recipe=dict(source=source_identity,input_format='mcap_qnbot_daq_v1' if is_mcap else 'qnbot_ego_human_hdf5_v2',
        adapter=adapter,config={k:str(v) for k,v in config.items()},
        config_files={k:path_record(v) for k,v in config.items() if isinstance(v,Path) and v.is_file()},
        start_s=a.start_s,end_s=a.end_s,visual=a.visual,visual_assist=bool(getattr(a, 'visual_assist', True)),
        render_video=bool(getattr(a, 'render_video', False)),gpu=a.gpu,device=a.device,threads=a.threads,
        mcap_offset=mcap_offset_profile,
        mcap_depth_mode=mcap_depth_mode,
        tag_search=(dict(mode=a.mcap_tag_search, margin_px=180, refresh_frames=30,
                         frame_workers=tag_frame_workers)
                    if is_mcap else dict(mode='full', frame_workers=tag_frame_workers)),
        tag_binding=dict(left_start=0, right_start=mcap_right_tag_start,
                         status=('explicit_dataset_override' if is_mcap and mcap_right_tag_start != 13 else ('canonical_h5_binding' if is_mcap else 'legacy_compatibility')),
                         bound_local_ids=list(range(10)),
                         declared_right_ids=list(range(mcap_right_tag_start, mcap_right_tag_start+12)) if is_mcap else list(range(13,25))),
        mcap_cache=path_record(mcap_cache_arg) if mcap_cache_arg else None,
        hawor_backend=a.hawor_backend,openvino_device=a.openvino_device,
        openvino_model=path_record(a.openvino_model) if a.openvino_model else None,
        openvino_validation=path_record(a.openvino_model.with_name('validation.json')) if a.openvino_model else None,
        sam_backend=a.sam_backend,
        sam_model=a.sam_model,sam_model_status='experimental' if selection['experimental'] else 'baseline',
        sam_model_checkpoint=path_record(selection['checkpoint']) if a.visual=='full' and getattr(a, 'visual_assist', True) else None,
        sam_openvino_dir=str(a.sam_openvino_dir.resolve()) if a.sam_openvino_dir else None,
        sam_openvino_validation=path_record(a.sam_openvino_dir/'validation.json') if a.sam_openvino_dir else None,
        sam_openvino_conversion=path_record(a.sam_openvino_dir/'conversion.json') if a.sam_openvino_dir else None,
        sam_python=str(a.sam_python or sys.executable),mano_python=str(a.mano_python or sys.executable))
    if (output/'request.json').exists():
        if read(output/'request.json')!=recipe:raise ValueError('Resume request/input/config identity changed')
    else:write_json(output/'request.json',recipe)
    source=input_source
    ephemeral_mcap_view=None
    if is_mcap:
        from qn_ego.mcap_adapter import materialize_mcap_view, remove_mcap_view
        # Native MCAP entry point. The canonical read view is a derived cache
        # bound to this source/reference/calibration identity. Reusing it on
        # resume avoids a second full MCAP zstd/protobuf pass; it is never the
        # source identity or a delivered converted dataset.
        cache = mcap_cache_arg or output/'provenance/mcap-canonical.view'
        cached_meta = (cache.parent/'mcap-adapter.json') if mcap_cache_arg else output/'provenance/mcap-adapter.json'
        cached = read(cached_meta) if cache.is_file() and cached_meta.is_file() else None
        same_cache = bool(cached and cached.get('input_sha256') == source_identity['sha256']
                          and cached.get('humanv2_reference_sha256') == path_record(adapter_reference)['sha256']
                          and cached.get('depth_mode', 'raw') == mcap_depth_mode
                          and cached.get('persisted_view') is True
                          and cached.get('view_sha256') == __import__('qn_ego.mcap_adapter', fromlist=['sha256']).sha256(cache))
        if same_cache:
            source, adapter_result = cache, cached
            print('REUSE MCAP canonical cache', cache, flush=True)
        else:
            source, adapter_result = materialize_mcap_view(
                input_source, reference_json=adapter_reference,
                calibration_dir=adapter_calibration_dir, output_path=cache,
                source_sha256=source_identity['sha256'], depth_mode=mcap_depth_mode)
        adapter_result.update(input_source=str(input_source), input_sha256=source_identity['sha256'],
                              native_input=True, persisted_view=True,
                              cache_path=str(cache), cache_reused_from=str(cache) if mcap_cache_arg else None)
        write_json(output/'provenance/mcap-adapter.json',adapter_result)
    stop_after_tag=getattr(a,'stop_after_apriltag',False)
    # A pause is an execution boundary, not a different immutable processing recipe.
    # Do not import/check visual runtimes before the caller has opted in.
    assist=getattr(a,'visual_assist',True)
    render_video=bool(getattr(a, 'render_video', False))
    preflight=doctor(config,a.visual=='full' and not stop_after_tag and assist,False);write_json(output/'preflight.json',preflight)
    env=env_for(a.gpu,a.device,a.threads);j=Journal(output,a.resume,env);py=sys.executable
    def cmd(script,*args):return [py,ROOT/'scripts'/script,*args]
    episode=(input_source.parent.name if is_mcap else source.stem);base=output/'baseline'/episode;visual=output/'visual'
    j.call('01_read',cmd('prepare_ball23_recording.py','--source',source,'--run',output,
           '--registration-template',config['registration']),[output/'provenance/input.json',output/'provenance/registration.json'])
    with h5py.File(source,'r') as f:times=f['time/relative_s'][:].ravel()
    start=0 if a.start_s is None else int(np.argmin(abs(times-a.start_s)))
    stop=len(times) if a.end_s is None else int(np.argmin(abs(times-a.end_s)))+1
    if (a.start_s is not None and not times[0]<=a.start_s<=times[-1]) or (a.end_s is not None and not times[0]<=a.end_s<=times[-1]) or stop<=start:
        raise ValueError('Requested render time range is outside source or reversed')
    write_json(output/'provenance/focus-selection.json',dict(first_frame_1based=start+1,last_frame_1based=stop,
        frames=stop-start,requested_start_s=a.start_s,requested_end_s=a.end_s,
        selected_sample_times_s=[float(times[start]),float(times[stop-1])],
        full_timeline_inference=True,selection='nearest RGB source sample, inclusive endpoints'))
    args=['--source',source,'--output',output/'baseline','--layout',config['layout'],'--mount',config['mount'],
          '--package',config['humanv2'],'--workers','1','--episode',episode]
    args += ['--right-tag-start', str(mcap_right_tag_start), '--frame-workers', str(tag_frame_workers)]
    if is_mcap and a.mcap_tag_search == 'roi':
        args += ['--roi', '--roi-margin', '180', '--roi-refresh', '30']
    j.call('02_apriltag',cmd('process_ball23_humanv2_tactile.py',*args,'--stage','trajectory'),[base/'trajectory-and-tactile.npz',base/'configuration.json',base/'detections-and-poses.jsonl'])
    from qn_ego.tag_checkpoint import tag_checkpoint
    checkpoint=tag_checkpoint(output,episode,source_identity,start,stop)
    write_json(output/'apriltag-checkpoint.json',checkpoint)
    if stop_after_tag:
        print(json.dumps(checkpoint,ensure_ascii=False,indent=2),flush=True)
        if ephemeral_mcap_view:
            remove_mcap_view(ephemeral_mcap_view)
        return checkpoint
    # Pause does not freeze this decision; the first continuation does. Later
    # resumes cannot mix a Tag-only output with a visually assisted output.
    plan=dict(visual_assist_enabled=assist,mode=a.visual if assist else 'tag-only',render_video=render_video)
    plan_path=output/'execution-plan.json'
    if plan_path.exists() and read(plan_path)!=plan:
        raise ValueError('Visual assistance decision changed after continuation; use a new RUN')
    if not plan_path.exists():write_json(plan_path,plan)
    if a.visual=='full' and assist:
        for mode, interpreter in [('sam',a.sam_python or py),('mano',a.mano_python or py)]:
            runtime=[str(interpreter),ROOT/'scripts/check_visual_runtime.py',
                '--mode',mode,'--output',output/('runtime-'+mode+'.json')]
            if mode=='sam' and a.sam_backend=='openvino':
                runtime.extend(['--backend','openvino','--openvino-dir',a.sam_openvino_dir.resolve(),
                    '--checkpoint',selection['checkpoint'],'--sam-model',a.sam_model])
            j.call('00_runtime_'+mode,runtime,[output/('runtime-'+mode+'.json')])
    for side in SIDES:
        folder=base/('native-'+side)
        j.call('03_humanv2_'+side,cmd('run_human_hand_v2.py','--input-npz',folder/'prepared-input.npz',
            '--output',folder,'--package',config['humanv2'],'--side',side,
            '--config',output/'provenance'/('embedded-seven-gesture-'+side+'.yaml')),
            [folder/'native-output.npz',folder/'native-report.json',folder/'input-poses.npz'])
    j.call('04_tactile',cmd('prepare_ball23_tactile_baseline.py','--source',source,'--output',output/'baseline',
            '--config',config['tactile_baseline'],'--episode',episode),[base/'tactile-baseline-v1/frame-residual.npz',base/'tactile-baseline-v1/baseline-model.json'])
    prior=base/'tag3-tag16-handmap-residual-v4'
    j.call('05_mapping',cmd('process_ball23_humanv2_tactile.py',*args,'--stage','render',
            '--mechanical-registration',output/'provenance/registration.json',
            '--render-subdir',prior.name,'--tactile-display','hands','--tactile-signal','residual',
            '--tactile-residual-subdir','tactile-baseline-v1','--prepare-only'),
            [prior/'bone-projections.npz',prior/'applied-camera-trajectory.npz',prior/'tactile-display.npz'])
    mapping_subdir='visual/fused-stable-v3' if assist else 'mapped/tag-only'
    if assist:
        va=['--episode',episode,'--run',visual,'--baseline-root',output/'baseline']
        j.call('06_visual_input',cmd('prepare_ball23_visual_root.py',*va),[visual/'input'/episode/'manifest.json',visual/'input'/episode/'input.json'])
        j.call('07_tag_prior',cmd('prepare_ball23_tag_prior.py',*va,'--create'),[visual/'input'/episode/'tag-prior.npz'])
        j.call('08_flow',cmd('process_ball23_visual_root.py',*va,'--mode','flow'),[visual/'flow'/episode/'flow-camera-roots.npz'])
        if a.visual=='full':
            sam=[str(a.sam_python or py),ROOT/'scripts/ball23_sam_visual_crops.py','--input',visual/'input'/episode,
                 '--output',visual/'sam'/episode,'--checkpoint',selection['checkpoint'],'--backend',a.sam_backend,
                 '--sam-model',a.sam_model]
            if a.sam_openvino_dir:sam.extend(['--openvino-dir',a.sam_openvino_dir.resolve()])
            j.call('09_sam3',sam,[visual/'sam'/episode/'sam-hand-crops.npz',visual/'sam'/episode/'association.json'])
            mano=[str(a.mano_python or py),ROOT/'scripts/infer_visual_mano.py',
                  '--input',visual/'input'/episode,'--output',visual/'sam-mano'/episode,
                  '--checkpoint',config['hawor_checkpoint'],'--image-extension','jpg','--chunk-batch-size','1',
                  '--assisted-crops-only','--sam-crops',visual/'sam'/episode/'sam-hand-crops.npz',
                  '--backbone-backend',a.hawor_backend]
            if a.openvino_model:
                mano.extend(['--openvino-model',a.openvino_model.resolve(),'--openvino-device',a.openvino_device])
            j.call('10_hawor_mano',mano,[visual/'sam-mano'/episode/'summary.json',visual/'sam-mano'/episode/'manifest.json'])
        j.call('11_root_fusion',cmd('process_ball23_visual_root.py',*va,'--mode','fuse','--output-stage','fused-stable-v3'),
               [visual/'fused-stable-v3'/episode/'fused-humanv2.npz',visual/'fused-stable-v3'/episode/'fusion-report.json'])
    else:
        folder=output/mapping_subdir/episode
        j.call('11_tag_only_mapping',cmd('prepare_tag_only_mapping.py','--run',output,'--episode',episode),
               [folder/'fused-humanv2.npz',folder/'fusion-report.json'])
    if not (output/'processing-verification.json').exists():freeze(output,episode,plan['mode'],mapping_subdir)
    videos={}
    if render_video:
        render=output/'final-display'
        j.call('12_render',cmd('render_motion_dashboard.py','--input-run',output,'--output',render,
            '--episode',episode,'--baseline-subdir','baseline','--visual-subdir',mapping_subdir,
            '--evidence-prefix','processing-','--first-frame',str(start+1),'--last-frame',str(stop),
            '--mode','render','--variants',*(['tag','assisted'] if assist else ['tag']),
            '--layout','final16x9','--tactile-detail','channels',*(['--tag-only'] if not assist else [])),
            [render/'delivery/delivery.json',render/'delivery/01_tag_only_final_16x9.mp4',
             *([render/'delivery/02_visual_assisted_final_16x9.mp4'] if assist else [])])
        j.call('13_display_audit',cmd('audit_final_display.py','--run',render),[render/'delivery/qa/display-contract-audit.json'])
        videos=read(render/'delivery/delivery.json')['videos']
    else:
        write_json(output/'render-disabled.json',dict(schema='qn_ego.render_policy.v1',enabled=False,
            reason='final video is rendered by the collection platform from intermediate artifacts',
            intermediate_root=str(output),mapping_subdir=mapping_subdir,
            expected_inputs=['baseline/trajectory-and-tactile.npz',
                             f'{mapping_subdir}/{episode}/fused-humanv2.npz',
                             f'{mapping_subdir}/{episode}/fusion-report.json',
                             f'baseline/{episode}/tag3-tag16-handmap-residual-v4/bone-projections.npz']))
    # Raw observations, candidate/display outputs and validity remain distinguishable.
    z=np.load(base/'trajectory-and-tactile.npz');f=np.load(output/mapping_subdir/episode/'fused-humanv2.npz')
    quality={s:dict(tag_valid=int(z[s+'_pose_valid'][start:stop].sum()),
         fused_valid=int(f[s+'_valid'][start:stop].sum()),
         sources=dict(Counter(f[s+'_source'][start:stop].tolist()))) for s in SIDES}
    summary=dict(schema='qn_ego.result.v1',at=now(),completed=True,full_visual_pipeline=a.visual=='full' and assist,
        visual_assist_enabled=assist,processing_mode=plan['mode'],
        inference_device=a.device,threads=a.threads,hawor_backend=a.hawor_backend,
        sam_backend=a.sam_backend if assist else None,sam_device=('GPU' if a.sam_backend=='openvino' else a.device) if assist else None,
        sam_model=a.sam_model if assist else None,sam_model_status=('experimental' if selection['experimental'] else 'baseline') if assist else 'not_run',
        sam_numeric_validation_passed=(read(output/'runtime-sam.json').get('numeric_validation_passed')
            if a.visual=='full' and assist and a.sam_backend=='openvino' else None),
        openvino_device=a.openvino_device if assist and a.hawor_backend=='openvino' else None,
        input=source_identity,input_format=recipe['input_format'],adapter=read(output/'provenance/mcap-adapter.json') if is_mcap else None,frames=stop-start,full_input_frames=len(times),quality=quality,
        diagnostic_derivative=read(output/'provenance/input.json').get('diagnostic_derivative'),
        videos=videos,
        camera_frame='original RGB optical (+X right,+Y down,+Z forward), metres; NOT fixed world',
        native_local_fingers_unchanged=True,training_ready=False,visual_review='pending',
        model_manifest_sha256=sha256(ROOT/'weights/manifest.json'),
        raw_unchanged=path_record(input_source)==source_identity)
    if not summary['raw_unchanged']:raise RuntimeError('Source changed during pipeline')
    write_json(output/'result.json',summary)
    files=[p for p in output.rglob('*') if p.is_file() and p.name not in ('.pipeline.lock','manifest.json','verification.json')]
    records=[dict(path=p.relative_to(output).as_posix(),bytes=p.stat().st_size,sha256=sha256(p)) for p in sorted(files)]
    write_json(output/'manifest.json',records);errors=verify_manifest(output,records)
    write_json(output/'verification.json',dict(at=now(),passed=not errors,files=len(records),
        bytes=sum(r['bytes'] for r in records),errors=errors,manifest_sha256=sha256(output/'manifest.json')))
    if errors:raise RuntimeError(errors)
    if ephemeral_mcap_view:
        from qn_ego.mcap_adapter import remove_mcap_view
        remove_mcap_view(ephemeral_mcap_view)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    return summary

def main(argv=None):
    """Run the command-line interface.

    ``argv`` is optional so an embedding application can call the same entry
    point without modifying ``sys.argv``.  The CLI behavior is unchanged when
    called without arguments.
    """
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='command',required=True)
    d=sub.add_parser('doctor',help='Check core environment and actual bundled model files')
    d.add_argument('--deep',action='store_true');d.add_argument('--config',type=Path,default=ROOT/'configs/pipeline.json')
    p=sub.add_parser('run',help='Run HDF5 or Qn DAQ v1 MCAP through every processing stage')
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/pipeline.json')
    p.add_argument('--visual',choices=['full','flow'],default='full',help='full requires SAM3+HaWoR; flow is explicit diagnostic degradation')
    p.add_argument('--sam-python',type=Path);p.add_argument('--mano-python',type=Path)
    p.add_argument('--gpu',type=int,default=0);p.add_argument('--start-s',type=float);p.add_argument('--end-s',type=float)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda',help='Torch/tensor device; OpenVINO SAM3 always runs on GPU, CPU is only its tensor glue')
    p.add_argument('--threads',type=int,choices=range(1,65),default=6,help='Per-process CPU threads; HDF5 full AprilTag defaults to six bounded workers')
    p.add_argument('--hawor-backend',choices=['torch','openvino'],default='torch')
    p.add_argument('--sam-backend',choices=['torch','openvino'],default='torch')
    p.add_argument('--sam-model',choices=['sam3','efficientsam3'],default='sam3')
    p.add_argument('--sam-openvino-dir',type=Path,help='Validated GPU SAM3 IR directory; never falls back to CPU')
    p.add_argument('--openvino-model',type=Path)
    p.add_argument('--openvino-device',choices=['CPU','GPU'],default='CPU')
    p.add_argument('--mcap-reference',type=Path,
        help='Explicit seven_gesture_reference.json for this MCAP recording; without it the bundled H5 reference is marked unverified')
    p.add_argument('--mcap-tag-search', choices=['full','roi'], default='full',
        help='MCAP AprilTag search; full preserves recall, roi enables experimental 180px/30-frame local search')
    p.add_argument('--mcap-right-tag-start', type=int, default=13,
        help='MCAP physical right-hand tag start; default follows the HDF5 order. Use 15 only to replay the historical mislabeled recording.')
    p.add_argument('--mcap-cache', type=Path,
        help='Optional previously verified provenance/mcap-canonical.view. Reuses the decoded MCAP view and avoids a second multi-GB decode; source SHA and cache metadata are checked.')
    p.add_argument('--mcap-offset-profile', choices=['ball23-h5','ball26-new'], default='ball23-h5',
        help='MCAP offset relation. ball23-h5 is the reviewed MX baseline; ball26-new is an explicit experimental CAD candidate.')
    p.add_argument('--mcap-depth-mode', choices=['raw', 'metadata'], default='raw',
        help='MCAP canonical view depth policy: raw preserves every depth payload; metadata keeps depth stream/count/scale metadata and empty payloads for Tag/HumanV2-only runs. The choice is part of cache identity.')
    p.add_argument('--tag-frame-workers', type=int, default=0,
        help='Full-frame AprilTag workers; 0 reserves two CPUs and selects min(6, --threads, CPU-2) for MCAP or min(6, --threads) for HDF5. ROI mode forces serial.')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--stop-after-apriltag',action='store_true',
        help='Pause after Tag detection, write apriltag-checkpoint.json; resume same recipe to run visual stages')
    p.add_argument('--no-visual-assist',dest='visual_assist',action='store_false',
        help='Complete Tag skeleton/tactile/video without optical flow, SAM or MANO; missing Tag remains missing')
    p.add_argument('--render-video', action='store_true',
        help='Render the final dashboard video and run display audit; disabled by default because the collection platform consumes intermediate artifacts')
    from qn_ego.guided import add_arguments as guided_arguments
    guided=sub.add_parser('guided-example',help='Tag scan -> missing counts -> ask -> optional Intel GPU visual assistance')
    guided_arguments(guided)
    t=sub.add_parser('benchmark-tag-skeleton',help='Causal recorded Tag + persistent HumanV2 test; no visual correction')
    t.add_argument('--input',type=Path,required=True);t.add_argument('--output',type=Path,required=True)
    t.add_argument('--config',type=Path,default=ROOT/'configs/pipeline.json')
    t.add_argument('--frames',type=int,default=0)
    t.add_argument('--threads',type=int,choices=range(1,65),default=4)
    t.add_argument('--modes',nargs='+',choices=['full','roi'],default=['full','roi'])
    t.add_argument('--roi-margin',type=int,default=100);t.add_argument('--refresh',type=int,default=30)
    t.add_argument('--parallel-workers',action='store_true')
    l=sub.add_parser('live-tag-skeleton',help='Actual camera and serial HumanV2; no visual correction')
    l.add_argument('--output',type=Path,required=True)
    l.add_argument('--seconds',type=float,default=30.)
    l.add_argument('--threads',type=int,choices=range(1,9),default=4)
    l.add_argument('--decimate',type=float,default=2.)
    l.add_argument('--preprocess',choices=['cpu','opencl'],default='cpu')
    l.add_argument('--solver-hz',type=int,choices=[0,30,60,120],default=0,
                   help='0 preserves all distinct receive times; others explicitly sample native solves')
    l.add_argument('--preview-port',type=int,default=0)
    l.add_argument('--preview-host',default='127.0.0.1')
    l.add_argument('--calibration',type=Path,default=ROOT/'configs/live')
    b=sub.add_parser('benchmark-openvino',help='Export and numerically benchmark real HaWoR crops; no automatic switch')
    b.add_argument('--input',type=Path,required=True);b.add_argument('--crops',type=Path,required=True)
    b.add_argument('--output',type=Path,required=True);b.add_argument('--threads',type=int,choices=range(1,65),default=4)
    s=sub.add_parser('benchmark-sam3-openvino',help='Convert original fixed-prompt SAM3 and validate GPU masks')
    s.add_argument('--input',type=Path,required=True);s.add_argument('--output',type=Path,required=True)
    s.add_argument('--threads',type=int,choices=range(1,65),default=4)
    s.add_argument('--precision',choices=['f32','f16'],default='f32')
    s.add_argument('--reuse-from',type=Path)
    s.add_argument('--validate-export-from',type=Path)
    s.add_argument('--trace-first',action='store_true')
    s.add_argument('--empty-geometry',action='store_true')
    e=sub.add_parser('benchmark-efficientsam3',help='Isolated EV-M audit/export and Intel GPU comparison; never changes full')
    e.add_argument('--phase',choices=['reference','image','grounding','validate','benchmark'],required=True)
    e.add_argument('--input',type=Path,required=True);e.add_argument('--output',type=Path,required=True)
    e.add_argument('--ir',type=Path);e.add_argument('--model',choices=['efficient','sam3'],default='efficient')
    e.add_argument('--frames',type=int,default=16);e.add_argument('--threads',type=int,choices=range(1,9),default=4)
    e.add_argument('--allow-unvalidated-diagnostic',action='store_true',help='Explicit isolated timing only; no full promotion')
    for name in ['enqueue','status','worker']:
        q=sub.add_parser(name,help='Independent serial processing queue: '+name)
        q.add_argument('--queue',type=Path,default=ROOT/'.runtime/processing')
        if name=='enqueue':
            q.add_argument('--input',type=Path,required=True)
            q.add_argument('--visual',choices=['full','flow'],default='full')
            q.add_argument('--device',choices=['cuda','cpu'],default='cpu')
            q.add_argument('--threads',type=int,choices=range(1,65),default=6)
            q.add_argument('--start-s',type=float);q.add_argument('--end-s',type=float)
            q.add_argument('--sam-backend',choices=['torch','openvino'],default='openvino',
                help='Mini PC queue defaults to GPU SAM3; missing validation is an error')
            q.add_argument('--sam-model',choices=['sam3','efficientsam3'],default='sam3')
            q.add_argument('--sam-openvino-dir',type=Path,help='Defaults to selected model IR; never silently switches model')
            q.add_argument('--hawor-backend',choices=['torch','openvino'],default='openvino')
            q.add_argument('--openvino-model',type=Path,default=ROOT/'weights/openvino/hawor-backbone-v2/model.xml')
            q.add_argument('--openvino-device',choices=['GPU'],default='GPU')
        elif name=='status':q.add_argument('--job')
        else:q.add_argument('--once',action='store_true')
    a=ap.parse_args(argv)
    if a.command=='doctor':
        value=doctor(load_config(a.config),True,a.deep)
        print(json.dumps(value,ensure_ascii=False,indent=2))
        return value
    elif a.command=='run':
        return run_pipeline(a)
    elif a.command=='guided-example':
        from qn_ego.guided import run_guided
        return run_guided(a)
    elif a.command=='benchmark-tag-skeleton':
        subprocess.run([sys.executable,str(ROOT/'scripts/benchmark_tag_skeleton.py'),
            '--input',str(a.input.resolve()),'--output',str(a.output.resolve()),
            '--config',str(a.config.resolve()),'--frames',str(a.frames),'--threads',str(a.threads),
            '--modes',*a.modes,'--roi-margin',str(a.roi_margin),'--refresh',str(a.refresh),
            *(['--parallel-workers'] if a.parallel_workers else [])],
            cwd=ROOT,env=env_for(0,'cpu',a.threads),check=True)
        return read(a.output.resolve()/'summary.json')
    elif a.command=='live-tag-skeleton':
        subprocess.run([sys.executable,str(ROOT/'scripts/live_tag_skeleton.py'),
            '--output',str(a.output.resolve()),'--seconds',str(a.seconds),
            '--threads',str(a.threads),'--decimate',str(a.decimate),
            '--preprocess',a.preprocess,'--solver-hz',str(a.solver_hz),
            '--preview-port',str(a.preview_port),'--preview-host',a.preview_host,
            '--calibration',str(a.calibration.resolve())],
            cwd=ROOT,env=env_for(0,'cpu',1),check=True)
        return read(a.output.resolve()/'summary.json')
    elif a.command=='benchmark-openvino':
        subprocess.run([sys.executable,str(ROOT/'scripts/benchmark_openvino_hawor.py'),
            '--input',str(a.input.resolve()),'--crops',str(a.crops.resolve()),
            '--output',str(a.output.resolve()),'--threads',str(a.threads)],
            cwd=ROOT,env=env_for(0,'cpu',a.threads),check=True)
    elif a.command=='benchmark-efficientsam3':
        subprocess.run([sys.executable,str(ROOT/'scripts/benchmark_efficientsam3.py'),
            '--phase',a.phase,'--input',str(a.input.resolve()),'--output',str(a.output.resolve()),
            '--model',a.model,'--frames',str(a.frames),'--threads',str(a.threads),
            *(['--ir',str(a.ir.resolve())] if a.ir else []),
            *(['--allow-unvalidated-diagnostic'] if a.allow_unvalidated_diagnostic else [])],
            cwd=ROOT,env=env_for(0,'cpu',a.threads),check=True)
    elif a.command=='benchmark-sam3-openvino':
        subprocess.run([sys.executable,str(ROOT/'scripts/prepare_openvino_sam3.py'),
            '--input',str(a.input.resolve()),'--output',str(a.output.resolve()),
            '--threads',str(a.threads),'--precision',a.precision,
            *(['--reuse-from',str(a.reuse_from.resolve())] if a.reuse_from else []),
            *(['--validate-export-from',str(a.validate_export_from.resolve())] if a.validate_export_from else []),
            *(['--trace-first'] if a.trace_first else []),
            *(['--empty-geometry'] if a.empty_geometry else [])],
            cwd=ROOT,env=env_for(0,'cpu',a.threads),check=True)
        return read(a.output.resolve()/'summary.json') if (a.output.resolve()/'summary.json').exists() else None
    else:
        from qn_ego import jobs
        if a.command=='worker':return jobs.worker(a,ROOT)
        return getattr(jobs,a.command)(a)
