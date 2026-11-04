"""Serial, explicit HDF5 jobs; never opens a camera or changes a recording."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from qn_smith.artifacts import now, sha256


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    os.replace(temp, path)


def enqueue(a):
    import h5py
    source = a.input.resolve(strict=True)
    before = source.stat()
    with h5py.File(source, 'r') as f:
        if f.attrs.get('complete') != 1:
            raise ValueError('Only completed, closed recordings may be queued')
        frames = len(f['time/relative_s'])
    digest = sha256(source)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Input is still changing')
    # Queue jobs freeze their accelerator recipe; the worker never guesses CPU.
    backends=dict(sam_backend=getattr(a,'sam_backend','torch'),
        sam_model=getattr(a,'sam_model','sam3'),
        hawor_backend=getattr(a,'hawor_backend','torch'),
        openvino_device=getattr(a,'openvino_device','GPU'))
    for key in ('sam_openvino_dir','openvino_model'):
        value=getattr(a,key,None)
        backends[key]=str(value.resolve()) if value else None
    from qn_ego.sam_selection import resolve_sam
    from qn_ego.pipeline import ROOT, load_config
    selection=resolve_sam(backends['sam_model'],backends['sam_backend'],a.device,
        backends['sam_openvino_dir'],load_config(ROOT/'configs/pipeline.json'),ROOT)
    backends['sam_openvino_dir']=str(selection['directory']) if selection['directory'] else None
    validation_files={};sam_numeric_validation_passed=None
    if a.visual=='full' and backends['sam_backend']=='openvino':
        if a.device!='cpu' or not backends['sam_openvino_dir']:
            raise ValueError('SAM3 OpenVINO queue requires CPU tensor glue and model directory')
        from qn_ego.sam3_assets import validate_sam3_assets,validate_efficientsam3_assets
        modeldir=Path(backends['sam_openvino_dir'])
        validate=validate_sam3_assets if backends['sam_model']=='sam3' else validate_efficientsam3_assets
        model_receipt=validate(modeldir,selection['checkpoint'])
        sam_numeric_validation_passed=model_receipt['numeric_passed']
        path=modeldir/'validation.json'
        validation_files[str(path)]=sha256(path)
        validation_files[str(modeldir/'conversion.json')]=sha256(modeldir/'conversion.json')
        validation_files[str(selection['checkpoint'])]=sha256(selection['checkpoint'])
    if a.visual=='full' and backends['hawor_backend']=='openvino':
        if a.device!='cpu' or not backends['openvino_model']:
            raise ValueError('HaWoR OpenVINO queue requires CPU tensor glue and model path')
        path=Path(backends['openvino_model']).with_name('validation.json')
        validation_files[str(path)]=sha256(path)
    job_id = time.strftime('%Y%m%d-%H%M%S-')+uuid.uuid4().hex[:8]
    queue = a.queue.resolve();queue.mkdir(parents=True, exist_ok=True)
    record = dict(schema='qn_ego.job.v1', id=job_id, status='pending', submitted_at=now(),
        input=str(source), input_sha256=digest, input_bytes=before.st_size,
        input_mtime_ns=before.st_mtime_ns, frames=frames, device=a.device,
        visual=a.visual, threads=a.threads, start_s=a.start_s, end_s=a.end_s,
        output=str(queue/'results'/job_id),backends=backends,validation_files=validation_files,
        sam_model_status='experimental' if selection['experimental'] else 'baseline',
        sam_numeric_validation_passed=sam_numeric_validation_passed,
        training_ready=False)
    save(queue/(job_id+'.json'), record)
    print(json.dumps(record, ensure_ascii=False, indent=2))


def status(a):
    if a.job and (Path(a.job).name != a.job or not a.job):
        raise ValueError('Invalid job id')
    paths = [a.queue/(a.job+'.json')] if a.job else sorted(a.queue.glob('*.json'))
    print(json.dumps([json.loads(p.read_text()) for p in paths], ensure_ascii=False, indent=2))


def worker(a, root):
    queue = a.queue.resolve();queue.mkdir(parents=True, exist_ok=True)
    with (queue/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        # A killed process is not silently retried or overwritten.
        for p in queue.glob('*.json'):
            rec=json.loads(p.read_text())
            if rec['status']=='running':
                rec.update(status='interrupted', finished_at=now(), error='Previous worker ended before receipt')
                save(p,rec)
        stopping = False
        def stop(*_):
            nonlocal stopping
            stopping = True
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        while not stopping:
            for path in sorted(queue.glob('*.json')):
                if stopping:break
                rec=json.loads(path.read_text())
                if rec['status']!='pending':continue
                try:
                    source=Path(rec['input']);st=source.stat()
                    if (st.st_size,st.st_mtime_ns)!=(rec['input_bytes'],rec['input_mtime_ns']) or sha256(source)!=rec['input_sha256']:
                        raise ValueError('Queued input changed')
                    if 'backends' not in rec:
                        raise ValueError('Legacy queued job lacks explicit backend recipe; submit a new job')
                    for name,digest in rec['validation_files'].items():
                        if sha256(name)!=digest:raise ValueError('Queued accelerator validation changed')
                    cmd=[sys.executable,str(root/'run_pipeline.py'),'run','--input',str(source),
                         '--output',rec['output'],'--visual',rec['visual'],'--device',rec['device'],
                         '--threads',str(rec['threads'])]
                    for name,value in rec['backends'].items():
                        if value is not None:cmd.extend(['--'+name.replace('_','-'),str(value)])
                    for name in ('start_s','end_s'):
                        if rec[name] is not None:cmd.extend(['--'+name.replace('_','-'),str(rec[name])])
                    log=queue/(rec['id']+'.log')
                    rec.update(status='running',started_at=now(),argv=cmd,log=str(log))
                    save(path,rec)
                    with log.open('x') as stream:
                        child=subprocess.Popen(cmd,cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                        rec['pid']=child.pid;save(path,rec)
                        while child.poll() is None:
                            if stopping:
                                os.killpg(child.pid,signal.SIGTERM)
                                try:child.wait(timeout=20)
                                except subprocess.TimeoutExpired:
                                    os.killpg(child.pid,signal.SIGKILL);child.wait()
                                break
                            time.sleep(.5)
                    rec.update(exit_code=child.returncode,finished_at=now())
                    result=Path(rec['output'])/'result.json'
                    verified=Path(rec['output'])/'verification.json'
                    if stopping:rec.update(status='interrupted')
                    elif child.returncode==0 and result.is_file() and verified.is_file() and json.loads(verified.read_text())['passed']:
                        rec.update(status='completed',result=str(result),verification=str(verified))
                    else:rec.update(status='failed',error='Pipeline failed; inspect log and stage receipt')
                except Exception as exc:
                    rec.update(status='failed',error=str(exc),finished_at=now())
                save(path,rec)
                print(rec['id'],rec['status'],flush=True)
            if a.once:break
            time.sleep(2)
