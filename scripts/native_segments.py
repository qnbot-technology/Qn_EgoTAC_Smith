"""Run all-valid native driver independently per contiguous valid segment."""
from pathlib import Path
import json
import subprocess
import sys
import numpy as np


def segment_ranges(valid, ticks):
    valid = np.asarray(valid, bool)
    ticks = np.asarray(ticks, np.int64)
    if valid.shape != ticks.shape or valid.ndim != 1:
        raise ValueError('Validity and timestamps must have matching one-dimensional shapes')
    start = None
    result = []
    for i, good in enumerate(valid):
        boundary = i > 0 and (ticks[i] <= ticks[i-1] or ticks[i]-ticks[i-1] > 100_000_000)
        if start is not None and (not good or boundary):
            result.append((start, i)); start = None
        if good and start is None:
            start = i
    if start is not None: result.append((start, len(valid)))
    return result


def run_segments(args, script, arrays):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    poses = np.asarray(arrays['poses'])
    ticks = np.asarray(arrays['receive_ns'], np.int64)
    valid = np.asarray(arrays['valid'], bool) & np.isfinite(poses).all(tuple(range(1, poses.ndim)))
    ranges = segment_ranges(valid, ticks)
    if not ranges:
        raise ValueError('No valid glove segment; no native skeleton can be solved')
    merged = {}; reports = []; commands = []
    for number, (start, stop) in enumerate(ranges):
        folder = out / 'segments' / f'{number:04d}'
        folder.mkdir(parents=True, exist_ok=True)
        prepared = folder / 'prepared-input.npz'
        np.savez_compressed(prepared, poses=poses[start:stop], receive_ns=ticks[start:stop], source_index=arrays['source_index'][start:stop])
        command = [sys.executable, str(script), '--input-npz', str(prepared), '--output', str(folder), '--package', str(args.package), '--side', args.side]
        if args.config: command += ['--config', str(args.config)]
        with (folder / 'driver.log').open('w') as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        commands.append(command)
        with np.load(folder / 'native-output.npz') as result:
            for key in result.files:
                value = result[key]
                if key in ('timestamp_sec', 'source_index'): continue
                if key not in merged:
                    merged[key] = np.full((len(valid),) + value.shape[1:], False if value.dtype.kind == 'b' else np.nan, dtype=bool if value.dtype.kind == 'b' else float)
                merged[key][start:stop] = value
        reports.append(json.loads((folder / 'native-report.json').read_text()))
    timestamp = (ticks-ticks[0])/1e9
    merged.update(timestamp_sec=timestamp, source_index=arrays['source_index'])
    np.savez_compressed(out / 'native-output.npz', **merged)
    np.savez_compressed(out / 'input-poses.npz', poses=poses, receive_ns=ticks, source_index=arrays['source_index'], timestamp_sec=timestamp)
    report = dict(reports[0])
    report.update(input_npz=str(args.input_npz.resolve()), input_sha256=__import__('hashlib').sha256(args.input_npz.read_bytes()).hexdigest(), frames=len(valid), tracking_valid_frames=int(merged['tracking_valid'].sum()), tracking_valid_fraction=float(merged['tracking_valid'].mean()), invalid_input_rows=int((~valid).sum()), segment_ranges_0based_halfopen=ranges, independent_runtime_per_segment=True, original_timeline_preserved=True, segment_reports=[str(out / 'segments' / f'{i:04d}' / 'native-report.json') for i in range(len(ranges))])
    (out / 'native-report.json').write_text(json.dumps(report, indent=2)+'\n')
    (out / 'run-command.json').write_text(json.dumps({'commands':commands,'independent_runtime_per_segment':True},indent=2)+'\n')
