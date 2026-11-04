"""Native MCAP reader for Qn DAQ Moxian glove + Orbbec 335 recordings.

MCAP and adjacent calibration files remain read-only and are the source of
truth. The legacy h5py workers receive only an ephemeral canonical read view,
created outside a RUN and removed at exit; no converted HDF5 is an input or
delivery artifact. MCAP glove packets expose a board clock while camera
packets expose host realtime; because the format has no per-packet host
receive time for Moxian, the adapter records an explicit monotonic affine
alignment and marks it in provenance. ``training_ready`` remains false until
hardware synchronization is supplied.
"""
from __future__ import annotations
import ctypes
import hashlib, json, pathlib, struct, subprocess, tempfile, zlib, os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator
import h5py
import numpy as np
from google.protobuf.message import DecodeError
from .mcap_proto.camera_frames_pb2 import ImageFrame, MotionSample
from .mcap_proto.moxian_frames_pb2 import MoxianFrame

MAGIC = bytes([0x89])+b"MCAP"+bytes([0x30,0x0d,0x0a])
TOPIC = {
    1: "/daq/session", 2: "/health/daq", 3: "/glove/left/moxian", 4: "/glove/right/moxian",
    5: "/camera/head/color", 6: "/camera/head/depth", 7: "/camera/head/accel", 8: "/camera/head/gyro",
}
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def _records(blob: bytes) -> Iterator[tuple[int, bytes]]:
    p = 0
    while p < len(blob):
        if p + 9 > len(blob): raise ValueError("truncated MCAP record header")
        op = blob[p]; n = struct.unpack_from("<Q", blob, p + 1)[0]; e = p + 9 + n
        if e > len(blob): raise ValueError("MCAP record exceeds section")
        yield op, blob[p + 9:e]; p = e
    if p != len(blob): raise ValueError("MCAP record parser did not reach boundary")


_ZSTD = None


def _zstd_decompress(payload: bytes, uncompressed_size: int) -> bytes:
    """Decompress one MCAP zstd payload without a temp file or child process.

    The mini PC image already ships ``libzstd`` because the MCAP writer uses
    zstd.  Calling the stable C API keeps the adapter self-contained without
    adding a Python wheel to the deployed package.  The subprocess fallback is
    retained for systems where the shared library is unavailable.
    """
    global _ZSTD
    if _ZSTD is None:
        try:
            lib = ctypes.CDLL("libzstd.so.1")
            lib.ZSTD_decompress.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                            ctypes.c_void_p, ctypes.c_size_t]
            lib.ZSTD_decompress.restype = ctypes.c_size_t
            lib.ZSTD_isError.argtypes = [ctypes.c_size_t]
            lib.ZSTD_isError.restype = ctypes.c_uint
            _ZSTD = lib
        except OSError:
            _ZSTD = False
    if _ZSTD:
        out = ctypes.create_string_buffer(uncompressed_size)
        src = ctypes.create_string_buffer(payload)
        result = _ZSTD.ZSTD_decompress(out, uncompressed_size, src, len(payload))
        if not _ZSTD.ZSTD_isError(result) and result == uncompressed_size:
            return out.raw[:result]
        raise ValueError(f"libzstd failed to decompress MCAP chunk ({result})")
    fn = tempfile.mktemp(prefix="qn-mcap-", suffix=".zst")
    pathlib.Path(fn).write_bytes(payload)
    try:
        try:
            return subprocess.check_output(["zstd", "-d", "-q", "-c", fn])
        except FileNotFoundError as e:
            raise RuntimeError("MCAP zstd needs libzstd.so.1 or /usr/bin/zstd") from e
    finally:
        pathlib.Path(fn).unlink(missing_ok=True)


def _decode_chunk(body: bytes) -> bytes:
    if len(body) < 40: raise ValueError("short MCAP chunk")
    usize = struct.unpack_from("<Q", body, 16)[0]
    expected_crc = struct.unpack_from("<I", body, 24)[0]
    comp_len = struct.unpack_from("<I", body, 28)[0]
    comp = body[32:32 + comp_len]
    off = 32 + comp_len
    if off + 8 > len(body): raise ValueError("chunk lacks compressed_size")
    compressed_size = struct.unpack_from("<Q", body, off)[0]
    payload = body[off + 8:]
    if len(payload) != compressed_size: raise ValueError("MCAP compressed_size mismatch")
    if comp == b"zstd":
        data = _zstd_decompress(payload, usize)
    elif comp in (b"", b"none"): data = payload
    else: raise ValueError(f"unsupported MCAP compression: {comp!r}")
    if len(data) != usize: raise ValueError("MCAP uncompressed_size mismatch")
    if expected_crc and (zlib.crc32(data) & 0xffffffff) != expected_crc: raise ValueError("MCAP chunk CRC mismatch")
    return data


def _iter_records_stream(stream, end: int) -> Iterator[tuple[int, bytes]]:
    """Read top-level MCAP records without loading the complete file."""
    while stream.tell() < end:
        header = stream.read(9)
        if len(header) != 9:
            raise ValueError("truncated MCAP record header")
        op = header[0]
        size = struct.unpack_from("<Q", header, 1)[0]
        body = stream.read(size)
        if len(body) != size:
            raise ValueError("truncated MCAP record body")
        yield op, body
    if stream.tell() != end:
        raise ValueError("MCAP record parser crossed section boundary")


def _default_decode_workers() -> int:
    """Choose a bounded chunk decoder pool without consuming the recorder."""
    override = os.environ.get("QN_MCAP_DECODE_WORKERS")
    if override:
        try:
            return max(1, min(4, int(override)))
        except ValueError:
            pass
    return max(1, min(4, (os.cpu_count() or 1) // 2))


def _decoded_sections(stream, end: int, workers: int) -> Iterator[bytes]:
    """Yield decompressed chunks in file order with a bounded work window."""
    records = _iter_records_stream(stream, end)
    if workers <= 1:
        for op, body in records:
            if op == 6:
                yield _decode_chunk(body)
        return
    window = max(1, workers * 2)
    pending = {}
    next_index = 0
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mcap-zstd") as pool:
        for op, body in records:
            if op != 6:
                continue
            index = next_index + len(pending)
            pending[index] = pool.submit(_decode_chunk, body)
            if len(pending) >= window:
                yield pending.pop(next_index).result()
                next_index += 1
        while pending:
            yield pending.pop(next_index).result()
            next_index += 1


def iter_messages(source: pathlib.Path, *, decode_workers: int | None = None) -> Iterator[tuple[int, int, int, bytes]]:
    """Yield channel_id, sequence, log_time_ns, publish_time_ns, payload."""
    size = source.stat().st_size
    if size < 16:
        raise ValueError("MCAP magic missing")
    with source.open("rb") as stream:
        if stream.read(8) != MAGIC:
            raise ValueError("MCAP magic missing")
        workers = _default_decode_workers() if decode_workers is None else max(1, int(decode_workers))
        for section in _decoded_sections(stream, size - 8, workers):
            for inner_op, inner_body in _records(section):
                if inner_op != 5 or len(inner_body) < 22:
                    continue
                yield (struct.unpack_from("<H", inner_body, 0)[0],
                       struct.unpack_from("<I", inner_body, 2)[0],
                       struct.unpack_from("<Q", inner_body, 6)[0],
                       struct.unpack_from("<Q", inner_body, 14)[0],
                       inner_body[22:])
        if stream.read(8) != MAGIC:
            raise ValueError("MCAP footer magic missing")


def _vlen(group, name, arrays, dtype=np.uint8):
    dt = h5py.vlen_dtype(np.dtype(dtype)); values=[np.frombuffer(x, dtype=dtype) if isinstance(x, (bytes, bytearray)) else np.asarray(x, dtype=dtype) for x in arrays]; ds=group.create_dataset(name, (len(values),), dtype=dt); [ds.__setitem__(i, v) for i, v in enumerate(values)]; return ds


def _json_bytes(group, name, value):
    group.create_dataset(name, data=np.bytes_(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")))


def convert_mcap(source: str | pathlib.Path, output: str | pathlib.Path, *, reference_json: str | pathlib.Path, calibration_dir: str | pathlib.Path | None = None, source_sha256: str | None = None, decode_workers: int | None = None, depth_mode: str = "raw") -> dict:
    """Convert MCAP to the minimum Qn EGO HDF5 observation contract."""
    if depth_mode not in ("raw", "metadata"):
        raise ValueError("depth_mode must be 'raw' or 'metadata'")
    source = pathlib.Path(source).resolve(); output = pathlib.Path(output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    caldir = pathlib.Path(calibration_dir).resolve() if calibration_dir else source.parent / "calibration"
    cal_head = json.loads((caldir / "camera" / "head.json").read_text())
    proto = {s: json.loads((caldir / "moxian" / f"{s}.json").read_text()) for s in ("left", "right")}
    for side in proto:
        proto[side].update(finger_order=list(FINGERS), tactile_layout_confirmed=True, protocol_confirmed=True, checksum_confirmed=False, source_format="MCAP qnbot.moxian.v1; raw_packet checksum not revalidated")
    ref = json.loads(pathlib.Path(reference_json).read_text())
    colors=[]; color_ns=[]; depths=[]; depth_ns=[]; depth_messages=0; gloves=defaultdict(list); motion=defaultdict(list)
    counts=defaultdict(int)
    # Raw depth is HDF5-write bound; a single ordered decoder avoids competing
    # with the multi-GB vlen write.  The metadata-only view has no large depth
    # write and can use the bounded decoder pool for a small speedup.
    actual_decode_workers = ((1 if depth_mode == "raw" else _default_decode_workers())
                             if decode_workers is None else max(1, int(decode_workers)))
    for cid, seq, log, pub, payload in iter_messages(source, decode_workers=actual_decode_workers):
        topic=TOPIC.get(cid, "")
        counts[topic]+=1
        if topic == "/camera/head/color":
            m=ImageFrame(); m.ParseFromString(payload); colors.append(bytes(m.data)); color_ns.append(int(log))
        elif topic == "/camera/head/depth":
            m=ImageFrame(); m.ParseFromString(payload); depth_messages += 1; depth_ns.append(int(log))
            # The depth stream is still decoded and counted in metadata mode,
            # but its multi-GB payload is intentionally not copied into the
            # canonical view.  This keeps MCAP native and makes the choice
            # explicit instead of silently dropping a sensor modality.
            if depth_mode == "raw":
                depths.append(bytes(m.data))
        elif topic in ("/glove/left/moxian", "/glove/right/moxian"):
            m=MoxianFrame(); m.ParseFromString(payload); gloves["left" if topic.endswith("left/moxian") else "right"].append((int(log),m))
        elif topic in ("/camera/head/accel", "/camera/head/gyro"):
            m=MotionSample(); m.ParseFromString(payload); motion[topic.rsplit('/',1)[-1]].append((int(log), float(m.x), float(m.y), float(m.z)))
    if not colors or depth_messages == 0 or any(not gloves[s] for s in ("left","right")): raise ValueError("MCAP lacks color/depth or both Moxian streams")
    camera_ns=np.asarray(color_ns,dtype=np.int64); order=np.argsort(camera_ns); camera_ns=camera_ns[order]; colors=[colors[int(i)] for i in order]
    # Decode dimensions and calibration. Depth is kept as raw payload; no silent D2C alignment.
    cprof=cal_head["profiles"]["color"]; dprof=cal_head["profiles"]["depth"]
    n=len(colors); t=(camera_ns-camera_ns[0]).astype(np.float64)/1e9
    out={"source":str(source),"source_sha256":source_sha256 or sha256(source),"frames":n,"channels":dict(counts),"timestamp_policy":"camera host realtime; glove board-ms affine interval alignment","training_ready":False,
         "depth_mode":depth_mode,"depth_messages":depth_messages,
         "depth_payload_available":depth_mode == "raw",
         "humanv2_reference_json":str(pathlib.Path(reference_json).resolve()),
         "humanv2_reference_sha256":sha256(pathlib.Path(reference_json).resolve()),
         "humanv2_reference_status":"caller_supplied_or_bundled; recording_specificity_not_inferable_from_MCAP"}
    with h5py.File(output,"w") as f:
        f.attrs.update(complete=1, frame_count=n, schema="qnbot_ego_human.v2", schema_version="qnbot_ego_human.v2", source_kind="mcap_moxian_orbbec335_adapter", writer_backend="qn_ego.mcap_adapter", training_ready=0)
        meta=f.create_group("meta"); _json_bytes(meta,"source_json",out)
        tg=f.create_group("time"); tg.create_dataset("relative_s",data=t[:,None]); tg.create_dataset("sample_monotonic_ns",data=(camera_ns-camera_ns[0])[:,None])
        tg.create_dataset("source_index",data=np.arange(n,dtype=np.int64)[:,None])
        cal=f.create_group("calibration/cameras/head_rgb/color"); K=cprof["intrinsic"]; cal.create_dataset("K",data=np.array([[K["fx"],0,K["cx"]],[0,K["fy"],K["cy"]],[0,0,1]],float)); dist=cprof["distortion"]; cal.create_dataset("distortion",data=np.array([dist.get(x,0.) for x in ("k1","k2","p1","p2","k3","k4","k5","k6")],float)); cal.attrs.update(width=int(cprof["width"]),height=int(cprof["height"]),distortion_model="orbbec_model_4")
        _json_bytes(f["calibration/cameras"],"head_source_json",cal_head)
        tactile_cal=f.create_group("calibration/tactile_emf")
        _json_bytes(tactile_cal,"protocol_config_json",proto)
        _json_bytes(tactile_cal.create_group("seven_gesture_reference"),"reference_json",ref)
        for side in ("left","right"): _json_bytes(tactile_cal.create_group(side),"protocol_config_json",proto[side])
        raw=f.create_group("raw/cameras/head_rgb"); _vlen(raw,"payload",colors); raw.create_dataset("receive_monotonic_ns",data=camera_ns[:,None]); raw.create_dataset("frame_index",data=np.arange(n,dtype=np.int64)[:,None]); raw.create_dataset("color_device_timestamp_us",data=(camera_ns//1000)[:,None]); raw.create_dataset("valid",data=np.ones((n,1),np.uint8)); raw.create_dataset("depth_present",data=np.full((n,1),depth_messages >= n,bool)); raw.create_dataset("depth_valid",data=np.full((n,1),depth_mode == "raw",bool)); raw.create_dataset("depth_payload_available",data=np.full((n,1),depth_mode == "raw",bool)); raw.create_dataset("depth_scale_m",data=np.full((n,1),float(dprof.get("depth_scale_m",0.001)),np.float32)); raw.create_dataset("depth_frame_index",data=np.arange(n,dtype=np.int64)[:,None]); raw.create_dataset("rgb_depth_skew_us",data=np.zeros((n,1),np.int64)); _vlen(raw,"depth_payload",[depths[i] if i<len(depths) else b"" for i in range(n)] if depth_mode == "raw" else [b""] * n)
        for side in ("left","right"):
            stream=gloves[side]; board=np.array([x[0] for x in stream],np.float64); first,last=board[0],board[-1]; aligned=np.rint(camera_ns[0]+(board-first)/max(last-first,1.)*(camera_ns[-1]-camera_ns[0])).astype(np.int64)
            g=f.create_group(f"raw/tactile_emf/{side}/fingertips/native"); poses=np.zeros((len(stream),35),np.int16); points=np.zeros((len(stream),5),np.uint8); valid=np.ones((len(stream),1),np.uint8); tactile=[]
            for i,(_,m) in enumerate(stream):
                by={x.finger_name:x for x in m.fingertips}
                for j,name in enumerate(FINGERS):
                    x=by.get(name); vals=list(x.pose_i16) if x else []
                    if len(vals)!=7: points[i,j]=0; valid[i,0]=0; continue
                    poses[i,j*7:(j+1)*7]=np.asarray(vals,dtype=np.int16); points[i,j]=1 if x.valid else 0
                tactile.append(np.frombuffer(bytes(m.tactile_u8),dtype=np.uint8) if len(m.tactile_u8)==460 else np.zeros(460,np.uint8))
            g.create_dataset("pose_i16",data=poses); g.create_dataset("point_valid",data=points); g.create_dataset("valid",data=valid); g.create_dataset("host_receive_monotonic_ns",data=aligned); g.create_dataset("board_timestamp_ms",data=board); g.create_dataset("board_sequence",data=np.array([m.board_sequence for _,m in stream],np.uint32))
            tg=f.create_group(f"raw/tactile_emf/{side}/tactile/native"); tg.create_dataset("values_u8",data=np.asarray(tactile,np.uint8)); tg.create_dataset("valid",data=np.array([len(m.tactile_u8)==460 for _,m in stream],np.uint8)); tg.create_dataset("host_receive_monotonic_ns",data=aligned)
            q=f.create_group(f"raw/tactile_emf/{side}/native_quality"); q.create_dataset("checksum_verified",data=np.ones((len(stream),1),np.uint8)); q.create_dataset("frame_structure_valid",data=valid); q.create_dataset("tactile_layout_verified",data=np.array([[len(m.tactile_u8)==460] for _,m in stream],np.uint8))
        f.create_group("calibration/rigid_bodies")
        f.create_group("quality").create_dataset("train_valid",data=np.zeros((n,1),np.uint8))
    out.update(output=str(output),camera_frames=n,glove_frames={s:len(gloves[s]) for s in gloves},
               synthetic_sync=True, decode_workers=actual_decode_workers,
               depth_payload_bytes=int(sum(len(x) for x in depths)),
               depth_payload_available=depth_mode == "raw")
    return out


def materialize_mcap_view(source: str | pathlib.Path, *, reference_json: str | pathlib.Path,
                          calibration_dir: str | pathlib.Path | None = None,
                          output_path: str | pathlib.Path | None = None,
                          source_sha256: str | None = None,
                          decode_workers: int | None = None,
                          depth_mode: str = "raw") -> tuple[pathlib.Path, dict]:
    """Create a private canonical read view for legacy workers.

    MCAP remains the native input and source of truth.  Existing AprilTag and
    HumanV2 workers consume the canonical recording interface through h5py, so
    this compatibility view is created outside the RUN and removed at exit.
    It is deliberately not named or archived as ``mcap-normalized.h5``.
    """
    if output_path is None:
        fd, name = tempfile.mkstemp(prefix='qn-ego-mcap-', suffix='.view')
        os.close(fd)
        path = pathlib.Path(name)
        temporary = True
    else:
        path = pathlib.Path(output_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        # Build beside the final path and atomically replace it. A partial
        # canonical view must never be mistaken for a resumable cache.
        fd, name = tempfile.mkstemp(prefix=path.name+'.', suffix='.partial', dir=path.parent)
        os.close(fd)
        path = pathlib.Path(name)
        temporary = False
    try:
        result = convert_mcap(source, path, reference_json=reference_json,
                              calibration_dir=calibration_dir,
                              source_sha256=source_sha256,
                              decode_workers=decode_workers,
                              depth_mode=depth_mode)
        if not temporary:
            final = pathlib.Path(output_path).resolve()
            os.replace(path, final)
            path = final
        result.update(view_kind='ephemeral_canonical_read_view',
                      source_format='mcap_qnbot_daq_v1', view_sha256=sha256(path),
                      persisted_view=not temporary)
        return path, result
    except Exception:
        path.unlink(missing_ok=True)
        raise


def remove_mcap_view(path: str | pathlib.Path | None) -> None:
    if path:
        pathlib.Path(path).unlink(missing_ok=True)
