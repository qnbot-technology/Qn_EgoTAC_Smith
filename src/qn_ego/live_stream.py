"""Bounded Linux x86-64 live transport. Fresh snapshots and causal solutions only."""
from collections import deque
import json
import mmap
import struct
from threading import Lock

import numpy as np


def cad_root_bridges(layout, render_root_bridges):
    """C++ fits bindings in CAD-bottom axes; Python maps in render-root axes.

    T_R_Tag = T_R_B @ T_B_Tag, hence T_C_R = T_C_B @ inv(T_R_B).
    Fold that one basis conversion into the mapping bridge, never into EMF.
    """
    C = np.asarray(layout['T_render_root_cad_bottom'], float)
    for key, top in layout['T_bottom_tag'].items():
        np.testing.assert_allclose(top,C @ layout['bindings'][key]['T_bottom_tag'],atol=1e-12)
    return {s:np.linalg.inv(C) @ B for s,B in render_root_bridges.items()}


class SolveRate:
    """Causal first packet per monotonic time bucket; never rewrite timestamps."""
    def __init__(self, hz):
        if not np.isfinite(hz) or hz <= 0:
            raise ValueError('Positive finite solve rate required')
        self.period_ns = round(1e9 / hz)
        self.last_bucket = -1

    def take(self, ns):
        bucket = ns // self.period_ns
        if bucket <= self.last_bucket:
            return False
        self.last_bucket = bucket
        return True


class FrameReader:
    def __init__(self, path, width, height):
        self.file = open(path, 'rb')
        self.memory = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        self.width, self.height = width, height
        self.last_sequence = 0
        if len(self.memory) != 16+8192+width*height*3:
            self.close()
            raise ValueError('Unexpected frame mailbox geometry')

    def take(self):
        m = self.memory
        before = struct.unpack_from('<Q', m, 0)[0]
        if not before or before & 1 or before == self.last_sequence:
            return None
        count = struct.unpack_from('<Q', m, 8)[0]
        if count > 8192:
            return None  # writer may have begun; retry on next call
        payload = m[16:16+count]
        image = np.frombuffer(m, dtype=np.uint8, offset=8208).reshape(self.height,self.width,3).copy()
        if before != struct.unpack_from('<Q', m, 0)[0]:
            return None
        row = json.loads(payload)
        if row['width'] != self.width or row['height'] != self.height or row.get('input_rectified') is not True:
            raise ValueError('Unreviewed live image contract')
        row['mailbox_sequence'] = before
        self.last_sequence = before
        return image, row

    def close(self):
        self.memory.close(); self.file.close()


class SolutionHistory:
    def __init__(self, capacity=512):
        self.rows = deque(maxlen=capacity)
        self.lock = Lock()

    def append(self, ns, reply):
        with self.lock:
            if self.rows and ns <= self.rows[-1][0]:
                raise ValueError('Nonincreasing glove receive time')
            self.rows.append((ns,reply))

    def causal(self, ns, max_age_ms=30.):
        with self.lock:
            for stamp, reply in reversed(self.rows):
                if stamp <= ns:
                    age = (ns-stamp)/1e6
                    return reply, stamp, age, age <= max_age_ms
        return None, -1, None, False
