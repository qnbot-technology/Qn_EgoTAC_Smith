"""Read-only tactile/EMF protocol support. Unknown firmware is never guessed."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct
from .tactile_v1 import PROTOCOL as V1_PROTOCOL, decode_frame, MAX_FRAME_SIZE

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / 'configs/live/tactile_emf.json'
MAGIC = b'\xa5\x5a'
PACKET_SIZE = 548  # 2+1+2+1+460+80+2, from PDF field widths
REFERENCE_PROTOCOL = 'moxian_reference_593_v1'


def is_reference(config):
    return config.get('protocol') == REFERENCE_PROTOCOL


def tactile_required(config):
    return config.get('tactile_required', True)


def capture_sides(config):
    return tuple(config.get('capture_sides', [config['expected_side']]))


def side_config(config, side):
    if side not in ('left', 'right') or side not in capture_sides(config):
        raise ValueError('hand side not selected in capture configuration')
    # Keep the session SHA: all streams share one immutable session contract.
    return dict(config, expected_side=side, capture_sides=[side])


def capture_channels_ready(flags, config):
    return flags['fingertips'] and (flags['tactile'] or not tactile_required(config))


def load_config(path=DEFAULT_CONFIG):
    path = Path(path)
    content = path.read_bytes()
    config = json.loads(content)
    if config.get('schema') != 'independent.tactile_emf.config.v1':
        raise ValueError('unsupported tactile configuration')
    validate_config(config)
    return {**config, 'source_path': str(path.resolve()), 'sha256': hashlib.sha256(content).hexdigest()}


def validate_config(config):
    if config.get('schema') != 'independent.tactile_emf.config.v1':
        raise ValueError('unsupported independent manufacturer configuration')
    if not isinstance(config.get('vendor'), str) or not config['vendor'].strip() or config['vendor'].lower() == 'qnbot':
        raise ValueError('independent manufacturer identity required; QnBot config is forbidden')
    if not isinstance(config.get('protocol_confirmed'), bool):
        raise ValueError('protocol_confirmed must be boolean')
    sides = capture_sides(config)
    if (not sides or len(set(sides)) != len(sides) or any(s not in ('left', 'right') for s in sides)
            or config.get('expected_side') not in ('left', 'right', 'auto')
            or (config.get('expected_side') != 'auto' and sides != (config['expected_side'],))
            or (config.get('protocol') == 'pdf_tactile_emf_0x71' and sides != ('left',))):
        raise ValueError('invalid capture sides for independent glove protocol')
    if type(config.get('baudrate')) is not int or not 0 < config['baudrate'] <= 4_000_000:
        raise ValueError('invalid baudrate')
    order = config.get('finger_order')
    if order is not None and (not isinstance(order, list) or len(order) != 5 or
                              any(not isinstance(v, str) for v in order) or
                              set(order) != {'thumb','index','middle','ring','pinky'}):
        raise ValueError('finger_order must be null or all five unique finger names')
    if config.get('protocol') not in ('pdf_tactile_emf_0x71', REFERENCE_PROTOCOL, V1_PROTOCOL):
        raise ValueError('unsupported independent glove protocol')
    if config.get('protocol') == V1_PROTOCOL:
        if (config.get('checksum_scope') != 'all_bytes_before_checksum' or
                config.get('integrity_policy') != 'crc16_modbus' or
                config.get('checksum_confirmed') is not True or
                config.get('tactile_layout_confirmed') is not True or
                order != ['thumb', 'index', 'middle', 'ring', 'pinky']):
            raise ValueError('V1 requires verified CRC16/MODBUS and vendor finger/tactile layout')
    elif is_reference(config):
        # This is an explicit reference protocol, not a disabled PDF checksum.
        if (config.get('length_field_value') != 591
                or config.get('checksum_scope') != 'not_defined_by_reference'
                or config.get('checksum_confirmed') is not False
                or config.get('integrity_policy') != 'reference_structure_no_checksum'
                or config.get('tactile_layout_confirmed') is not False
                or config.get('tactile_required') is not False):
            raise ValueError('593 reference requires explicit unverified checksum/tactile policy')
        if order != ['thumb', 'index', 'middle', 'ring', 'pinky']:
            raise ValueError('593 reference finger order must match colleague example')
    elif config.get('checksum_scope') != 'all_bytes_before_checksum':
        raise ValueError('unsupported checksum scope; obtain firmware definition first')
    if not isinstance(config.get('length_field_value'), int) or not 0 <= config['length_field_value'] <= 65535:
        raise ValueError('invalid length field value')
    required = ('finger_order_confirmed', 'length_confirmed') if is_reference(config) else (
        'finger_order_confirmed', 'checksum_confirmed', 'length_confirmed')
    if config['protocol_confirmed'] and (not order or not all(config.get(k) is True for k in required)):
        raise ValueError('confirmed protocol requires confirmed finger order, length and checksum')


def decode_packet(packet, config):
    if config.get('protocol') == V1_PROTOCOL:
        return decode_frame(packet, config)
    if is_reference(config):
        return decode_reference_packet(packet, config)
    if len(packet) != PACKET_SIZE:
        raise ValueError('packet_size_mismatch')
    magic, kind, length, data_type = struct.unpack_from('<HBHB', packet)
    if (magic, kind, length, data_type) != (0x5aa5, 1, config['length_field_value'], 0x71):
        raise ValueError('header_mismatch')
    if sum(packet[:-2]) & 0xffff != struct.unpack_from('<H', packet, len(packet)-2)[0]:
        raise ValueError('checksum_mismatch')
    emf = packet[466:546]
    sequence, frame, mask, hand = struct.unpack_from('<IIBB', emf)
    if hand not in (0, 1):
        raise ValueError('invalid_hand_side')
    if mask & ~0x1f:
        raise ValueError('invalid_finger_mask')
    if ('right' if hand == 0 else 'left') not in capture_sides(config):
        raise ValueError('unexpected_hand_side')
    poses = [list(struct.unpack_from('<7h', emf, 10+i*14)) for i in range(5)]
    decoded = [[*(v * 1e-5 for v in p[:3]), *(v / 32767. for v in p[3:])] for p in poses]
    valid = [bool(mask & (1<<i)) and sum(v*v for v in p[3:]) > 1e-16 for i,p in enumerate(decoded)]
    return dict(side='right' if hand == 0 else 'left', stream_sequence=sequence,
                frame_sequence=frame, valid_mask=mask, finger_valid=valid,
                finger_pose_i16=poses, finger_pose_m_wxyz=decoded,
                tactile_u8=list(packet[6:466]), packet_hex=packet.hex(),
                checksum_valid=True, protocol_confirmed=bool(config.get('protocol_confirmed')))


def decode_reference_packet(packet, config):
    """Colleague's 593-byte EMF format; unknown touch/trailer remain raw evidence."""
    if len(packet) != 593:
        raise ValueError('packet_size_mismatch')
    if packet[:3] != b'\xa5\x5a\x01' or int.from_bytes(packet[5:7], 'little') != 591:
        raise ValueError('header_mismatch')
    emf = packet[494:574]
    sequence, frame, mask, hand = struct.unpack_from('<IIBB', emf)
    if hand not in (0, 1):
        raise ValueError('invalid_hand_side')
    if ('right' if hand == 0 else 'left') not in capture_sides(config):
        raise ValueError('unexpected_hand_side')
    if mask & ~31:
        raise ValueError('invalid_finger_mask')
    poses = [list(struct.unpack_from('<7h', emf, 10+i*14)) for i in range(5)]
    decoded, norms, valid = [], [], []
    for i, p in enumerate(poses):
        q = [v/32767. for v in p[3:]]
        norm = math.sqrt(sum(v*v for v in q))
        norms.append(norm)
        ok = bool(mask & (1 << i)) and norm > 1e-9
        valid.append(ok)
        decoded.append([*(v/100000. for v in p[:3]), *(v/norm if norm > 1e-9 else 0. for v in q)])
    return dict(side='right' if hand == 0 else 'left', stream_sequence=sequence,
        frame_sequence=frame, valid_mask=mask, finger_valid=valid,
        finger_pose_i16=poses, finger_pose_m_wxyz=decoded, raw_quaternion_norm=norms,
        tactile_u8=list(packet[32:492]), tactile_layout_verified=False,
        packet_hex=packet.hex(), frame_structure_valid=True,
        checksum_valid=False, checksum_status='not_defined_by_reference',
        protocol_confirmed=bool(config.get('protocol_confirmed')))


class TactileFramer:
    def __init__(self, config):
        self.config, self.buffer, self.counts = config, bytearray(), Counter()
        self.observed_header_hex = ''

    def feed(self, data):
        self.buffer.extend(data)
        packets = []
        while True:
            start = self.buffer.find(MAGIC)
            if start < 0:
                # Retain a split two-byte magic; bound noise buffering.
                keep = 1 if self.buffer.endswith(MAGIC[:1]) else 0
                self.counts['discarded_bytes'] += len(self.buffer)-keep
                self.buffer = self.buffer[-1:] if keep else bytearray()
                break
            self.counts['discarded_bytes'] += start
            del self.buffer[:start]
            reference = is_reference(self.config)
            v1 = self.config.get('protocol') == V1_PROTOCOL
            if len(self.buffer) < (7 if reference or v1 else 6):
                break
            self.observed_header_hex = bytes(self.buffer[:6]).hex(' ')
            if v1:
                header_ok = (self.buffer[:5] == b'\xa5\x5a\x01\xfd\xfe' and
                    22 <= int.from_bytes(self.buffer[5:7], 'little') + 2 <= MAX_FRAME_SIZE)
            elif reference:
                header_ok = self.buffer[2] == 1 and int.from_bytes(self.buffer[5:7], 'little') == 591
            else:
                header_ok = (self.buffer[2] == 1 and self.buffer[5] == 0x71 and
                    int.from_bytes(self.buffer[3:5], 'little') == self.config['length_field_value'])
            if not header_ok:
                self.counts['header_mismatch'] += 1
                del self.buffer[0]
                continue
            size = int.from_bytes(self.buffer[5:7], 'little') + 2 if v1 else 593 if reference else PACKET_SIZE
            if len(self.buffer) < size:
                break
            try:
                event = decode_packet(bytes(self.buffer[:size]), self.config)
            except ValueError as error:
                if v1 and str(error) in ('non_stream_frame', 'emf_not_subscribed', 'stream_status_error'):
                    self.counts[str(error)] += 1
                    del self.buffer[:size]
                    continue
                if str(error) == 'unexpected_hand_side':
                    self.counts['unselected_hand_packets'] += 1
                    del self.buffer[:size]
                    continue
                self.counts[str(error)] += 1
                del self.buffer[0]
                continue
            del self.buffer[:size]
            self.counts['decoded_packets'] += 1
            packets.append(event)
        return packets


def snapshot(event, config):
    """Publish explicit channels; a bad EMF point cannot invalidate touch."""
    confirmed = config.get('protocol_confirmed') is True and (
        event.get('frame_structure_valid') is True if is_reference(config) else event.get('checksum_valid') is True)
    tips_ok = confirmed and bool(config.get('finger_order')) and all(event['finger_valid'])
    common = dict(stream_sequence=event['stream_sequence'], frame_sequence=event['frame_sequence'],
                  receive_monotonic_ns=event['host_receive_monotonic_ns'])
    return dict(schema='independent.tactile_emf.snapshot.v1', side=event['side'],
        vendor=config['vendor'], valid=tips_ok, packet=event,
        source_monotonic_ns=event['host_receive_monotonic_ns'], source_index=event['stream_sequence'],
        tactile_config_sha256=config['sha256'],
        fingertips=dict(common, pose_i16=event['finger_pose_i16'],
                        point_valid=event['finger_valid'], valid_mask=event['valid_mask'], valid=tips_ok),
        tactile=dict(common, values_u8=event['tactile_u8'], valid=confirmed and event.get('tactile_present', True) and not is_reference(config)))


def validate_event(event, config):
    """Check typed values against wire bytes before they enter sensor arrays."""
    require_ready_config(config)
    if event.get('config_sha256') != config['sha256']:
        raise ValueError('configuration_mismatch')
    decoded = decode_packet(bytes.fromhex(event['packet_hex']), config)
    for field in ('side', 'stream_sequence', 'frame_sequence', 'valid_mask', 'finger_valid',
                  'finger_pose_i16', 'tactile_u8', 'checksum_valid', 'protocol_confirmed'):
        if event.get(field) != decoded[field]:
            raise ValueError('decoded_field_mismatch')
    if config.get('protocol') == V1_PROTOCOL:
        for field in decoded:
            if event.get(field) != decoded[field]:
                raise ValueError('decoded_field_mismatch')
    if is_reference(config):
        for field in ('frame_structure_valid', 'checksum_status', 'tactile_layout_verified',
                      'raw_quaternion_norm', 'finger_pose_m_wxyz'):
            if event.get(field) != decoded[field]:
                raise ValueError('decoded_field_mismatch')
    return decoded


def channel_validity(value, config, now_ns, max_age_ns=250_000_000):
    """Revalidate the actual packet; never trust a JSON validity bit alone."""
    invalid = dict(packet=False, fingertips=False, tactile=False)
    if not isinstance(value, dict) or not isinstance(value.get('packet'), dict):
        return invalid
    event = value['packet']
    try:
        age = now_ns - int(event['host_receive_monotonic_ns'])
        require_ready_config(config)
        if not (value.get('schema') == 'independent.tactile_emf.snapshot.v1'
                and value.get('vendor') == config['vendor']
                and value.get('side') in capture_sides(config)
                and event.get('side') == value.get('side')
                and value.get('tactile_config_sha256') == config['sha256']
                and event.get('config_sha256') == config['sha256']
                and 0 <= age <= max_age_ns):
            return invalid
        decoded = validate_event(event, config)
        # The two channels come from the same checked wire frame. Their quality
        # flags remain independent: touch is usable even when an EMF point drops.
        return dict(packet=True, fingertips=all(decoded['finger_valid']), tactile=decoded.get('tactile_present', True) and not is_reference(config))
    except (KeyError, TypeError, ValueError, OverflowError):
        return invalid


def tactile_snapshot_valid(value, config, now_ns):
    flags = channel_validity(value, config, now_ns)
    return capture_channels_ready(flags, config)


class EventTail:
    """Read append-only native events without blocking the 30 Hz sampler."""
    def __init__(self, path, *, from_end=False):
        self.path, self.offset, self.partial, self.pending = Path(path), 0, b'', []
        self.skip_initial_partial = False
        if from_end and self.path.is_file():
            with self.path.open('rb') as source:
                source.seek(0, 2)
                self.offset = source.tell()
                if self.offset:
                    source.seek(self.offset-1)
                    self.skip_initial_partial = source.read(1) != b'\n'

    def read(self, start_ns, end_ns=None):
        if not self.path.is_file():
            return []
        with self.path.open('rb') as source:
            source.seek(self.offset)
            data = source.read(4*1024*1024)
            self.offset = source.tell()
        lines = (self.partial + data).split(b'\n')
        self.partial = lines.pop()
        if self.skip_initial_partial and lines:
            lines.pop(0)
            self.skip_initial_partial = False
        events = self.pending + [json.loads(line) for line in lines if line]
        self.pending = [e for e in events if end_ns is not None and e['host_receive_monotonic_ns'] > end_ns]
        return [e for e in events if start_ns <= e['host_receive_monotonic_ns']
                and (end_ns is None or e['host_receive_monotonic_ns'] <= end_ns)]


def require_ready_config(config):
    validate_config(config)
    if not config.get('protocol_confirmed') or not config.get('finger_order'):
        raise ValueError('独立触觉手套协议尚未确认：请核对实际帧格式、长度、校验范围及五指顺序；诊断数据不能作为有效采集')


def write_protocol_calibration(file, config, *, side=None):
    group = file.require_group('calibration').require_group('tactile_emf')
    if side is not None:
        group = group.require_group(side)
    group.attrs.update(schema='independent.tactile_emf.calibration.v1',
        protocol_confirmed=config.get('protocol_confirmed', False),
        sha256=config.get('sha256', ''), expected_side=config['expected_side'],
        vendor=config['vendor'], tactile_units='native_uint8_counts; not Newtons',
        emf_position_scale_m=1e-5, emf_quaternion_scale=1.0/32767,
        emf_quaternion_order='wxyz', device_timestamp_available=False,
        tactile_layout='23x20; wire index 0..19 is bottom row in PDF; physical force not calibrated')
    if is_reference(config):
        group.attrs.update(protocol=config['protocol'], protocol_confirmation_scope='frame and EMF layout only',
            frame_size_bytes=593, length_field_offset=5, emf_offset=494, emf_size_bytes=80,
            checksum_verified=False, integrity_policy=config['integrity_policy'],
            tactile_layout_verified=False, tactile_required=False,
            tactile_layout='unverified candidate bytes [32:492]; no physical channel mapping',
            unknown_bytes_policy='complete 593-byte packet preserved in transport/packets packet_hex')
    if config.get('protocol') == V1_PROTOCOL:
        group.attrs.update(protocol=V1_PROTOCOL, checksum_verified=True, integrity_policy='crc16_modbus',
            emf_size_bytes=97, emf_offset_policy='dynamic subscription bit4',
            tactile_layout_verified=True, tactile_required=tactile_required(config),
            sequence_semantics='stream_sequence=frame_sequence=outer board uint16 at8; modulo65536; not EMF sample counter',
            emf_reserved_semantics='EMF bytes0..7 reserved; never sequence/timestamp',
            board_timestamp_semantics='bit0 uint64 ms; epoch specified but clock synchronization unverified',
            device_timestamp_available=True, device_clock_synchronized=False,
            emf_imu_clock='uint32 module tick_ms; modulo2^32, independent from board clock/finger samples',
            emf_imu_gyro_scale_rad_s=0.035*math.pi/180, emf_imu_accel_scale_m_s2=0.000122*9.80665,
            emf_imu_flags_policy='usable: (flags &7)==7 and acceleration not all zero; no integration performed',
            emf_sample_freshness='host receive freshness only; board may repeat cached finger poses; no finger timestamp provided')
    group.create_dataset('protocol_config_json', data=json.dumps(config, ensure_ascii=False))
