"""Vendor V1 subscription frames and EMF97; read-only, no firmware commands."""
import math
import struct

PROTOCOL = 'independent_glove_v1_emf97'
MAX_FRAME_SIZE = 8192


def _crc_entry(value):
    for _ in range(8):
        value = (value >> 1) ^ (0xa001 if value & 1 else 0)
    return value


_CRC_TABLE = tuple(_crc_entry(i) for i in range(256))


def crc16_modbus(data):
    crc = 0xffff
    for value in data:
        crc = (crc >> 8) ^ _CRC_TABLE[(crc ^ value) & 255]
    return crc


def decode_emf97(block):
    if len(block) != 97:
        raise ValueError('emf_block_size_mismatch')
    mask, hand = block[8:10]
    if hand not in (0, 1):
        raise ValueError('invalid_hand_side')
    poses = [list(struct.unpack_from('<7h', block, 10+i*14)) for i in range(5)]
    converted, norms, valid = [], [], []
    for i, pose in enumerate(poses):
        quat = [v/32767. for v in pose[3:]]
        norm = math.sqrt(sum(v*v for v in quat))
        norms.append(norm)
        valid.append(bool(mask & (1 << i)) and norm > 1e-9)
        converted.append([*(v*1e-5 for v in pose[:3]),
                          *(v/norm if norm > 1e-9 else 0. for v in quat)])
    gyro = list(struct.unpack_from('<3h', block, 81))
    accel = list(struct.unpack_from('<3h', block, 87))
    flags = block[80]
    return dict(side='right' if hand == 0 else 'left', valid_mask=mask & 31,
        valid_mask_raw=mask, emf_reserved_u8=list(block[:8]),
        finger_valid=valid, finger_pose_i16=poses, finger_pose_m_wxyz=converted,
        raw_quaternion_norm=norms, emf_imu=dict(flags=flags, flags_valid=(flags & 7) == 7,
            usable=(flags & 7) == 7 and any(accel), gyro_i16=gyro, accel_i16=accel,
            tick_ms=struct.unpack_from('<I', block, 93)[0]))


def decode_frame(packet, config):
    if not 22 <= len(packet) <= MAX_FRAME_SIZE or int.from_bytes(packet[5:7], 'little') + 2 != len(packet):
        raise ValueError('packet_size_mismatch')
    if packet[:5] != b'\xa5\x5a\x01\xfd\xfe':
        raise ValueError('header_mismatch')
    if crc16_modbus(packet[:-2]) != int.from_bytes(packet[-2:], 'little'):
        raise ValueError('checksum_mismatch')
    if packet[7] != 2 or packet[12:15] != b'\x03\x03\x00':
        raise ValueError('non_stream_frame')
    if packet[15] != 0:
        raise ValueError('stream_status_error')
    bitmap = int.from_bytes(packet[16:20], 'little')
    cursor, blocks, offsets = 20, {}, {}
    for bit in range(32):
        if not bitmap & (1 << bit):
            continue
        if cursor + 2 > len(packet)-2:
            raise ValueError('block_length_missing')
        size = int.from_bytes(packet[cursor:cursor+2], 'little')
        cursor += 2
        if cursor+size > len(packet)-2:
            raise ValueError('block_truncated')
        blocks[bit], offsets[str(bit)] = packet[cursor:cursor+size], cursor
        cursor += size
    if cursor != len(packet)-2:
        raise ValueError('stream_trailing_bytes')
    for bit, size in ((0, 8), (1, 460), (2, 40), (3, 10), (4, 97)):
        if bit in blocks and len(blocks[bit]) != size:
            raise ValueError('emf_block_size_mismatch' if bit == 4 else 'block_size_mismatch')
    if 4 not in blocks:
        raise ValueError('emf_not_subscribed')
    event = decode_emf97(blocks[4])
    if event['side'] not in config.get('capture_sides', [config['expected_side']]):
        raise ValueError('unexpected_hand_side')
    sequence = int.from_bytes(packet[8:10], 'little')
    event.update(board_sequence_u16=sequence,
        # Compatibility aliases, explicitly documented as the same BOARD counter.
        stream_sequence=sequence, frame_sequence=sequence,
        board_timestamp_present=0 in blocks,
        board_timestamp_ms=int.from_bytes(blocks[0], 'little') if 0 in blocks else 0,
        subscription_bitmap=bitmap, block_offsets=offsets,
        block_lengths={str(bit): len(data) for bit, data in blocks.items()},
        tactile_present=1 in blocks, tactile_u8=list(blocks.get(1, bytes(460))),
        tactile_layout_verified=1 in blocks, checksum_valid=True, checksum_status='crc16_modbus',
        frame_structure_valid=True, packet_hex=packet.hex(),
        protocol_confirmed=bool(config.get('protocol_confirmed')))
    return event
