#!/usr/bin/env python3
"""Generate decoder_test testdata for the dshot and elrs decoders.

Writes testdata/<name>/{config.json,input.bin} in the format which
decoder_test.c reads, next to the script. Run decoder_test with the
decoder name and that directory, and add --python for the Python
decoders::

    ./decoder_test -d dshot -t testdata/dshot --python
    ./decoder_test -d elrs -t testdata/elrs --python

Only the dshot case ships reference output. A stacked run also contains
the annotations of the upstream decoder, so keeping them as expected
output would make this test fail whenever the uart decoder's text
changes. Generate the reference for the elrs case on demand with
--generate-only and inspect it instead.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'testdata')

GCR_SYMBOLS = (
    0x19, 0x1b, 0x12, 0x13, 0x1c, 0x15, 0x16, 0x17,
    0x1a, 0x09, 0x0a, 0x0b, 0x1e, 0x0d, 0x0e, 0x0f,
)


def pack(bits):
    data = bytearray((len(bits) + 7) // 8)
    for idx, bit in enumerate(bits):
        if bit:
            data[idx // 8] |= 1 << (idx % 8)
    return bytes(data)


def rasterize(edges, total):
    bits = [0] * total
    level = 0
    pos = 0
    for sample, new_level in edges:
        sample = min(sample, total)
        for idx in range(pos, sample):
            bits[idx] = level
        level = new_level
        pos = sample
    for idx in range(pos, total):
        bits[idx] = level
    return bits


def write_case(name, samplerate, edges, total, config):
    bits = rasterize(edges, total)
    path = os.path.join(OUT, name)
    os.makedirs(path, exist_ok=True)
    config = dict(config)
    config['samplerate'] = samplerate
    config['num_channels'] = 1
    config['sample_count'] = len(bits)
    with open(os.path.join(path, 'config.json'), 'w') as handle:
        json.dump(config, handle, indent=2)
        handle.write('\n')
    with open(os.path.join(path, 'input.bin'), 'wb') as handle:
        handle.write(pack(bits))
    print('{}: {} samples, {} bytes packed'.format(
        name, len(bits), len(pack(bits))))


def crc4(value):
    value &= 0xfff
    return (value ^ (value >> 4) ^ (value >> 8)) & 0xf


def dshot_bits(value16, count):
    return [(value16 >> (count - 1 - idx)) & 1 for idx in range(count)]


def dshot_frame(edges, pos, samplerate, rate, value16):
    period = int(round(samplerate / rate))
    for bit in dshot_bits(value16, 16):
        high = int(round(period * (0.75 if bit else 0.375)))
        edges.append((pos, 1))
        edges.append((pos + high, 0))
        pos += period
    return pos


def dshot_reply(edges, pos, samplerate, rate, value16):
    period = int(round(samplerate / rate))
    nibbles = [(value16 >> (4 * idx)) & 0xf for idx in range(3, -1, -1)]
    bits = [0]
    for nibble in nibbles:
        bits.extend(dshot_bits(GCR_SYMBOLS[nibble], 5))
    pos += int(round(period * 20))
    edges.append((pos, 1))
    pos += int(round(period * 8))
    for bit in bits:
        low = int(round(period * (0.75 if bit else 0.375)))
        edges.append((pos, 0))
        edges.append((pos + low, 1))
        pos += period
    return pos


def generate_dshot():
    samplerate = 24000000
    rate = 600e3
    edges = []
    # Start with an idle low level, a capture which begins in the middle
    # of a pulse cannot be decoded.
    pos = 200
    # Throttle 1000 with the telemetry request bit, checksum 0xb.
    throttle = 0x7d1b
    command = (12 << 1) << 4 | crc4(12 << 1)
    payload = 100
    reply = (crc4(payload) << 12) | payload
    pos = dshot_frame(edges, pos, samplerate, rate, throttle)
    pos = dshot_reply(edges, pos, samplerate, rate, reply)
    pos += 3000
    edges.append((pos, 0))
    pos += 3000
    pos = dshot_frame(edges, pos, samplerate, rate, command)
    pos += 6000
    pos = dshot_frame(edges, pos, samplerate, rate, throttle)
    total = pos + 6000
    write_case('dshot', samplerate, edges, total, {
        'decoder': 'dshot',
        'channels': {'data': 0},
        'options': {},
    })


def crc8(data):
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0xd5) & 0xff if crc & 0x80 else (crc << 1) & 0xff
    return crc


def crsf_frame(body):
    frame = bytes([0xc8, len(body) + 1]) + bytes(body)
    return frame + bytes([crc8(frame[2:])])


def generate_elrs():
    samplerate = 12000000
    baud = 115200
    # 16 channels, endpoints and centre, plus the ELRS armed byte.
    channels = [172, 992, 1811] + [992] * 13
    packed = 0
    for idx, value in enumerate(channels):
        packed |= value << (11 * idx)
    payload = packed.to_bytes(22, 'little') + bytes([0x03])
    data = crsf_frame(bytes([0x16]) + payload)
    # Link statistics: RSSI 60/62, LQ 100, SNR 8, antenna 0, mode 27
    # (2.4 GHz 250 Hz), TX power 3 (100 mW), downlink 55/97/5.
    data += crsf_frame(bytes([0x14, 60, 62, 100, 8, 0, 27, 3, 55, 97, 5]))
    # ELRS status: 3 bad, 300 good packets, connected and armed.
    body = bytes([0x2e, 0xea, 0xee, 3, 0x01, 0x2c, 0x09])
    body += b'[ ! Armed ! ]\x00'
    data += crsf_frame(body)
    # Bind command from the flight controller to the receiver.
    data += crsf_frame(bytes([0x32, 0xec, 0xc8, 0x10, 0x01]))

    edges = []
    bit_period = samplerate / baud
    pos = 0
    edges.append((0, 1))
    pos = 20
    for byte in data:
        levels = [0]
        levels.extend([(byte >> idx) & 1 for idx in range(8)])
        levels.append(1)
        for level in levels:
            edges.append((int(round(pos)), level))
            pos += bit_period
    total = int(round(pos)) + 200
    write_case('elrs', samplerate, edges, total, {
        'decoder': 'elrs',
        'channels': {},
        'options': {},
        'stack': [{'id': 'uart', 'channels': {'rx': 0}}],
    })


if __name__ == '__main__':
    generate_dshot()
    generate_elrs()
