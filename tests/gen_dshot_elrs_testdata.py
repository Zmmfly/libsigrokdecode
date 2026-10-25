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

The cases of the native ports live in testdata/dshot_c/<case> and
testdata/elrs_c/<case>, the layout which run_tests.sh expects. Their
expected.json is the output of the Python decoder, which the port has
to reproduce. The elrs_c cases set "top_only", so that decoder_test
only compares the annotations of the elrs decoder and not those of the
uart decoder below it. Regenerate the references after changing a
case, from the directory of this script::

    export PY_DECODERS_PATH=$PWD/../decoders
    for case in testdata/dshot_c/*/ testdata/elrs_c/*/; do
        name=$(basename $(dirname $case)); name=${name%_c}
        decoder_test -d $name -t $case --python --generate-only \\
            -f $case/expected.json
    done

The elrs_c/two_lines case is the exception: the Python decoder mixes the
bytes of both UART lines, the port assembles them separately. Its
reference is the output of elrs_c, and its config sets
"expected_deviations". Generate it with -d elrs_c instead of -d elrs.
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


def write_case(name, samplerate, edges, total, config, num_channels=1):
    if num_channels == 1:
        channels = [rasterize(edges, total)]
    else:
        channels = [rasterize(chan, total) for chan in edges]
    path = os.path.join(OUT, name)
    os.makedirs(path, exist_ok=True)
    config = dict(config)
    config['samplerate'] = samplerate
    config['num_channels'] = num_channels
    config['sample_count'] = total
    with open(os.path.join(path, 'config.json'), 'w') as handle:
        json.dump(config, handle, indent=2)
        handle.write('\n')
    packed = b''.join(pack(bits) for bits in channels)
    with open(os.path.join(path, 'input.bin'), 'wb') as handle:
        handle.write(packed)
    print('{}: {} samples, {} bytes packed'.format(
        name, total, len(packed)))


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
    samplerate, edges, total = dshot_mixed_edges()
    write_case('dshot', samplerate, edges, total, {
        'decoder': 'dshot',
        'channels': {'data': 0},
        'options': {},
    })


def dshot_mixed_edges():
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
    return samplerate, edges, total


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


#-- Native decoder cases -----------------------------------------------

def symbol_pulses(edges, pos, period, duties, active=1):
    """One pulse per symbol, active for the given share of the period."""
    for duty in duties:
        edges.append((pos, active))
        edges.append((pos + int(round(period * duty)), 1 - active))
        pos += period
    return pos


def bit_duties(bits):
    return [0.75 if bit else 0.375 for bit in bits]


def gcr_bits(value16, high_first=True):
    nibbles = [(value16 >> (4 * idx)) & 0xf for idx in range(4)]
    if high_first:
        nibbles.reverse()
    bits = []
    for nibble in nibbles:
        bits.extend(dshot_bits(GCR_SYMBOLS[nibble], 5))
    return bits


def frame16(value12):
    return (value12 << 4) | crc4(value12)


def throttle16(value11, tlm=False):
    return frame16((value11 << 1) | (1 if tlm else 0))


def crc_valid(value16):
    return (value16 & 0xf) == crc4(value16 >> 4)


def request(edges, pos, period, value16, gap):
    pos = symbol_pulses(edges, pos, period, bit_duties(dshot_bits(value16, 16)))
    return pos + gap


def reply_burst(edges, pos, period, bits, gap, duties=None):
    """The line rises, then the ESC drives one low pulse per symbol.

    The line returns to the low idle level after the gap, that single
    edge burst does not produce annotations."""
    edges.append((pos, 1))
    pos += period * 8
    pos = symbol_pulses(edges, pos, period, duties or bit_duties(bits), 0)
    pos += gap
    edges.append((pos, 0))
    return pos + gap


def tlm_bits(raw12):
    return [0] + gcr_bits((crc4(raw12) << 12) | raw12)


def dshot_case(name, samplerate, edges, pos, options):
    # A trailing edge flushes the last burst.
    edges.append((pos, 1))
    edges.append((pos + 10, 0))
    write_case('dshot_c/' + name, samplerate, edges, pos + 2000, {
        'decoder': 'dshot_c',
        'channels': {'data': 0},
        'options': options,
    })


def generate_dshot_c():
    samplerate, edges, total = dshot_mixed_edges()
    write_case('dshot_c/mixed', samplerate, edges, total, {
        'decoder': 'dshot_c',
        'channels': {'data': 0},
        'options': {},
    })

    # DShot1200 GCR frames, telemetry types and the symbol bits. The XOR
    # checksum does not depend on the nibble order, so a frame is either
    # valid in both orders or in none.
    period, gap, edges, pos = 40, 2000, [], 200
    no_crc = next(v for v in range(0x10, 0x10000) if not crc_valid(v))
    for bits in (gcr_bits(0x7d1b), gcr_bits(0x7d1b, high_first=False),
                 [0] * 20, gcr_bits(no_crc)):
        pos = symbol_pulses(edges, pos, period, bit_duties(bits)) + gap
    for raw12 in ((2 << 9) | 300, 0x200 | 45, 0x400 | 50, 0x600 | 20,
                  0x800 | 7, 0xa00 | 8, 0xc00 | 9, 0xe00 | 3, 0, 0x064):
        pos = reply_burst(edges, pos, period, tlm_bits(raw12), gap)
    bits = tlm_bits(0x123)
    duties = bit_duties(bits)
    duties[bits.index(0, 1)] = 0.55
    pos = reply_burst(edges, pos, period, bits, gap, duties)
    pos = reply_burst(edges, pos, period, [1] + bits[1:], gap)
    dshot_case('gcr_bits', 48000000, edges, pos, {'bits': 'on', 'motor': '3'})

    # 3D throttle ranges, commands and a configured rate which does not
    # match the DShot600 frames.
    period, gap, edges, pos = 40, 2000, [], 200
    for value11, tlm in ((500, False), (1500, False), (1047, False),
                         (1048, False), (2047, True), (48, False), (0, False),
                         (40, False), (17, False), (1, False), (21, False),
                         (31, False), (13, True)):
        pos = request(edges, pos, period, throttle16(value11, tlm), gap)
    pos = request(edges, pos, period, throttle16(1000) ^ 0x0400, gap)
    pos = request(edges, pos, 80, throttle16(1000), gap)
    dshot_case('throttle_3d', 24000000, edges, pos,
               {'3d': 'on', 'rate': 'DShot300', 'motor': '8'})

    # Bursts which are no frames, ambiguous timing, unusual rates.
    period, gap, edges, pos = 40, 2000, [], 200
    pos = symbol_pulses(edges, pos, period, [0.75] * 10) + gap
    pos = symbol_pulses(edges, pos, period, [0.75] * 50) + gap
    duties = bit_duties(dshot_bits(throttle16(1000), 16))
    duties[duties.index(0.375)] = 0.55
    duties[duties.index(0.75)] = 0.6
    pos = symbol_pulses(edges, pos, period, duties) + gap
    pos = request(edges, pos, 53, throttle16(1000), gap)
    pos = request(edges, pos, 160, throttle16(300), gap)
    pos = request(edges, pos, 20, throttle16(2000), gap)
    pos = reply_burst(edges, pos, period, [1] + tlm_bits(100)[1:], gap)
    pos = reply_burst(edges, pos, period, [0] * 21, gap)
    dshot_case('errors', 24000000, edges, pos, {})

    # Without telemetry decoding a response is just another burst.
    period, gap, edges, pos = 40, 2000, [], 200
    pos = request(edges, pos, period, throttle16(1000, True), gap)
    pos = reply_burst(edges, pos, period, tlm_bits(100), gap)
    dshot_case('tlm_off', 24000000, edges, pos, {'tlm': 'off'})

    # 12.5 MHz makes the idle limit 62.5 samples, which rounds to 62. A
    # gap of 63 samples separates two frames, one of 62 does not.
    period, edges, pos = 83, [], 200
    for gap in (63, 63, 62, 2000):
        symbol_pulses(edges, pos, period, bit_duties(dshot_bits(throttle16(700), 16)))
        pos = edges[-1][0] + gap
    dshot_case('samplerate_12m5', 12500000, edges, pos, {})


def uart_levels(data, stop_errors=()):
    levels = []
    for idx, byte in enumerate(data):
        levels.append(0)
        levels.extend((byte >> bit) & 1 for bit in range(8))
        if idx in stop_errors:
            # A framing error, the line returns high after a while.
            levels.extend([0, 1, 1])
        else:
            levels.append(1)
    return levels


def uart_edges(levels, bit_period=100, start=200):
    edges = [(0, 1)]
    pos = start
    for level in levels:
        edges.append((pos, level))
        pos += bit_period
    return edges, pos


def crsf(address, body):
    frame = bytes([address, len(body) + 1]) + bytes(body)
    return frame + bytes([crc8(frame[2:])])


def channels_payload(values, extra=b''):
    packed = 0
    for idx, value in enumerate(values):
        packed |= value << (11 * idx)
    return packed.to_bytes(22, 'little') + extra


def device_info(name, version=(3, 4, 2), params=35, param_version=7):
    fields = (0x11223344).to_bytes(4, 'big')
    fields += (0x55667788).to_bytes(4, 'big')
    fields += ((version[0] << 24) | (version[1] << 16) |
               (version[2] << 8)).to_bytes(4, 'big')
    fields += bytes([params])
    if param_version is not None:
        fields += bytes([param_version])
    return bytes([0x29, 0xea, 0xec]) + name + b'\x00' + fields


def be16(value):
    return (value & 0xffff).to_bytes(2, 'big')


def be32(value):
    return (value & 0xffffffff).to_bytes(4, 'big')


def elrs_case(name, frames, options, stop_errors=()):
    data = b''.join(frames)
    edges, pos = uart_edges(uart_levels(data, stop_errors))
    write_case('elrs_c/' + name, 11520000, edges, pos + 400, {
        'decoder': 'elrs_c',
        'channels': {},
        'options': options,
        'stack': [{'id': 'uart_c', 'channels': {'rx': 0}}],
        'top_only': True,
    })


def generate_elrs_c():
    # 11.52 MHz gives exactly 100 samples per bit at 115200 baud, the
    # default rate of the uart decoders.
    sticks = [172, 992, 1811, 0, 2047, 1000, 1500, 500] + [992] * 8
    bad_link = bytearray(crsf(0xc8, [0x14, 60, 62, 100, 8, 0, 27, 3, 55, 97, 5]))
    bad_link[-1] ^= 0xff
    bad_link = bytes(bad_link)
    bad_channels = bytearray(crsf(0xc8, b'\x16' + channels_payload(sticks, b'\x01')))
    bad_channels[-1] ^= 0x55
    bad_channels = bytes(bad_channels)
    frames = [
        # Noise and a broadcast address with an implausible length.
        bytes([0x00, 0x55, 0xaa]),
        crsf(0xc8, b'\x16' + channels_payload(sticks, b'\x03')),
        crsf(0xc8, b'\x16' + channels_payload([992] * 16)),
        crsf(0xc8, b'\x16' + bytes(10)),
        crsf(0xc8, [0x14, 60, 62, 100, 8, 0, 27, 3, 55, 97, 5]),
        crsf(0xea, [0x14, 100, 0, 50, 0xf6, 1, 50, 9, 120, 10, 0x80]),
        crsf(0xc8, [0x14, 1, 2, 3]),
        crsf(0xea, [0x2e, 0xea, 0xee, 0, 0, 5, 0x00] + list(b'Caf\xe9 ok\x00')),
        crsf(0xea, [0x2e, 0xea, 0xee, 255, 0xff, 0xff, 0xff, 0]),
        crsf(0xea, [0x2e, 0xea, 0xee, 1, 2]),
        crsf(0xea, device_info(b'ExpressLRS RX')),
        crsf(0xea, device_info(b'R\xf6x', (1, 2, 3), 4, None)),
        crsf(0xea, device_info(b'Short')[:12]),
        crsf(0xea, [0x29, 0xea, 0xec] + list(b'NoNul')),
        crsf(0xea, [0x2b, 0xea, 0xee, 3, 0, 0, 9] + list(b'Packet Rate\x00') +
             list(b'25Hz;50Hz;100Hz\x00') + [1, 0, 2, 0]),
        crsf(0xea, [0x2b, 0xea, 0xee, 5, 1, 2, 0x89] + list(b'Telem\x00')),
        crsf(0xea, [0x2b, 0xea, 0xee, 6, 0, 0, 1, 0]),
        crsf(0xea, [0x2b, 0xea, 0xee, 7, 2, 0x41, 0x42, 0x43]),
        crsf(0xea, [0x2b, 0xea, 0xee, 7]),
        crsf(0xc8, [0x32, 0xec, 0xc8, 0x10, 0x01]),
        crsf(0xc8, [0x32, 0xec, 0xc8, 0x10, 0x05]),
        crsf(0xc8, [0x32, 0xec, 0xc8, 0x10, 0x07]),
        crsf(0xee, [0x32, 0xee, 0xea, 0x05, 0x01]),
        crsf(0xee, [0x32, 0xee, 0xea, 0x09]),
        crsf(0xc8, [0x32, 0xec, 0xc8, 0x10]),
        crsf(0xc8, [0x32, 0xec, 0xc8]),
        crsf(0xea, b'\x1e' + be16(-1234) + be16(5678) + be16(-31415)),
        crsf(0xea, [0x1e, 1, 2, 3]),
        crsf(0xea, b'\x02' + be32(-33856000) + be32(1512150000) + be16(123) +
             be16(27000) + be16(1100) + bytes([12])),
        crsf(0xea, [0x02, 1, 2, 3]),
        crsf(0xea, b'\x08' + be16(168) + be16(125) + (1300).to_bytes(3, 'big') + b'\x4b'),
        crsf(0xea, b'\x08' + be16(111) + be16(5) + (7).to_bytes(3, 'big')),
        crsf(0xea, [0x08, 1, 2, 3, 4, 5]),
        crsf(0xea, b'\x21ACRO\x00'),
        crsf(0xea, b'\x21\x00'),
        crsf(0xc8, [0x7a, 0xc8, 0xea, 0x30, 0x00, 0x64, 0x00, 0x02, 0xaa, 0xbb]),
        crsf(0xea, [0x7b, 0xea, 0xc8, 0x31, 0x01, 0x2c, 0x00, 0x00]),
        crsf(0xc8, [0x7c, 0xc8, 0xea, 0x32, 0x00, 0x05, 0x00, 0x01, 0x07]),
        crsf(0xc8, [0x7a, 0xc8, 0xea, 1, 2]),
        crsf(0xc8, [0x55, 1, 2, 3]),
        crsf(0xc8, [0x05, 9, 8]),
        crsf(0xc8, [0x28, 0xc8]),
        crsf(0x00, [0x28, 0x00, 0xea]),
        crsf(0xc8, [0x28, 0x55, 0x66]),
        crsf(0xc8, [0x0b] + list(range(60))),
        crsf(0xc8, [0x0b]),
        bad_link,
        # Lengths which are no lengths, the second one is an address.
        bytes([0xc8, 0x01]),
        bytes([0xc8]) + crsf(0xee, [0x32, 0xee, 0xea, 0x10, 0x01]),
    ]
    elrs_case('frames', frames, {})
    elrs_case('options', [
        crsf(0xc8, b'\x16' + channels_payload(sticks, b'\x02')),
        bad_link,
        bad_channels,
    ], {'crc': 'on', 'channels': 'on', 'usec': 'off'})
    # A framing error inside a frame drops that byte, the frame then ends
    # with the first byte of the next one.
    bind = crsf(0xc8, [0x32, 0xec, 0xc8, 0x10, 0x01])
    elrs_case('uart_error', [bind, bind, bind, bind], {},
              stop_errors=(3, 10))

    # Both UART lines at once, with overlapping frames. The port
    # assembles each line separately.
    rx = uart_levels(crsf(0xc8, [0x32, 0xec, 0xc8, 0x10, 0x01]) +
                     crsf(0xc8, [0x14, 60, 62, 100, 8, 0, 27, 3, 55, 97, 5]))
    tx = uart_levels(crsf(0xea, [0x2e, 0xea, 0xee, 3, 0x01, 0x2c, 0x09]) +
                     crsf(0xea, [0x32, 0xee, 0xea, 0x10, 0x05]))
    rx_edges, rx_end = uart_edges(rx, start=200)
    tx_edges, tx_end = uart_edges(tx, start=650)
    write_case('elrs_c/two_lines', 11520000, [rx_edges, tx_edges],
               max(rx_end, tx_end) + 400, {
                   'decoder': 'elrs_c',
                   'channels': {},
                   'options': {},
                   'stack': [{'id': 'uart_c', 'channels': {'rx': 0, 'tx': 1}}],
                   'top_only': True,
                   'expected_deviations': True,
               }, num_channels=2)


if __name__ == '__main__':
    generate_dshot()
    generate_elrs()
    generate_dshot_c()
    generate_elrs_c()
