#!/usr/bin/env python3
"""Generate decoder_test testdata for the dshot, elrs and crsf decoders.

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

The cases of the native ports live in testdata/dshot_c/<case>,
testdata/elrs_c/<case> and testdata/crsf_c/<case>, the layout which
run_tests.sh expects. Their expected.json is the output of the Python
decoder, which the port has to reproduce. The elrs_c and crsf_c cases set
"top_only", so that decoder_test only compares the annotations of the
top decoder and not those of the uart decoder below it. Regenerate the
references after changing a case, from the directory of this script::

    export PY_DECODERS_PATH=$PWD/../decoders
    for case in testdata/dshot_c/*/ testdata/elrs_c/*/ testdata/crsf_c/*/; do
        name=$(basename $(dirname $case)); name=${name%_c}
        decoder_test -d $name -t $case --python --generate-only \\
            -f $case/expected.json
    done

The elrs_c/two_lines case is the exception: the Python decoder mixes the
bytes of both UART lines, the port assembles them separately. Its
reference is the output of elrs_c, and its config sets
"expected_deviations". Generate it with -d elrs_c instead of -d elrs.

The dshot_c/capture_* cases are not generated: they are cropped from logic
analyzer captures of real hardware (an N32H474 DShot generator with
injected errors, decimated to 25 MHz). Their reference is the output of
the Python decoder as well.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'testdata')

GCR_SYMBOLS = (
    0x19, 0x1b, 0x12, 0x13, 0x1d, 0x15, 0x16, 0x17,
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


def dshot_word(value11, tlm=False, inverted=False, crc_xor=0):
    """16 bit request: value, telemetry request bit and checksum, which
    bidirectional DShot inverts."""
    value12 = (value11 << 1) | (1 if tlm else 0)
    crc = crc4(value12)
    if inverted:
        crc ^= 0xf
    return (value12 << 4) | (crc ^ crc_xor)


def dshot_bits(value16, count=16):
    return [(value16 >> (count - 1 - idx)) & 1 for idx in range(count)]


# T0H/T1H as shares of the bit period: DShot specification, Betaflight
# timer DMA (7/20, 14/20) and Betaflight bit-banged outputs (1/3, 2/3).
DUTIES = {'spec': (0.375, 0.75), 'bf': (0.35, 0.70), 'bb': (1 / 3, 2 / 3)}


def dshot_request(edges, pos, period, value16, inverted=False, duty='spec',
                  duties=None):
    """One request frame. An inverted (bidirectional) line rests high and
    every bit starts with the low level."""
    active = 0 if inverted else 1
    zero, one = DUTIES[duty]
    for idx, bit in enumerate(dshot_bits(value16)):
        share = duties[idx] if duties else (one if bit else zero)
        edges.append((int(round(pos)), active))
        edges.append((int(round(pos + period * share)), 1 - active))
        pos += period
    return pos


def tlm_word(value12, csum_xor=0):
    """16 bit response word: the four nibbles XOR to 0xf."""
    csum = ~(value12 ^ (value12 >> 4) ^ (value12 >> 8)) & 0xf
    return (value12 << 4) | (csum ^ csum_xor)


def gcr20(word16, bad_first_symbol=False):
    gcr = 0
    for idx in range(3, -1, -1):
        gcr = (gcr << 5) | GCR_SYMBOLS[(word16 >> (4 * idx)) & 0xf]
    if bad_first_symbol:
        gcr &= ~(0x1f << 15)
    return gcr


def dshot_response(edges, pos, tb, gcr):
    """The ESC's 21 bits: a low start bit, then the 20 GCR bits as NRZI
    (a level change is a '1'). The line is released high at the end."""
    level = 0
    edges.append((int(round(pos)), 0))
    for idx in range(20):
        if (gcr >> (19 - idx)) & 1:
            level ^= 1
            edges.append((int(round(pos + (idx + 1) * tb)), level))
    if level == 0:
        edges.append((int(round(pos + 21 * tb)), 1))
    return pos + 21 * tb


def erpm12(period_us):
    """eeem mmmm mmmm: commutation period as mantissa << exponent."""
    exp = 0
    while (period_us >> exp) > 511:
        exp += 1
    return (exp << 9) | (period_us >> exp)


def dshot_exchange(edges, pos, samplerate, kbit, value16, value12=None,
                   delay_us=30.0, clock=1.0, csum_xor=0, bad_gcr=False,
                   duty='spec'):
    """Inverted request, then the ESC's response unless value12 is None.
    `clock` scales the response bit time like an ESC clock error."""
    period = samplerate / (kbit * 1e3)
    pos = dshot_request(edges, pos, period, value16, True, duty)
    if value12 is not None:
        tb = period * 0.8 * clock
        dshot_response(edges, pos + samplerate * delay_us * 1e-6, tb,
                       gcr20(tlm_word(value12, csum_xor), bad_gcr))
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
    period = samplerate / 600e3
    edges = []
    # DShot600 on a line which rests low: throttle 1000 with the
    # telemetry request bit, the save settings command, motor stop.
    pos = 200
    pos = dshot_request(edges, pos, period, dshot_word(1000, True)) + 3000
    pos = dshot_request(edges, pos, period, dshot_word(12, True)) + 3000
    pos = dshot_request(edges, pos, period, dshot_word(0)) + 3000
    # Bidirectional DShot600: the line rests high, the ESC answers each
    # frame 30 us later with the commutation period or an EDT frame.
    edges.append((int(pos), 1))
    pos += 3000
    pos = dshot_exchange(edges, pos, samplerate, 600, dshot_word(1000, False, True),
                         erpm12(2000)) + 3000
    pos = dshot_exchange(edges, pos, samplerate, 600, dshot_word(48, True, True),
                         0x200 | 45) + 3000
    total = int(pos) + 3000
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

def dshot_case(name, samplerate, edges, pos, options):
    write_case('dshot_c/' + name, samplerate, edges, int(pos) + 2000, {
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

    # Bidirectional DShot600 at 48 MHz with the bit annotations: every
    # exponent of the commutation period, motor stopped, every EDT type,
    # telemetry checksum and GCR errors, a silent ESC, ESC clocks 5 % off,
    # early and late responses, and a request with a non-inverted CRC.
    sr, edges, pos = 48000000, [(0, 1)], 400
    gap = 6000
    for period_us in (20, 500, 1000, 5000, 20000, 65000):
        pos = dshot_exchange(edges, pos, sr, 600, dshot_word(1000, False, True),
                             erpm12(period_us)) + gap
    pos = dshot_exchange(edges, pos, sr, 600, dshot_word(0, False, True), 0xfff) + gap
    for raw12 in (0x200 | 45, 0x400 | 50, 0x600 | 20, 0x800 | 7, 0xa00 | 8,
                  0xc00 | 9, 0xe00 | 3):
        pos = dshot_exchange(edges, pos, sr, 600, dshot_word(48, True, True),
                             raw12) + gap
    pos = dshot_exchange(edges, pos, sr, 600, dshot_word(700, False, True),
                         erpm12(3000), csum_xor=0x5) + gap
    pos = dshot_exchange(edges, pos, sr, 600, dshot_word(700, False, True),
                         erpm12(3000), bad_gcr=True) + gap
    pos = dshot_exchange(edges, pos, sr, 600, dshot_word(700, False, True), None) + gap
    for clock in (1.05, 0.95):
        pos = dshot_exchange(edges, pos, sr, 600, dshot_word(1500, False, True),
                             erpm12(777), clock=clock) + gap
    for delay in (20.0, 45.0):
        pos = dshot_exchange(edges, pos, sr, 600, dshot_word(1500, False, True),
                             erpm12(888), delay_us=delay) + gap
    pos = dshot_exchange(edges, pos, sr, 600, dshot_word(1000, False, False),
                         erpm12(999)) + gap
    dshot_case('bidir', sr, edges, pos, {'bits': 'on', 'motor': '3'})

    # Every rate with the three duty cycle schemes, on a line which rests
    # low and on a bidirectional one.
    sr, edges, pos = 48000000, [], 400
    for kbit in (150, 300, 600, 1200):
        for duty in ('spec', 'bf', 'bb'):
            pos = dshot_request(edges, pos, sr / (kbit * 1e3), dshot_word(1234, True),
                                duty=duty) + 4000
    edges.append((int(pos), 1))
    pos += 4000
    for kbit in (150, 300, 600, 1200):
        for duty in ('spec', 'bf', 'bb'):
            pos = dshot_exchange(edges, pos, sr, kbit, dshot_word(321, False, True),
                                 erpm12(1234), duty=duty) + 8000
    dshot_case('rates', sr, edges, pos, {})

    # 3D throttle ranges, commands and a configured rate which does not
    # match the DShot600 frames.
    period, edges, pos = 40, [], 200
    for value11, tlm in ((500, False), (1500, False), (1047, False),
                         (1048, False), (2047, True), (48, False), (0, False),
                         (40, False), (17, False), (1, False), (21, False),
                         (31, False), (13, True), (33, True), (47, True)):
        pos = dshot_request(edges, pos, period, dshot_word(value11, tlm)) + 2000
    pos = dshot_request(edges, pos, period, dshot_word(1000) ^ 0x0400) + 2000
    pos = dshot_request(edges, pos, 80, dshot_word(1000)) + 2000
    dshot_case('throttle_3d', 24000000, edges, pos,
               {'3d': 'on', 'rate': 'DShot300', 'motor': '8'})

    # Bursts which are no frames, ambiguous timing, unusual rates and
    # responses which do not fit into 21 bits.
    period, edges, pos = 40, [], 200
    for count in (10, 50):
        for _ in range(count):
            edges.append((int(pos), 1))
            edges.append((int(pos + period * 0.75), 0))
            pos += period
        pos += 2000
    duties = [0.375 if bit == 0 else 0.75 for bit in dshot_bits(dshot_word(1000))]
    duties[duties.index(0.375)] = 0.55
    duties[duties.index(0.75)] = 0.6
    pos = dshot_request(edges, pos, period, dshot_word(1000), duties=duties) + 2000
    pos = dshot_request(edges, pos, 53, dshot_word(1000)) + 2000
    pos = dshot_request(edges, pos, 160, dshot_word(300)) + 2000
    pos = dshot_request(edges, pos, 20, dshot_word(2000)) + 2000
    edges.append((int(pos), 1))
    pos += 2000
    edges.append((int(pos), 0))
    edges.append((int(pos + 30), 1))
    pos += 2000
    pos = dshot_exchange(edges, pos, 24000000, 600, dshot_word(900, False, True), None)
    start = pos + 720
    dshot_response(edges, start, 32, gcr20(tlm_word(erpm12(500))))
    edges.append((int(start + 23 * 32), 0))
    edges.append((int(start + 24 * 32), 1))
    pos = start + 6000
    dshot_case('errors', 24000000, edges, pos, {})

    # Without telemetry decoding the responses are skipped silently.
    sr, edges, pos = 24000000, [(0, 1)], 400
    pos = dshot_exchange(edges, pos, sr, 600, dshot_word(1000, True, True),
                         erpm12(1000)) + 3000
    pos = dshot_exchange(edges, pos, sr, 600, dshot_word(1000, True, True), None) + 3000
    pos = dshot_exchange(edges, pos, sr, 600, dshot_word(1000, True, True),
                         erpm12(1000)) + 3000
    dshot_case('tlm_off', sr, edges, pos, {'tlm': 'off'})


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
        # Vario and barometric altitude, in dm + 10000 or in m, with and
        # without the packed vertical speed, and too short.
        crsf(0xea, b'\x07' + be16(-56)),
        crsf(0xea, b'\x07' + be16(199)),
        crsf(0xea, [0x07, 5]),
        crsf(0xea, b'\x09' + be16(10000 + 1234) + bytes([0xce])),
        crsf(0xea, b'\x09' + be16(10000 - 55) + bytes([0x00])),
        crsf(0xea, b'\x09' + be16(10001) + bytes([0x80])),
        crsf(0xea, b'\x09' + be16(0x8000 | 3000) + bytes([0x7f])),
        crsf(0xea, b'\x09' + be16(10500) + be16(123)),
        crsf(0xea, [0x09, 0x27]),
        # Crossfire numbers 250 mW and 50 mW behind 2000 mW.
        crsf(0xc8, [0x14, 60, 62, 100, 8, 0, 27, 7, 55, 97, 5]),
        crsf(0xc8, [0x14, 60, 62, 100, 8, 0, 27, 8, 55, 97, 5]),
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


def crsf_case(name, frames, stop_errors=()):
    # 42 MHz gives exactly 100 samples per bit at 420000 baud, the CRSF
    # rate. crsf_c splits the data bits of a byte evenly, which matches the
    # uart decoder exactly only for a whole number of samples per bit.
    data = b''.join(frames)
    edges, pos = uart_edges(uart_levels(data, stop_errors))
    write_case('crsf_c/' + name, 42000000, edges, pos + 400, {
        'decoder': 'crsf_c',
        'channels': {},
        'options': {},
        'stack': [{'id': 'uart_c', 'channels': {'rx': 0},
                   'options': {'baudrate': 420000}}],
        'top_only': True,
    })


def generate_crsf_c():
    sticks = [172, 992, 1811, 0, 2047, 1000, 1500, 500] + [992] * 8
    bad_channels = bytearray(crsf(0xc8, b'\x16' + channels_payload(sticks)))
    bad_channels[-1] ^= 0x55
    bad_vario = bytearray(crsf(0xea, b'\x07' + be16(-300)))
    bad_vario[-1] ^= 0xa5
    frames = [
        # Noise in front of the first sync byte.
        bytes([0x00, 0x55]),
        crsf(0xc8, b'\x16' + channels_payload(sticks)),
        # ELRS appends the armed state to the channels.
        crsf(0xc8, b'\x16' + channels_payload([992] * 16, b'\x01')),
        crsf(0xc8, b'\x16' + bytes(10)),
        bytes(bad_channels),
        # Positive and negative SNR, TX power 100, 250, 50 mW and unknown.
        crsf(0xec, [0x14, 60, 62, 100, 8, 0, 27, 3, 55, 97, 5]),
        crsf(0xea, [0x14, 100, 0, 50, 0xf6, 1, 50, 7, 120, 10, 0x80]),
        crsf(0xea, [0x14, 90, 91, 70, 0x81, 0, 2, 8, 99, 50, 0x7f]),
        crsf(0xea, [0x14, 90, 91, 70, 12, 0, 2, 9, 99, 50, 0]),
        crsf(0xc8, [0x14, 1, 2, 3]),
        crsf(0xea, b'\x02' + be32(-33856000) + be32(1512150000) + be16(123) +
             be16(27000) + be16(1100) + bytes([12])),
        crsf(0xea, b'\x02' + be32(311234567) + be32(-1214567890) + be16(0) +
             be16(0) + be16(0) + bytes([0])),
        crsf(0xea, [0x02, 1, 2, 3]),
        crsf(0xea, b'\x07' + be16(-56)),
        crsf(0xea, b'\x07' + be16(199)),
        crsf(0xea, b'\x07' + be16(0)),
        crsf(0xea, [0x07, 5]),
        bytes(bad_vario),
        crsf(0xea, b'\x08' + be16(168) + be16(125) + (1300).to_bytes(3, 'big') + b'\x4b'),
        crsf(0xea, b'\x08' + be16(111) + be16(5) + (7).to_bytes(3, 'big')),
        crsf(0xea, [0x08, 1, 2, 3, 4, 5]),
        # Barometric altitude in dm + 10000, or in m with the top bit set,
        # with the packed vertical speed or without it (the 4 byte layout
        # of TBS carries none).
        crsf(0xea, b'\x09' + be16(10000 + 1234) + bytes([0xce])),
        crsf(0xea, b'\x09' + be16(10000 - 55) + bytes([0x00])),
        crsf(0xea, b'\x09' + be16(10000) + bytes([0x7f])),
        crsf(0xea, b'\x09' + be16(10001) + bytes([0x80])),
        crsf(0xea, b'\x09' + be16(10000) + bytes([0xff])),
        crsf(0xea, b'\x09' + be16(0x8000 | 3000) + bytes([0x01])),
        crsf(0xea, b'\x09' + be16(0)),
        crsf(0xea, b'\x09' + be16(10500) + be16(123)),
        crsf(0xea, [0x09, 0x27]),
        crsf(0xea, b'\x1e' + be16(-1234) + be16(5678) + be16(-31415)),
        crsf(0xea, [0x1e, 1, 2, 3]),
        crsf(0xea, b'\x21ACRO\x00'),
        crsf(0xea, b'\x21!FS!*\x00'),
        crsf(0xea, b'\x21Caf\xe9'),
        crsf(0xea, b'\x21\x00'),
        crsf(0xc8, [0x28, 0xec, 0xc8]),
        crsf(0xc8, [0x28]),
        crsf(0xea, device_info(b'ExpressLRS RX')),
        crsf(0xc8, b'\x0b' + be16(0xc8)),
        crsf(0xc8, [0x10] + [0] * 8),
        crsf(0xea, [0x2a, 0xee, 0xea]),
        crsf(0xea, [0x2b, 0xea, 0xee, 3, 0, 0, 9] + list(b'Packet Rate\x00')),
        crsf(0xee, [0x2c, 0xee, 0xea, 3, 0]),
        crsf(0xee, [0x2d, 0xee, 0xea, 3, 1]),
        crsf(0xc8, [0x32, 0xec, 0xc8, 0x10, 0x01]),
        crsf(0xea, [0x3a, 0xea, 0xee, 0x10]),
        # An unknown type ends the frame, its other bytes are no sync bytes.
        crsf(0xc8, [0x55, 1, 2, 3]),
        # Lengths which are no lengths.
        bytes([0xc8, 0x01]),
        bytes([0xee, 63]),
        crsf(0xee, [0x32, 0xee, 0xea, 0x10, 0x01]),
    ]
    crsf_case('frames', frames)
    # A framing error does not drop the byte, the frames still decode.
    bind = crsf(0xc8, [0x32, 0xec, 0xc8, 0x10, 0x01])
    crsf_case('uart_error', [bind, bind, bind], stop_errors=(3, 10))


if __name__ == '__main__':
    generate_dshot()
    generate_elrs()
    generate_dshot_c()
    generate_elrs_c()
    generate_crsf_c()
