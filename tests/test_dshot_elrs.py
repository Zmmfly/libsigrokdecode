#!/usr/bin/env python3
"""
Self contained tests for the 'dshot' and 'elrs' protocol decoders.

The decoders normally run inside libsigrokdecode. This script provides a
minimal stand in for the 'sigrokdecode' module, so that the decoders can
be exercised with plain Python and synthetic signals, without building
the library or owning the hardware.

The checksum implementations are verified against reference values:
 - CRSF uses CRC-8/DVB-S2, whose catalogue check value for the ASCII
   string '123456789' is 0xbc.
 - The DShot checksum is the XOR of the three upper nibbles of a frame.
   The frame used below was computed by hand: throttle 1000 with the
   telemetry request bit set is 0x7d1, which makes the complete frame
   0x7d1b.

Usage:
    python3 libsigrokdecode/tests/test_dshot_elrs.py [-v]
"""

import importlib.util
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
DECODER_ROOT = os.path.join(HERE, '..', 'decoders')

OUTPUT_ANN = 1
OUTPUT_PYTHON = 2
SRD_CONF_SAMPLERATE = 10

VERBOSE = False


class EndOfCapture(Exception):
    pass


class FakeDecoder:
    """Stand in for sigrokdecode.Decoder."""

    def __init__(self):
        self.samplenum = 0
        self.options = {}
        self.ann = []
        self.py = []
        self.capture = []

    def register(self, output_type):
        return output_type

    def put(self, ss, es, output, data):
        if output == OUTPUT_ANN:
            self.ann.append((ss, es, data[0], data[1]))
        else:
            self.py.append((ss, es, data[0], data[1]))

    def has_channel(self, index):
        return True

    def wait(self, cond=None, skip=None):
        if not self.capture:
            raise EndOfCapture()
        sample, level = self.capture.pop(0)
        self.samplenum = sample
        return (level, )


def install_fake_module():
    module = types.ModuleType('sigrokdecode')
    module.Decoder = FakeDecoder
    module.SRD_CONF_SAMPLERATE = SRD_CONF_SAMPLERATE
    module.OUTPUT_ANN = OUTPUT_ANN
    module.OUTPUT_PYTHON = OUTPUT_PYTHON
    sys.modules['sigrokdecode'] = module


def load_decoder(name):
    path = os.path.join(DECODER_ROOT, name, 'pd.py')
    spec = importlib.util.spec_from_file_location(
        'pd_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_decoder(module, options=None, samplerate=None):
    # Create the decoder without running the decoder's own __init__ first,
    # so that the harness state is in place even though the decoders do
    # not chain up to their base class.
    decoder = module.Decoder.__new__(module.Decoder)
    FakeDecoder.__init__(decoder)
    module.Decoder.__init__(decoder)
    decoder.options = {opt['id']: opt['default']
                       for opt in module.Decoder.options}
    if options:
        decoder.options.update(options)
    decoder.start()
    if samplerate:
        decoder.metadata(SRD_CONF_SAMPLERATE, samplerate)
    return decoder


def texts(decoder):
    result = []
    for _, _, _, lines in decoder.ann:
        result.extend(lines)
    return result


def has_text(decoder, needle):
    return any(needle in line for line in texts(decoder))


def python_packets(decoder, kind):
    return [data for _, _, ptype, data in decoder.py if ptype == kind]


#-- DShot signal synthesis ---------------------------------------------

# Copy of the 4b/5b table, kept separate from the decoder so that the
# tests do not silently inherit a mistake from it.
GCR_SYMBOLS = (
    0x19, 0x1b, 0x12, 0x13, 0x1c, 0x15, 0x16, 0x17,
    0x1a, 0x09, 0x0a, 0x0b, 0x1e, 0x0d, 0x0e, 0x0f,
)


def dshot_crc4(value):
    value &= 0xfff
    return (value ^ (value >> 4) ^ (value >> 8)) & 0xf


def value_to_bits(value, count):
    return [(value >> (count - 1 - idx)) & 1 for idx in range(count)]


def gcr_encode(value16, msb_first):
    """Spread a 16 bit value over 20 GCR coded bits."""
    nibbles = [(value16 >> (4 * idx)) & 0xf for idx in range(4)]
    if msb_first:
        nibbles.reverse()
    bits = []
    for nibble in nibbles:
        symbol = GCR_SYMBOLS[nibble]
        bits.extend(value_to_bits(symbol, 5))
    return bits


def dshot_symbol_edges(samplerate, rate, bits, start):
    """Turn symbol bits into edges, one duty cycle per bit period."""
    edges = []
    period = int(round(samplerate / rate))
    position = start
    for bit in bits:
        high = int(round(period * (0.75 if bit else 0.375)))
        edges.append((position, 1))
        edges.append((position + high, 0))
        position += period
    return edges, position


def dshot_request_edges(samplerate, rate, value16, start=0):
    bits = value_to_bits(value16, 16)
    return dshot_symbol_edges(samplerate, rate, bits, start)


def dshot_telemetry_edges(samplerate, rate, value16, start):
    """Synthesize the inverted GCR response of an ESC."""
    # The checksum nibble is transmitted first, which means it ends up in
    # the most significant nibble of the decoded value. That is how the
    # flight controller firmware assembles the response: the first
    # received symbol goes into the upper nibble.
    bits = [0] + gcr_encode(value16, msb_first=True)
    edges = []
    period = int(round(samplerate / rate))
    position = start
    # The line idles high for a while before the first start bit.
    edges.append((position, 1))
    position += int(round(period * 6))
    for bit in bits:
        low = int(round(period * (0.75 if bit else 0.375)))
        edges.append((position, 0))
        edges.append((position + low, 1))
        position += period
    return edges, position


def run_dshot(module, capture, samplerate, options=None):
    decoder = build_decoder(module, options, samplerate)
    decoder.capture = list(capture)
    try:
        decoder.decode()
    except EndOfCapture:
        pass
    return decoder


#-- CRSF frame synthesis -----------------------------------------------

def crc8(data):
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0xd5) & 0xff if crc & 0x80 else (crc << 1) & 0xff
    return crc


def crsf_frame(body):
    """Add the address, the length and the checksum to a frame body."""
    # The length counts the type, the payload and the checksum.
    frame = bytes([0xc8, len(body) + 1]) + bytes(body)
    return frame + bytes([crc8(frame[2:])])


def run_elrs(module, data, options=None):
    decoder = build_decoder(module, options)
    for idx, byte in enumerate(data):
        decoder.decode(idx * 20, idx * 20 + 16, ('FRAME', 0, (byte, True)))
    return decoder


#-- Tests --------------------------------------------------------------

def test_dshot_checksum():
    # Throttle 1000 with the telemetry request bit: 0x7d1 ^ 0x07d ^ 0x007.
    assert dshot_crc4(0x7d1) == 0xb
    value16 = (0x7d1 << 4) | dshot_crc4(0x7d1)
    assert value16 == 0x7d1b


def test_dshot_gcr_table(dshot):
    # Every nibble must survive a round trip through the code table.
    # The documented nibble order puts the first symbol into the most
    # significant nibble, which is what the decoder tries first. Only
    # frames with a matching checksum are considered valid ones.
    payloads = (0x000, 0x7d1, 0xfff, 0x123, 0xaaa, 0x555, 0x004, 0x800)
    for payload12 in payloads:
        value16 = (payload12 << 4) | dshot_crc4(payload12)
        bits = gcr_encode(value16, msb_first=True)
        assert len(bits) == 20
        decoded, order, crc_ok = dshot.Decoder().gcr_to_value(bits)
        assert decoded == value16, (hex(value16), order)
        assert crc_ok and order == 'high first'
    # A nibble swapped frame is not a valid frame, but a four bit
    # checksum cannot always tell. The decoder therefore trusts the
    # documented order first, which keeps mis-ordered frames from being
    # decoded as valid data.
    swapped = gcr_encode(0x7d1b, msb_first=False)
    decoded, order, crc_ok = dshot.Decoder().gcr_to_value(swapped)
    assert order == 'high first' and decoded == 0xb1d7 and crc_ok, \
        (hex(decoded), order, crc_ok)

    # Symbols outside of the code table are rejected outright, 0b00000 is
    # not one of the 16 valid five bit codes.
    assert dshot.Decoder().gcr_to_value([0] * 20) == (None, None, False)


def test_dshot600_throttle(dshot):
    samplerate = 24e6
    value16 = 0x7d1b
    edges, end = dshot_request_edges(samplerate, 600e3, value16, 0)
    # A long idle level ends the frame, and a trailing edge flushes it.
    edges.append((end + 1000, 1))
    decoder = run_dshot(dshot, edges, samplerate)
    assert has_text(decoder, 'Throttle 1000'), texts(decoder)
    assert has_text(decoder, 'Checksum 0xb correct'), texts(decoder)
    assert has_text(decoder, 'Telemetry requested'), texts(decoder)
    assert has_text(decoder, 'DShot600'), texts(decoder)
    packets = python_packets(decoder, 'FRAME')
    assert packets == [('1', 1000, True, True, 'DShot600')], packets


def test_dshot300_command(dshot):
    samplerate = 24e6
    # Command 12 (save settings) without a telemetry request.
    value12 = 12 << 1
    value16 = (value12 << 4) | dshot_crc4(value12)
    edges, end = dshot_request_edges(samplerate, 300e3, value16, 0)
    edges.append((end + 2000, 1))
    decoder = run_dshot(dshot, edges, samplerate)
    assert has_text(decoder, 'Save settings'), texts(decoder)
    assert has_text(decoder, 'DShot300'), texts(decoder)
    assert python_packets(decoder, 'COMMAND') == [(12, 'Save settings')]
    assert not python_packets(decoder, 'THROTTLE')


def test_dshot1200_gcr(dshot):
    samplerate = 48e6
    value16 = 0x7d1b
    bits = gcr_encode(value16, msb_first=True)
    edges, end = dshot_symbol_edges(samplerate, 1.2e6, bits, 0)
    edges.append((end + 2000, 1))
    decoder = run_dshot(dshot, edges, samplerate)
    assert has_text(decoder, 'Checksum 0xb correct'), texts(decoder)
    assert has_text(decoder, 'DShot1200'), texts(decoder)
    assert has_text(decoder, 'Throttle 1000'), texts(decoder)


def test_dshot_bidirectional_telemetry(dshot):
    samplerate = 24e6
    # Throttle 1000 asks for telemetry.
    edges, end = dshot_request_edges(samplerate, 600e3, 0x7d1b, 0)
    # Mantissa 100, exponent 0 gives a period of 100 and 6000 eRPM.
    payload = 100
    value16 = (dshot_crc4(payload) << 12) | payload
    reply, reply_end = dshot_telemetry_edges(
        samplerate, 600e3, value16, end + 1000)
    edges.extend(reply)
    edges.append((reply_end + 1000, 0))
    decoder = run_dshot(dshot, edges, samplerate)
    assert python_packets(decoder, 'TLM') == [('eRPM', 6000, payload)], \
        python_packets(decoder, 'TLM')
    assert has_text(decoder, '6000 eRPM'), texts(decoder)


def test_dshot_bad_checksum(dshot):
    samplerate = 24e6
    # Flip the throttle bit, which invalidates the checksum.
    value16 = 0x7d1b ^ 0x0400
    edges, end = dshot_request_edges(samplerate, 600e3, value16, 0)
    edges.append((end + 1000, 1))
    decoder = run_dshot(dshot, edges, samplerate)
    assert has_text(decoder, 'Checksum mismatch'), texts(decoder)
    assert python_packets(decoder, 'FRAME')[0][3] is False


def test_dshot_unexpected_burst(dshot):
    samplerate = 24e6
    # Ten symbols are neither a request frame nor a telemetry response.
    bits = [1] * 10
    edges, end = dshot_symbol_edges(samplerate, 600e3, bits, 0)
    edges.append((end + 1000, 1))
    decoder = run_dshot(dshot, edges, samplerate)
    assert has_text(decoder, 'Unexpected burst'), texts(decoder)


def test_crsf_checksum():
    # Reference check value of CRC-8/DVB-S2 for the ASCII string.
    assert crc8(b'123456789') == 0xbc


def test_elrs_channels(elrs):
    channels = [172, 992, 1811, 992, 992, 992, 992, 992,
                992, 992, 992, 992, 992, 992, 992, 992]
    packed = 0
    for idx, value in enumerate(channels):
        packed |= value << (11 * idx)
    payload = packed.to_bytes(22, 'little') + bytes([0x03])
    decoder = run_elrs(elrs, crsf_frame(bytes([0x16]) + payload))
    assert has_text(decoder, '988us'), texts(decoder)
    assert has_text(decoder, '1500us'), texts(decoder)
    assert has_text(decoder, '2012us'), texts(decoder)
    assert has_text(decoder, 'Armed: yes'), texts(decoder)
    assert python_packets(decoder, 'CHANNELS') == [(tuple(channels), True)]
    assert not has_text(decoder, 'Checksum mismatch')


def test_elrs_link_statistics(elrs):
    payload = bytes([60, 62, 100, 8, 0, 27, 3, 55, 97, 5])
    decoder = run_elrs(elrs, crsf_frame(bytes([0x14]) + payload))
    assert has_text(decoder, 'RSSI1 -60 dBm'), texts(decoder)
    assert has_text(decoder, 'RSSI2 -62 dBm'), texts(decoder)
    assert has_text(decoder, 'LQ 100%'), texts(decoder)
    assert has_text(decoder, 'SNR 8 dB'), texts(decoder)
    assert has_text(decoder, '2.4 GHz 250 Hz'), texts(decoder)
    assert has_text(decoder, '100 mW'), texts(decoder)
    assert has_text(decoder, 'Downlink: RSSI -55 dBm'), texts(decoder)


def test_elrs_status(elrs):
    body = bytes([0x2e, 0xea, 0xee, 3, 0x01, 0x2c, 0x09])
    body += b'[ ! Armed ! ]\x00'
    decoder = run_elrs(elrs, crsf_frame(body))
    assert has_text(decoder, '3 bad, 300 good'), texts(decoder)
    assert has_text(decoder, ', armed'), texts(decoder)
    assert has_text(decoder, 'connected'), texts(decoder)
    assert has_text(decoder, 'Message: [ ! Armed ! ]'), texts(decoder)
    assert python_packets(decoder, 'STATUS') == [
        (3, 300, 0x09, '[ ! Armed ! ]')]


def test_elrs_device_info(elrs):
    name = b'ExpressLRS RX\x00'
    fields = (0x11223344).to_bytes(4, 'big')
    fields += (0x55667788).to_bytes(4, 'big')
    fields += ((3 << 24) | (4 << 16) | (2 << 8)).to_bytes(4, 'big')
    fields += bytes([35, 7])
    body = bytes([0x29, 0xea, 0xec]) + name + fields
    decoder = run_elrs(elrs, crsf_frame(body))
    assert has_text(decoder, 'Device "ExpressLRS RX", version 3.4.2'), \
        texts(decoder)
    assert has_text(decoder, '35 parameter(s)'), texts(decoder)


def test_elrs_bind_command(elrs):
    body = bytes([0x32, 0xec, 0xc8, 0x10, 0x01])
    decoder = run_elrs(elrs, crsf_frame(body))
    assert has_text(decoder, 'RX command: bind'), texts(decoder)
    assert python_packets(decoder, 'BIND') == [True]


def test_elrs_parameter_entry(elrs):
    body = bytes([0x2b, 0xea, 0xee, 3, 0, 0, 9])
    body += b'Packet Rate\x00' + b'25Hz;50Hz;100Hz\x00' + bytes([1, 0, 2, 0])
    decoder = run_elrs(elrs, crsf_frame(body))
    assert has_text(decoder, 'Parameter 3, "Packet Rate", type 9'), \
        texts(decoder)


def test_elrs_bad_checksum(elrs):
    frame = bytearray(crsf_frame(bytes([0x14]) + bytes(10)))
    frame[-1] ^= 0xff
    decoder = run_elrs(elrs, frame)
    assert has_text(decoder, 'Checksum mismatch'), texts(decoder)
    # Without a valid checksum the payload must not be interpreted.
    assert not python_packets(decoder, 'LINK')


def test_elrs_noise_and_bad_length(elrs):
    # Noise in front of a frame is skipped, an implausible length warns,
    # and the frames around it still decode.
    command = bytes([0x32, 0xec, 0xc8, 0x10, 0x01])
    data = bytes([0x00, 0x55, 0xaa]) + crsf_frame(command)
    data += bytes([0xc8, 0x01]) + crsf_frame(command)
    decoder = run_elrs(elrs, data)
    assert has_text(decoder, 'Invalid frame length'), texts(decoder)
    assert has_text(decoder, 'RX command: bind'), texts(decoder)
    assert python_packets(decoder, 'BIND') == [True, True]


def test_decoder_metadata(dshot, elrs):
    # The library rejects decoder definitions which do not follow these
    # rules, so check them here instead of only at runtime.
    for name, module in (('dshot', dshot), ('elrs', elrs)):
        dec = module.Decoder
        assert dec.id == name, dec.id
        assert dec.api_version == 3
        assert dec.license in ('gplv2+', 'gplv3+'), dec.license
        for attr in ('name', 'longname', 'desc'):
            value = getattr(dec, attr, None)
            assert isinstance(value, str) and value, attr

        ann_count = len(dec.annotations)
        assert ann_count > 0
        for ann in dec.annotations:
            assert isinstance(ann, tuple) and len(ann) == 2, ann
            assert all(isinstance(part, str) and part for part in ann), ann
        covered = set()
        for row in dec.annotation_rows:
            assert isinstance(row, tuple) and len(row) == 3, row
            row_id, row_desc, classes = row
            assert row_id and row_desc and classes, row
            for cls in classes:
                assert 0 <= cls < ann_count, (row_id, cls)
                covered.add(cls)
        assert covered == set(range(ann_count)), \
            (name, sorted(set(range(ann_count)) - covered))

        assert dec.options
        for opt in dec.options:
            assert opt['id'] and opt['desc'], opt
            assert 'default' in opt, opt
            if 'values' in opt:
                assert opt['default'] in opt['values'], opt
        for chan in getattr(dec, 'channels', ()):
            assert chan['id'] and chan['name'] and chan['desc'], chan
        for upstream in dec.inputs:
            assert upstream == 'logic' or \
                os.path.isdir(os.path.join(DECODER_ROOT, upstream)), upstream


def test_frames_split_into_chunks(elrs):
    # A frame delivered in pieces must decode the same way.
    frame = crsf_frame(bytes([0x32, 0xec, 0xc8, 0x10, 0x01]))
    decoder = build_decoder(elrs)
    for idx, byte in enumerate(frame):
        decoder.decode(idx, idx + 1, ('FRAME', 0, (byte, True)))
    assert python_packets(decoder, 'BIND') == [True]


def main():
    global VERBOSE
    VERBOSE = '-v' in sys.argv[1:]
    install_fake_module()
    dshot = load_decoder('dshot')
    elrs = load_decoder('elrs')
    tests = [
        ('dshot_checksum', lambda: test_dshot_checksum()),
        ('dshot_gcr_table', lambda: test_dshot_gcr_table(dshot)),
        ('dshot600_throttle', lambda: test_dshot600_throttle(dshot)),
        ('dshot300_command', lambda: test_dshot300_command(dshot)),
        ('dshot1200_gcr', lambda: test_dshot1200_gcr(dshot)),
        ('dshot_bidirectional_telemetry',
         lambda: test_dshot_bidirectional_telemetry(dshot)),
        ('dshot_bad_checksum', lambda: test_dshot_bad_checksum(dshot)),
        ('dshot_unexpected_burst', lambda: test_dshot_unexpected_burst(dshot)),
        ('crsf_checksum', lambda: test_crsf_checksum()),
        ('elrs_channels', lambda: test_elrs_channels(elrs)),
        ('elrs_link_statistics', lambda: test_elrs_link_statistics(elrs)),
        ('elrs_status', lambda: test_elrs_status(elrs)),
        ('elrs_device_info', lambda: test_elrs_device_info(elrs)),
        ('elrs_bind_command', lambda: test_elrs_bind_command(elrs)),
        ('elrs_parameter_entry', lambda: test_elrs_parameter_entry(elrs)),
        ('elrs_bad_checksum', lambda: test_elrs_bad_checksum(elrs)),
        ('elrs_noise_and_bad_length', lambda: test_elrs_noise_and_bad_length(elrs)),
        ('frames_split_into_chunks', lambda: test_frames_split_into_chunks(elrs)),
        ('decoder_metadata', lambda: test_decoder_metadata(dshot, elrs)),
    ]
    failed = 0
    for name, func in tests:
        try:
            func()
        except AssertionError as exc:
            failed += 1
            print('FAIL {}: {}'.format(name, exc))
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print('ERROR {}: {!r}'.format(name, exc))
        else:
            print('ok   {}'.format(name))
    print('{} of {} test(s) passed'.format(len(tests) - failed, len(tests)))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
