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
   0x7d1b, or 0x7d14 with the inverted checksum of bidirectional DShot.
 - The GCR code of the telemetry response is checked against the decode
   table of Betaflight's dshot.c.

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

# 4b/5b GCR code of the telemetry response, in nibble order. It is kept
# separate from the decoder so that the tests do not silently inherit a
# mistake from it; test_dshot_gcr_table checks it against Betaflight.
GCR_SYMBOLS = (
    0x19, 0x1b, 0x12, 0x13, 0x1d, 0x15, 0x16, 0x17,
    0x1a, 0x09, 0x0a, 0x0b, 0x1e, 0x0d, 0x0e, 0x0f,
)


def dshot_crc4(value):
    value &= 0xfff
    return (value ^ (value >> 4) ^ (value >> 8)) & 0xf


def value_to_bits(value, count):
    return [(value >> (count - 1 - idx)) & 1 for idx in range(count)]


def dshot_word(value11, tlm=False, inverted=False):
    """Request word; bidirectional DShot inverts the checksum."""
    value12 = (value11 << 1) | int(tlm)
    crc = dshot_crc4(value12)
    if inverted:
        crc ^= 0xf
    return (value12 << 4) | crc


def dshot_request_edges(samplerate, rate, value16, start=0, inverted=False):
    """Every bit starts with the active level, which lasts 75 % of the
    period for a '1' and 37.5 % for a '0'. An inverted line rests high."""
    active = 0 if inverted else 1
    edges = []
    period = samplerate / rate
    position = start
    for bit in value_to_bits(value16, 16):
        edges.append((int(round(position)), active))
        edges.append((int(round(position + period * (0.75 if bit else 0.375))),
                      1 - active))
        position += period
    return edges, position


def tlm_word(value12, csum_xor=0):
    """Response word: the XOR of its four nibbles is 0xf."""
    csum = ~(value12 ^ (value12 >> 4) ^ (value12 >> 8)) & 0xf
    return (value12 << 4) | (csum ^ csum_xor)


def gcr_encode(value16):
    gcr = 0
    for idx in range(3, -1, -1):
        gcr = (gcr << 5) | GCR_SYMBOLS[(value16 >> (4 * idx)) & 0xf]
    return gcr


def dshot_telemetry_edges(samplerate, rate, value16, start, gcr=None, clock=1.0):
    """The ESC's response at 5/4 of the bit rate: a low start bit, then
    the 20 GCR bits as NRZI (a level change is a '1'), then the line is
    released high."""
    if gcr is None:
        gcr = gcr_encode(value16)
    tb = samplerate / rate * 0.8 * clock
    edges = [(int(round(start)), 0)]
    level = 0
    for idx in range(20):
        if (gcr >> (19 - idx)) & 1:
            level ^= 1
            edges.append((int(round(start + (idx + 1) * tb)), level))
    if level == 0:
        edges.append((int(round(start + 21 * tb)), 1))
    return edges, start + 21 * tb


def bidir_exchange(samplerate, rate, request, value16=None, start=100, **kwargs):
    """Inverted request, then the response 30 us later unless value16 is None."""
    edges, end = dshot_request_edges(samplerate, rate, request, start, inverted=True)
    if value16 is not None or 'gcr' in kwargs:
        reply, end = dshot_telemetry_edges(samplerate, rate, value16, end + 30e-6 * samplerate,
                                           **kwargs)
        edges += reply
    return edges, end


def run_dshot(module, capture, samplerate, options=None, idle=0):
    """The first wait() returns the idle level, end() runs at the end of
    the capture like srd_session_end() does."""
    decoder = build_decoder(module, options, samplerate)
    decoder.capture = [(0, idle)] + list(capture)
    try:
        decoder.decode()
    except EndOfCapture:
        pass
    decoder.end()
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
    assert dshot_word(1000, tlm=True) == 0x7d1b
    assert dshot_word(1000, tlm=True, inverted=True) == 0x7d14


def test_dshot_gcr_table(dshot):
    # Five bit code -> nibble table of Betaflight (dshot.c,
    # decode_bb_value()), None marks the invalid codes.
    iv = None
    betaflight = [iv, iv, iv, iv, iv, iv, iv, iv, iv, 9, 10, 11, iv, 13, 14, 15,
                  iv, iv, 2, 3, iv, 5, 6, 7, iv, 0, 8, 1, iv, 4, 12, iv]
    for nibble, symbol in enumerate(GCR_SYMBOLS):
        assert betaflight[symbol] == nibble, (nibble, hex(symbol))
    assert tuple(dshot.GCR_SYMBOLS) == GCR_SYMBOLS
    # Every nibble survives a round trip through the decoder.
    samplerate = 24e6
    for value12 in (0x444, 0x123, 0x567, 0x89a, 0xbcd, 0xef0, 0x004):
        edges, _ = bidir_exchange(samplerate, 600e3, dshot_word(1000, inverted=True),
                                  tlm_word(value12))
        decoder = run_dshot(dshot, edges, samplerate, idle=1)
        packets = python_packets(decoder, 'TLM')
        assert len(packets) == 1 and packets[0][2] == value12, \
            (hex(value12), packets, texts(decoder))


def test_dshot600_throttle(dshot):
    samplerate = 24e6
    edges, _ = dshot_request_edges(samplerate, 600e3, 0x7d1b, 100)
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
    edges, _ = dshot_request_edges(samplerate, 300e3, dshot_word(12), 100)
    decoder = run_dshot(dshot, edges, samplerate)
    assert has_text(decoder, 'Save settings'), texts(decoder)
    assert has_text(decoder, 'DShot300'), texts(decoder)
    assert python_packets(decoder, 'COMMAND') == [(12, 'Save settings')]
    assert not python_packets(decoder, 'THROTTLE')


def test_dshot1200_frame(dshot):
    # DShot1200 frames have 16 bits like the slower rates.
    samplerate = 48e6
    edges, _ = dshot_request_edges(samplerate, 1.2e6, 0x7d1b, 100)
    decoder = run_dshot(dshot, edges, samplerate)
    assert has_text(decoder, 'Checksum 0xb correct'), texts(decoder)
    assert has_text(decoder, 'DShot1200'), texts(decoder)
    assert has_text(decoder, 'Throttle 1000'), texts(decoder)


def test_dshot_bidirectional_telemetry(dshot):
    samplerate = 24e6
    request = dshot_word(1000, tlm=True, inverted=True)
    # Mantissa 100, exponent 0: a period of 100 us is 600000 eRPM.
    edges, _ = bidir_exchange(samplerate, 600e3, request, tlm_word(100))
    decoder = run_dshot(dshot, edges, samplerate, idle=1)
    assert has_text(decoder, 'M1 DShot600 bidir 0x7d14'), texts(decoder)
    assert has_text(decoder, 'Checksum 0x4 correct'), texts(decoder)
    assert python_packets(decoder, 'FRAME') == [('1', 1000, True, True, 'DShot600')]
    assert python_packets(decoder, 'TLM') == [('eRPM', 600000, 100)], \
        python_packets(decoder, 'TLM')
    assert has_text(decoder, 'eRPM 600000 (period 100 us'), texts(decoder)
    # 0xfff reports a motor which stands still.
    edges, _ = bidir_exchange(samplerate, 600e3, request, tlm_word(0xfff))
    decoder = run_dshot(dshot, edges, samplerate, idle=1)
    assert python_packets(decoder, 'TLM') == [('eRPM', 0, 0xfff)], texts(decoder)


def test_dshot_extended_telemetry(dshot):
    samplerate = 24e6
    request = dshot_word(48, inverted=True)
    for value12, packet, text in (
            (0x22d, ('Temperature', 45, 0x22d), 'Temperature: 45 degC'),
            (0x432, ('Voltage', 50, 0x432), 'Voltage: 12.50 V'),
            (0x614, ('Current', 20, 0x614), 'Current: 20 A'),
            (0xe03, ('State', 3, 0xe03), 'State: 0x03')):
        edges, _ = bidir_exchange(samplerate, 600e3, request, tlm_word(value12))
        decoder = run_dshot(dshot, edges, samplerate, idle=1)
        assert python_packets(decoder, 'TLM') == [packet], python_packets(decoder, 'TLM')
        assert has_text(decoder, text), texts(decoder)


def test_dshot_telemetry_errors(dshot):
    samplerate = 24e6
    request = dshot_word(1000, inverted=True)
    edges, _ = bidir_exchange(samplerate, 600e3, request, tlm_word(100, csum_xor=5))
    decoder = run_dshot(dshot, edges, samplerate, idle=1)
    assert has_text(decoder, 'Telemetry checksum mismatch'), texts(decoder)
    assert has_text(decoder, '(checksum error)'), texts(decoder)
    # 0b00000 is not one of the 16 valid five bit codes.
    gcr = gcr_encode(tlm_word(100)) & ~(0x1f << 15)
    edges, _ = bidir_exchange(samplerate, 600e3, request, gcr=gcr)
    decoder = run_dshot(dshot, edges, samplerate, idle=1)
    assert has_text(decoder, 'Invalid GCR symbol 0x00'), texts(decoder)
    assert not python_packets(decoder, 'TLM')
    # A silent ESC is reported once the next frame arrives.
    edges, end = bidir_exchange(samplerate, 600e3, request, None)
    more, _ = bidir_exchange(samplerate, 600e3, request, tlm_word(100), start=end + 3000)
    decoder = run_dshot(dshot, edges + more, samplerate, idle=1)
    assert has_text(decoder, 'No telemetry response'), texts(decoder)
    assert len(python_packets(decoder, 'TLM')) == 1


def test_dshot_esc_clock_error(dshot):
    # The response bit time is measured, an ESC clock 6 % off is fine.
    samplerate = 24e6
    for clock in (1.06, 0.94):
        edges, _ = bidir_exchange(samplerate, 300e3, dshot_word(1000, inverted=True),
                                  tlm_word(0x5f4), clock=clock)
        decoder = run_dshot(dshot, edges, samplerate, idle=1)
        assert python_packets(decoder, 'TLM') == [('eRPM', 30000, 0x5f4)], \
            (clock, texts(decoder))


def test_dshot_bad_checksum(dshot):
    samplerate = 24e6
    # Flip the throttle bit, which invalidates the checksum.
    edges, _ = dshot_request_edges(samplerate, 600e3, 0x7d1b ^ 0x0400, 100)
    decoder = run_dshot(dshot, edges, samplerate)
    assert has_text(decoder, 'Checksum mismatch'), texts(decoder)
    assert python_packets(decoder, 'FRAME')[0][3] is False


def test_dshot_unexpected_burst(dshot):
    samplerate = 24e6
    # Ten pulses are no request frame.
    edges, _ = dshot_request_edges(samplerate, 600e3, 0xffff, 100)
    decoder = run_dshot(dshot, edges[:20], samplerate)
    assert has_text(decoder, 'Unexpected burst of 10 pulses'), texts(decoder)


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
        ('dshot1200_frame', lambda: test_dshot1200_frame(dshot)),
        ('dshot_bidirectional_telemetry',
         lambda: test_dshot_bidirectional_telemetry(dshot)),
        ('dshot_extended_telemetry', lambda: test_dshot_extended_telemetry(dshot)),
        ('dshot_telemetry_errors', lambda: test_dshot_telemetry_errors(dshot)),
        ('dshot_esc_clock_error', lambda: test_dshot_esc_clock_error(dshot)),
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
