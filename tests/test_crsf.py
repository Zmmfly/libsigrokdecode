#!/usr/bin/env python3
"""
Self contained tests for the 'crsf' protocol decoder, with the stand in
for the 'sigrokdecode' module from test_dshot_elrs.py.

The uart decoder hands every byte to the decoder as ('DATA', rxtx,
(value, bits)), with a [bit, ss, es] entry per data bit. Like the uart
decoder, run_crsf() refills one list with the bits of every byte, so a
decoder which keeps the list instead of a copy fails here.

Usage:
    python3 libsigrokdecode/tests/test_crsf.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_dshot_elrs as harness  # noqa: E402

BIT = 10  # samples per UART bit


def build(module):
    decoder = module.Decoder.__new__(module.Decoder)
    harness.FakeDecoder.__init__(decoder)
    module.Decoder.__init__(decoder)
    decoder.start()
    return decoder


def data_span(idx):
    """Span of the data bits of byte `idx`, behind its start bit."""
    ss = idx * 10 * BIT + BIT
    return ss, ss + 8 * BIT


def run_crsf(module, data):
    decoder = build(module)
    bits = []
    for idx, byte in enumerate(data):
        ss, es = data_span(idx)
        bits.clear()
        for bit in range(8):
            bits.append([(byte >> bit) & 1, ss + bit * BIT, ss + (bit + 1) * BIT])
        decoder.decode(ss, es, ('DATA', 0, (byte, bits)))
    return decoder


def frame(address, body):
    data = bytes([address, len(body) + 1]) + bytes(body)
    return data + bytes([harness.crc8(data[2:])])


def anns(decoder):
    return [(ss, es, cls, texts[0]) for ss, es, cls, texts in decoder.ann]


def be16(value):
    return (value & 0xffff).to_bytes(2, 'big')


def test_channels(crsf):
    values = [172, 992, 1811, 0, 2047, 1000, 1500, 500] + [992] * 8
    packed = sum(value << (11 * idx) for idx, value in enumerate(values))
    decoder = run_crsf(crsf, frame(0xc8, b'\x16' + packed.to_bytes(22, 'little')))
    chans = [a for a in anns(decoder) if a[3].startswith('Chan:')]
    assert [a[3] for a in chans] == ['Chan:{} Value:{}'.format(idx, value)
                                     for idx, value in enumerate(values)], chans
    # Channel 0 spans bits 0..10: all of payload byte 0 (the fourth byte)
    # and three bits of the next one, channel 15 the last 11 bits.
    assert chans[0][:2] == (data_span(3)[0], data_span(4)[0] + 3 * BIT), chans[0]
    assert chans[15][:2] == (data_span(23)[0] + 5 * BIT, data_span(24)[1]), chans[15]
    assert anns(decoder)[-1] == (*data_span(25), 0, 'Checksum crc8 poly 0xD5.')


def test_link_statistics(crsf):
    payload = [100, 0, 50, 0xf6, 1, 50, 7, 120, 10, 0x80]
    decoder = run_crsf(crsf, frame(0xea, [0x14] + payload))
    fields = anns(decoder)[3:13]
    assert [a[3] for a in fields] == [
        'Uplink RSSI 1: -100dB', 'Uplink RSSI 2: -0dB', 'Uplink Link Quality: 50',
        'Uplink SNR: -10dB', 'Active Antenna: 1', 'RF Mode: 50',
        'Uplink TX Power: 250 mW', 'Downlink RSSI: -120dB',
        'Downlink Link Quality: 10', 'Downlink SNR: -128dB'], fields
    # Every field spans its own byte.
    assert [a[:2] for a in fields] == [data_span(idx) for idx in range(3, 13)], fields
    for power, text in ((8, '50 mW'), (9, 'level 9')):
        payload[6] = power
        decoder = run_crsf(crsf, frame(0xea, [0x14] + payload))
        assert 'Uplink TX Power: ' + text in [a[3] for a in anns(decoder)], anns(decoder)


def test_telemetry(crsf):
    cases = (
        (b'\x07' + be16(-56), 'Vario: -0.56 m/s'),
        (b'\x09' + be16(10000 + 1234) + b'\xce',
         'Baro altitude: 123.4 m, vertical speed -2.67 m/s'),
        (b'\x09' + be16(0x8000 | 3000), 'Baro altitude: 3000 m'),
        (b'\x08' + be16(168) + be16(125) + (1300).to_bytes(3, 'big') + b'\x4b',
         'Battery: 16.8 V, 12.5 A, 1300 mAh, 75%'),
        (b'\x1e' + be16(-1234) + be16(5678) + be16(-31415),
         'Attitude: pitch -0.1234 rad, roll 0.5678 rad, yaw -3.1415 rad'),
        (b'\x21!FS!*\x00', 'Flight mode: !FS!*'),
        (b'\x0b' + be16(0xc8), 'Heartbeat (2 bytes payload)'),
        (b'\x07\x05', 'Vario (1 bytes payload)'),
    )
    for body, text in cases:
        decoder = run_crsf(crsf, frame(0xea, body))
        # One line over the payload, between the type and the CRC byte.
        line = (data_span(3)[0], data_span(len(body) + 1)[1], 0, text)
        assert line in anns(decoder), (text, anns(decoder))


def test_short_frames_and_errors(crsf):
    bad = bytearray(frame(0xc8, [0x14, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
    bad[-1] ^= 0xff
    data = frame(0xc8, b'\x16' + bytes(10)) + frame(0xc8, [0x14, 1, 2, 3]) + bytes(bad)
    decoder = run_crsf(crsf, data)
    texts = [a[3] for a in anns(decoder)]
    assert 'CRSF_FRAMETYPE_RC_CHANNELS_PACKED (10 bytes payload)' in texts, texts
    assert 'LINK_STATISTICS (3 bytes payload)' in texts, texts
    errors = [a for a in decoder.ann if a[2] == 1]
    assert len(errors) == 1 and errors[0][3][1] == 'Checksum error', errors


def test_resync(crsf):
    # Noise, a length which is no length and an unknown type do not hide
    # the frame behind them.
    command = frame(0xee, [0x32, 0xee, 0xea, 0x10, 0x01])
    decoder = run_crsf(crsf, bytes([0x55, 0xc8, 0x01, 0xc8, 0x03, 0x99]) + command)
    texts = [a[3] for a in anns(decoder)]
    assert texts[:6] == ['Unknown packet type85', 'To Flight Controller',
                         'Unknown packet type1', 'To Flight Controller',
                         'Num of bytes succeeding: 1', 'Unknown packet type153'], texts
    assert texts[-3:] == ['CRSF_FRAMETYPE_COMMAND', 'CRSF_FRAMETYPE_COMMAND (4 bytes payload)',
                          'Checksum crc8 poly 0xD5.'], texts


def main():
    harness.install_fake_module()
    crsf = harness.load_decoder('crsf')
    tests = [
        ('channels', lambda: test_channels(crsf)),
        ('link_statistics', lambda: test_link_statistics(crsf)),
        ('telemetry', lambda: test_telemetry(crsf)),
        ('short_frames_and_errors', lambda: test_short_frames_and_errors(crsf)),
        ('resync', lambda: test_resync(crsf)),
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
