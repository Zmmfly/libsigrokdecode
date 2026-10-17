##
## This file is part of the libsigrokdecode project.
##
## Copyright (C) 2026 Zmmfly <shenqhy@qq.com>
##
## This program is free software; you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 2 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program; if not, see <http://www.gnu.org/licenses/>.
##

'''
DShot is the digital command protocol which flight controllers use to
talk to BLHeli_S, BLHeli_32 and Bluejay ESCs. A single wire per motor
carries 16 bit frames at 150, 300 or 600 kbit/s. DShot1200 transmits 20
symbols instead, because every 4 bit nibble is spread over 5 bits of
4b/5b GCR code.

Frame layout of the 16 bit variants, transmitted MSB first:
- Bits 15..5: 11 bit throttle value or command number.
- Bit 4:      request a telemetry response from the ESC.
- Bits 3..0:  4 bit checksum, the XOR of the upper three nibbles.

Values 1..47 are commands (beeps, spin direction, 3D mode, save
settings, LEDs, ...), values 48..2047 are throttle. When 3D mode is
enabled, 48..1047 selects the negative and 1048..2047 the positive
direction. A bit period carries the value in its duty cycle: 75% high
level is a '1', 37.5% high level is a '0'.

With bidirectional DShot the ESC answers on the same wire. The response
is inverted, GCR encoded, consists of a start bit and 20 payload bits
which hold a 12 bit value (3 bit exponent, 9 bit mantissa) and a 4 bit
checksum. Usually the commutation period is reported, from which the
motor's eRPM is derived. BLHeli_32 and Bluejay can also report
temperature, voltage, current and debug values instead.

Neither the symbol rate nor the exact gap between the request and the
response is trusted from a configuration. Both are measured from the
captured signal, and the checksums decide how a burst is interpreted.
A burst ends when the line stays idle for longer than five microseconds,
so the last frame of a capture is only annotated once another edge
follows it.

OUTPUT_PYTHON format:

Packet:
(<ptype>, <pdata>)

This is the list of <ptype> codes and their respective <pdata> values:
 - 'FRAME': (<motor>, <value11>, <tlm_req>, <crc_ok>, <rate>)
 - 'COMMAND': (<code>, <name>)
 - 'THROTTLE': (<value11>, <percent>)
 - 'TLM': (<kind>, <value>, <raw12>)
'''

import sigrokdecode as srd


class SamplerateError(Exception):
    pass


class ChannelError(Exception):
    pass


# Bit period in microseconds of the four DShot variants.
RATE_PERIODS = (
    ('DShot150', 1e6 / 150e3),
    ('DShot300', 1e6 / 300e3),
    ('DShot600', 1e6 / 600e3),
    ('DShot1200', 1e6 / 1200e3),
)

# A gap longer than this separates two request frames, and also the
# request from the ESC's telemetry response. Within a frame the longest
# low level period is 0.625 bit periods, which is less than four
# microseconds even for the slowest variant.
IDLE_USEC = 5.0

# 4b/5b GCR code. DShot1200 spreads each nibble of the frame over five
# bits, and the ESC's bidirectional telemetry response uses the same
# code. The values are listed in nibble order.
GCR_SYMBOLS = (
    0x19, 0x1b, 0x12, 0x13, 0x1c, 0x15, 0x16, 0x17,
    0x1a, 0x09, 0x0a, 0x0b, 0x1e, 0x0d, 0x0e, 0x0f,
)
GCR_NIBBLES = {sym: idx for idx, sym in enumerate(GCR_SYMBOLS)}

# Command numbers as defined by BLHeli. Values above DSHOT_CMD_MAX are
# throttle values instead.
COMMANDS = {
    1: 'Beep 1',
    2: 'Beep 2',
    3: 'Beep 3',
    4: 'Beep 4',
    5: 'Beep 5',
    6: 'ESC info',
    7: 'Spin direction 1',
    8: 'Spin direction 2',
    9: '3D mode off',
    10: '3D mode on',
    11: 'Settings request',
    12: 'Save settings',
    13: 'Extended telemetry enable',
    14: 'Extended telemetry disable',
    20: 'Spin direction normal',
    21: 'Spin direction reversed',
    22: 'LED 0 on',
    23: 'LED 1 on',
    24: 'LED 2 on',
    25: 'LED 3 on',
    26: 'LED 0 off',
    27: 'LED 1 off',
    28: 'LED 2 off',
    29: 'LED 3 off',
    30: 'Audio stream mode on/off',
    31: 'Silent mode on/off',
}

# Throttle value range. Values below the minimum either stop the motor
# or carry a command number.
THROTTLE_MIN = 48
THROTTLE_MAX = 2047
THROTTLE_3D_FORWARD_MIN = 1048

# Extended telemetry types (bird-sanctuary/extended-dshot-telemetry).
# An odd type, or type zero, carries the commutation period instead.
TLM_TYPES = {
    1: 'Temperature',
    2: 'Voltage',
    3: 'Current',
    4: 'Debug 1',
    5: 'Debug 2',
    6: 'Debug 3',
    7: 'State',
}


class Ann:
    FRAME, THROTTLE, COMMAND, CRC, TLM, BIT, WARN = range(7)


class Decoder(srd.Decoder):
    api_version = 3
    id = 'dshot'
    name = 'DShot'
    longname = 'DShot (digital ESC protocol)'
    desc = 'Digital throttle and telemetry protocol for brushless ESCs'
    license = 'gplv2+'
    inputs = ['logic']
    outputs = ['dshot']
    tags = ['Radio/control/RC']
    channels = (
        {'id': 'data', 'type': 0, 'name': 'Data', 'desc': 'DShot signal'},
    )
    options = (
        {'id': 'rate', 'desc': 'Bit rate', 'default': 'auto',
            'values': ('auto', 'DShot150', 'DShot300', 'DShot600',
                       'DShot1200')},
        {'id': 'motor', 'desc': 'Motor number', 'default': '1',
            'values': tuple(str(nr) for nr in range(1, 9))},
        {'id': 'tlm', 'desc': 'Decode ESC telemetry response',
            'default': 'on', 'values': ('on', 'off')},
        {'id': '3d', 'desc': '3D mode throttle ranges', 'default': 'off',
            'values': ('off', 'on')},
        {'id': 'bits', 'desc': 'Show individual symbol bits',
            'default': 'off', 'values': ('off', 'on')},
    )
    annotations = (
        ('frame', 'Frame'),
        ('throttle', 'Throttle'),
        ('command', 'Command'),
        ('crc', 'Checksum'),
        ('tlm', 'Telemetry'),
        ('bits', 'Symbol bits'),
        ('warn', 'Warning'),
    )
    annotation_rows = (
        ('frames', 'Frames', (Ann.FRAME, Ann.THROTTLE, Ann.COMMAND, Ann.CRC)),
        ('telemetry', 'Telemetry', (Ann.TLM,)),
        ('symbols', 'Symbols', (Ann.BIT,)),
        ('warnings', 'Warnings', (Ann.WARN,)),
    )

    def __init__(self):
        self.reset()

    def reset(self):
        self.samplerate = None
        self.idle_samples = 0
        self.edges = []

    def metadata(self, key, value):
        if key == srd.SRD_CONF_SAMPLERATE:
            self.samplerate = value

    def start(self):
        self.out_ann = self.register(srd.OUTPUT_ANN)
        self.out_py = self.register(srd.OUTPUT_PYTHON)

    def putx(self, ann, ss, es, texts):
        self.put(ss, es, self.out_ann, [ann, texts])

    def putpy(self, ann, ss, es, data):
        self.put(ss, es, self.out_py, data)

    #-- Signal segmentation --------------------------------------------

    def decode(self):
        if self.samplerate is None:
            raise SamplerateError('Cannot decode without samplerate.')
        if not self.has_channel(0):
            raise ChannelError('Need a DShot signal channel.')
        self.idle_samples = int(round(self.samplerate * IDLE_USEC * 1e-6))
        self.idle_samples = max(self.idle_samples, 1)
        while True:
            (level, ) = self.wait({0: 'e'})
            self.handle_edge(self.samplenum, level)

    def handle_edge(self, sample, level):
        # Split the edge stream into bursts. A level which does not
        # change for a while separates two bursts, it cannot occur
        # within a frame nor within a telemetry response.
        if not self.edges:
            self.edges.append((sample, level))
            return
        if sample - self.edges[-1][0] > self.idle_samples:
            self.handle_burst()
            self.edges = [(sample, level)]
        else:
            self.edges.append((sample, level))

    def burst_segments(self):
        # Turn the edge list into (level, start, end) segments. The
        # trailing idle period behind the last edge is not part of the
        # burst and is dropped.
        edges = self.edges
        segments = []
        for idx in range(len(edges) - 1):
            segments.append((edges[idx][1], edges[idx][0], edges[idx + 1][0]))
        return segments

    def handle_burst(self):
        segments = self.burst_segments()
        self.edges = []
        if len(segments) < 2:
            return
        starts_high = segments[0][0] == 1
        high = [seg for seg in segments if seg[0] == 1]
        low = [seg for seg in segments if seg[0] == 0]
        # A request frame starts with a high pulse, and carries exactly
        # one pulse per symbol. Twenty pulses are DShot1200 frames.
        if starts_high and len(high) in (16, 20) and len(low) == len(high) - 1:
            self.handle_request([(ss, es) for _, ss, es in high])
            return
        # The telemetry response is inverted: the requesting side leaves
        # the wire low, the ESC pulls it high and then drives one low
        # pulse per symbol. That makes 21 low pulses for 21 symbols. The
        # idle period before the first symbol may or may not be part of
        # the burst, depending on how long the ESC waits before it
        # answers, so the leading high level is not required to be here.
        if len(low) == 21 and self.options['tlm'] == 'on':
            ss, es = segments[0][1], segments[-1][2]
            if self.handle_response([(ss, es) for _, ss, es in low]) or \
                    self.handle_response([(ss, es) for _, ss, es in high]):
                return
            self.putx(Ann.WARN, ss, es, [
                'Telemetry response without valid symbols or checksum',
                'Undecodable telemetry response', 'TLM error'])
            return
        ss, es = segments[0][1], segments[-1][2]
        self.putx(Ann.WARN, ss, es, [
            'Unexpected burst, {} high and {} low pulses'.format(
                len(high), len(low)),
            'Unexpected burst, {} / {} pulses'.format(len(high), len(low)),
            'Unexpected burst'])

    #-- Helpers --------------------------------------------------------

    def sample_period(self, pulses):
        # Consecutive symbols start exactly one bit period apart, which
        # makes the first edge of each symbol the most reliable
        # reference. Do not use the duty cycle of the last symbol.
        if len(pulses) >= 2:
            return (pulses[-1][0] - pulses[0][0]) / (len(pulses) - 1)
        return pulses[-1][1] - pulses[0][0] if pulses else 0.0

    def pulses_to_bits(self, pulses, period):
        # A '1' symbol keeps the active level for 75% of the bit period,
        # a '0' symbol for 37.5%. The threshold is placed in the middle
        # of the two nominal duty cycles. Also count symbols which are
        # too far off to be trusted.
        bits = []
        dubious = 0
        for ss, es in pulses:
            duty = (es - ss) / period if period else 0.0
            bits.append(1 if duty > 0.5625 else 0)
            if 0.45 < duty < 0.65:
                dubious += 1
        return bits, dubious

    def rate_name(self, period):
        # Name the symbol rate which matches the measured bit period.
        # Never silently pretend that a configured rate was measured.
        if not self.samplerate or not period:
            return None, False
        usec = period * 1e6 / self.samplerate
        best = None
        for name, nominal in RATE_PERIODS:
            delta = abs(usec - nominal) / nominal
            if best is None or delta < best[1]:
                best = (name, delta)
        if best is None or best[1] > 0.25:
            return None, False
        return best[0], True

    @staticmethod
    def crc4(value):
        value &= 0xfff
        return (value ^ (value >> 4) ^ (value >> 8)) & 0xf

    def gcr_to_value(self, bits):
        # Collect five bit symbols from the 20 bit stream and look up
        # the GCR code. Neither the nibble order of DShot1200 frames nor
        # the bit order of the telemetry response is documented, so both
        # orders are tried and the checksum decides.
        symbols = []
        for idx in range(0, len(bits), 5):
            chunk = bits[idx:idx + 5]
            if len(chunk) < 5:
                return None, None, False
            symbol = 0
            for bit in chunk:
                symbol = (symbol << 1) | bit
            if symbol not in GCR_NIBBLES:
                return None, None, False
            symbols.append(GCR_NIBBLES[symbol])
        # The check sum sits in the low nibble of the decoded value, so
        # the nibble order which puts it there is the correct one.
        for order, label in ((symbols, 'high first'),
                             (list(reversed(symbols)), 'low first')):
            value = 0
            for nibble in order:
                value = (value << 4) | nibble
            if (value & 0xf) == self.crc4(value >> 4):
                return value, label, True
        value = 0
        for nibble in symbols:
            value = (value << 4) | nibble
        return value, None, False

    def annotate_bits(self, pulses, bits, period):
        if self.options['bits'] != 'on':
            return
        for (ss, es), bit in zip(pulses, bits):
            usec = (es - ss) * 1e6 / self.samplerate if self.samplerate else 0
            self.putx(Ann.BIT, ss, es, [
                '{} ({:.2f} us)'.format(bit, usec), '{}'.format(bit)])

    #-- Request frames -------------------------------------------------

    def handle_request(self, pulses):
        nsym = len(pulses)
        period = self.sample_period(pulses)
        ss, es = pulses[0][0], pulses[-1][1]
        bits, dubious = self.pulses_to_bits(pulses, period)
        self.annotate_bits(pulses, bits, period)
        gcr_order = None
        if nsym == 20:
            value16, gcr_order, crc_ok = self.gcr_to_value(bits)
            if value16 is None:
                self.putx(Ann.WARN, ss, es, [
                    'Invalid GCR symbol in DShot1200 frame',
                    'Invalid GCR symbol', 'GCR error'])
                return
        else:
            value16 = 0
            for bit in bits:
                value16 = (value16 << 1) | bit
            crc_ok = (value16 & 0xf) == self.crc4(value16 >> 4)
        value11 = value16 >> 5
        tlm_req = bool((value16 >> 4) & 1)
        crc_rx = value16 & 0xf
        crc_calc = self.crc4(value16 >> 4)
        rate, rate_ok = self.rate_name(period)
        motor = self.options['motor']

        # The configured rate is a hint for the user, not a constraint.
        # A mismatch keeps the measurement and raises a warning.
        configured = self.options['rate']
        if configured != 'auto' and rate_ok and configured != rate:
            self.putx(Ann.WARN, ss, es, [
                'Measured {} instead of configured {}'.format(rate, configured),
                'Measured {}, not {}'.format(rate, configured),
                'Rate mismatch'])
        if dubious:
            self.putx(Ann.WARN, ss, es, [
                '{} symbol(s) with ambiguous duty cycle'.format(dubious),
                '{} ambiguous symbol(s)'.format(dubious),
                'Ambiguous timing'])

        rate_txt = rate if rate else 'unknown rate'
        summary = 'M{} {} 0x{:04x}'.format(motor, rate_txt, value16)
        if gcr_order and gcr_order != 'high first':
            summary += ' ({})'.format(gcr_order)
        self.putx(Ann.FRAME, ss, es, [summary, 'M{}'.format(motor)])
        self.putpy(Ann.FRAME, ss, es, [
            'FRAME', (motor, value11, tlm_req, crc_ok, rate)])

        # Throttle values and command numbers share the same field.
        if value11 in COMMANDS:
            name = COMMANDS[value11]
            self.putx(Ann.COMMAND, ss, es, [
                'Command {} ({})'.format(value11, name),
                '{} ({})'.format(value11, name), name])
            self.putpy(Ann.COMMAND, ss, es, ['COMMAND', (value11, name)])
        elif value11 == 0:
            self.putx(Ann.THROTTLE, ss, es,
                      ['Motor stop (0)', 'Stop'])
            self.putpy(Ann.THROTTLE, ss, es, ['THROTTLE', (0, 0.0)])
        else:
            self.annotate_throttle(ss, es, value11)

        if crc_ok:
            crc_text = 'Checksum 0x{:x} correct'.format(crc_rx)
        else:
            crc_text = 'Checksum 0x{:x}, expected 0x{:x}'.format(
                crc_rx, crc_calc)
            self.putx(Ann.WARN, ss, es, [
                'Checksum mismatch, got 0x{:x} expected 0x{:x}'.format(
                    crc_rx, crc_calc),
                'Checksum mismatch', 'CRC error'])
        self.putx(Ann.CRC, ss, es, [crc_text, 'CRC {}'.format(
            'ok' if crc_ok else 'bad')])
        if tlm_req:
            self.putx(Ann.TLM, ss, es, ['Telemetry requested', 'TLM req'])

    def annotate_throttle(self, ss, es, value11):
        if value11 < THROTTLE_MIN:
            self.putx(Ann.THROTTLE, ss, es, [
                'Reserved value {}'.format(value11), 'Reserved'])
            self.putpy(Ann.THROTTLE, ss, es, ['THROTTLE', (value11, None)])
            return
        if self.options['3d'] == 'on':
            if value11 < THROTTLE_3D_FORWARD_MIN:
                span = THROTTLE_3D_FORWARD_MIN - 1 - THROTTLE_MIN
                percent = 100.0 * (value11 - THROTTLE_MIN) / span
                text = 'Throttle {} (3D reverse {:.1f}%)'.format(
                    value11, percent)
                self.putx(Ann.THROTTLE, ss, es, [
                    text, '{} reverse {:.0f}%'.format(value11, percent)])
                self.putpy(Ann.THROTTLE, ss, es,
                           ['THROTTLE', (value11, -percent)])
                return
            span = THROTTLE_MAX - THROTTLE_3D_FORWARD_MIN
            percent = 100.0 * (value11 - THROTTLE_3D_FORWARD_MIN) / span
            text = 'Throttle {} (3D forward {:.1f}%)'.format(value11, percent)
            self.putx(Ann.THROTTLE, ss, es, [
                text, '{} forward {:.0f}%'.format(value11, percent)])
            self.putpy(Ann.THROTTLE, ss, es, ['THROTTLE', (value11, percent)])
            return
        span = THROTTLE_MAX - THROTTLE_MIN
        percent = 100.0 * (value11 - THROTTLE_MIN) / span
        self.putx(Ann.THROTTLE, ss, es, [
            'Throttle {} ({:.1f}%)'.format(value11, percent),
            '{} ({:.0f}%)'.format(value11, percent),
            '{:.0f}%'.format(percent)])
        self.putpy(Ann.THROTTLE, ss, es, ['THROTTLE', (value11, percent)])

    #-- Telemetry responses --------------------------------------------

    def handle_response(self, pulses):
        # The response has a start bit and 20 payload bits, all of them
        # GCR encoded, which makes 21 symbols.
        if len(pulses) != 21:
            return False
        period = self.sample_period(pulses)
        bits, dubious = self.pulses_to_bits(pulses, period)
        if bits[0] != 0:
            return False
        value16, order, crc_ok = self.gcr_to_value(bits[1:])
        if value16 is None:
            return False
        ss, es = pulses[0][0], pulses[-1][1]
        if self.options['bits'] == 'on':
            self.annotate_bits(pulses, bits, period)

        raw12 = value16 & 0xfff
        crc_rx = value16 >> 12
        crc_calc = self.crc4(raw12)
        kind, value, text = self.decode_tlm_value(raw12)
        if kind == 'eRPM':
            self.putx(Ann.TLM, ss, es, [
                'Telemetry: {} eRPM (raw 0x{:03x})'.format(value, raw12),
                '{} eRPM'.format(value), 'eRPM'])
        else:
            self.putx(Ann.TLM, ss, es, [
                'Telemetry: {} (raw 0x{:03x})'.format(text, raw12),
                text, kind])
        self.putpy(Ann.TLM, ss, es, ['TLM', (kind, value, raw12)])

        if not crc_ok:
            self.putx(Ann.WARN, ss, es, [
                'Telemetry checksum 0x{:x}, expected 0x{:x}'.format(
                    crc_rx, crc_calc),
                'Telemetry checksum error', 'TLM CRC error'])
        if dubious:
            self.putx(Ann.WARN, ss, es, [
                'Telemetry with {} ambiguous symbol(s)'.format(dubious),
                '{} ambiguous TLM symbol(s)'.format(dubious),
                'Ambiguous timing'])
        return True

    def decode_tlm_value(self, raw12):
        # Bits 11..8 hold the telemetry type, bits 7..0 the value. An
        # odd type (or type zero) selects the plain commutation period
        # which every ESC reports.
        tlm_type = (raw12 >> 8) & 0xf
        if not tlm_type & 1 and tlm_type:
            index = tlm_type >> 1
            name = TLM_TYPES.get(index, 'Type {}'.format(index))
            value = raw12 & 0xff
            if name == 'Temperature':
                text = '{} degC'.format(value)
            elif name == 'Voltage':
                text = '{:.2f} V'.format(value * 0.25)
            elif name == 'Current':
                text = '{} A'.format(value)
            else:
                text = '{} {}'.format(name, value)
            return name, value, text
        exponent = (raw12 >> 9) & 0x7
        mantissa = raw12 & 0x1ff
        period = mantissa << exponent
        if not period:
            return 'eRPM', 0, '0 eRPM'
        # The same conversion which the flight controller firmware
        # applies to the commutation period.
        erpm = (600000 + period // 2) // period
        return 'eRPM', erpm, '{} eRPM'.format(erpm)
