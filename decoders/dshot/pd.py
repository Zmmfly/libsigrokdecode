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
talk to BLHeli_S, BLHeli_32, Bluejay and AM32 ESCs. A single wire per
motor carries 16 bit frames at 150, 300, 600 or 1200 kbit/s.

Frame layout, transmitted MSB first:
- Bits 15..5: 11 bit throttle value or command number.
- Bit 4:      request a telemetry response from the ESC.
- Bits 3..0:  4 bit checksum, the XOR of the upper three nibbles.

Values 1..47 are commands (beeps, spin direction, 3D mode, save
settings, LEDs, telemetry modes, ...), values 48..2047 are throttle.
When 3D mode is enabled, 48..1047 selects the negative and 1048..2047
the positive direction. Every bit period starts with the active level,
whose length carries the value: about 75% of the period is a '1', about
37.5% a '0'. The decision threshold also fits the 70/35% and 67/33%
duty cycles of timer-DMA and bit-banged outputs.

Bidirectional DShot inverts the line: it idles high, every bit starts
with a low level, and the checksum is inverted. The decoder recognises
this from the idle level. About 30 us after each inverted request the
ESC answers on the same wire. Its response runs at 5/4 of the request's
bit rate and has 21 bits: a low start bit, then 20 bits in which a level
change means '1' and a constant level '0' (NRZI). The 20 bits are four
5 bit GCR symbols which carry 16 bits: a 12 bit value and a 4 bit
checksum, chosen so that the XOR of all four nibbles is 0xf. The value
is either the commutation period, eeem mmmm mmmm = mantissa << exponent
in microseconds (0xfff means the motor stands still), from which
eRPM = 60000000 / period follows, or an extended telemetry frame (EDT):
an even, nonzero upper nibble selects temperature, voltage, current,
debug or state information in the lower 8 bits.

The bit period is measured from each frame, the telemetry bit time from
the response itself, so ESC clock errors of several percent are fine.

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

# Longest bit period which is accepted (DShot150 plus margin), in us.
PERIOD_MAX_USEC = 8.0
# A request whose bits are more than this far from the median period is
# not a frame.
PERIOD_TOLERANCE = 0.2
# Share of the bit period above which the active level is a '1', and the
# band around it in which the decision is reported as ambiguous.
DUTY_THRESHOLD = 0.53
DUTY_AMBIGUOUS = (0.46, 0.60)
# The ESC starts its response within this time after a request ends.
RESPONSE_WINDOW_USEC = 60.0
# Telemetry bits per request bit period.
TLM_BIT_RATIO = 0.8
TLM_BITS = 21

# 4b/5b GCR code of the telemetry response, in nibble order.
GCR_SYMBOLS = (
    0x19, 0x1b, 0x12, 0x13, 0x1d, 0x15, 0x16, 0x17,
    0x1a, 0x09, 0x0a, 0x0b, 0x1e, 0x0d, 0x0e, 0x0f,
)
GCR_NIBBLES = {sym: idx for idx, sym in enumerate(GCR_SYMBOLS)}

# Command numbers as defined by BLHeli and Betaflight. Values above
# THROTTLE_MIN are throttle values instead.
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
    32: 'Signal line telemetry disable',
    33: 'Signal line telemetry enable',
    34: 'Continuous eRPM telemetry',
    35: 'Continuous eRPM period telemetry',
    42: 'Temperature telemetry',
    43: 'Voltage telemetry',
    44: 'Current telemetry',
    45: 'Consumption telemetry',
    46: 'eRPM telemetry',
    47: 'eRPM period telemetry',
}

# Throttle value range. Values below the minimum either stop the motor
# or carry a command number.
THROTTLE_MIN = 48
THROTTLE_MAX = 2047
THROTTLE_3D_FORWARD_MIN = 1048

# Extended telemetry types (bird-sanctuary/extended-dshot-telemetry),
# indexed by the upper nibble of the 12 bit value divided by two.
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
        {'id': 'bits', 'desc': 'Show individual bits',
            'default': 'off', 'values': ('off', 'on')},
    )
    annotations = (
        ('frame', 'Frame'),
        ('throttle', 'Throttle'),
        ('command', 'Command'),
        ('crc', 'Checksum'),
        ('tlm', 'Telemetry'),
        ('bits', 'Bits'),
        ('warn', 'Warning'),
    )
    annotation_rows = (
        ('frames', 'Frames', (Ann.FRAME, Ann.THROTTLE, Ann.COMMAND, Ann.CRC)),
        ('telemetry', 'Telemetry', (Ann.TLM,)),
        ('symbols', 'Bits', (Ann.BIT,)),
        ('warnings', 'Warnings', (Ann.WARN,)),
    )

    def __init__(self):
        self.reset()

    def reset(self):
        self.samplerate = None
        self.state = 'IDLE'
        self.burst = []
        self.dropped = 0
        self.drop_ss = 0
        self.last_edge = None
        self.period = 0.0
        self.req_end = 0
        self.resp = []

    def metadata(self, key, value):
        if key == srd.SRD_CONF_SAMPLERATE:
            self.samplerate = value

    def start(self):
        self.out_ann = self.register(srd.OUTPUT_ANN)
        self.out_py = self.register(srd.OUTPUT_PYTHON)

    def putx(self, ann, ss, es, texts):
        self.put(int(ss), int(es), self.out_ann, [ann, texts])

    def putpy(self, ss, es, data):
        self.put(int(ss), int(es), self.out_py, data)

    def usec(self, samples):
        return samples * 1e6 / self.samplerate

    def samples(self, usec):
        return usec * self.samplerate / 1e6

    #-- Edge stream ----------------------------------------------------

    def decode(self):
        if self.samplerate is None:
            raise SamplerateError('Cannot decode without samplerate.')
        if not self.has_channel(0):
            raise ChannelError('Need a DShot signal channel.')
        self.wait()
        self.last_edge = self.samplenum
        while True:
            (level, ) = self.wait({0: 'e'})
            self.handle_edge(self.samplenum, level)

    def end(self):
        # Complete whatever the end of the capture interrupted: a
        # response which is still being received, or a burst of pulses.
        if self.samplerate is None:
            return
        if self.state == 'RESP':
            self.finish_response()
        elif self.state == 'REQ':
            self.finish_burst()
        self.state = 'IDLE'

    def handle_edge(self, sample, level):
        self.expire(sample)
        if self.state == 'WAIT':
            if level == 0:
                # The ESC pulls the idle-high line low: a response.
                self.state = 'RESP'
                self.resp = [(sample, level)]
                self.last_edge = sample
                return
            self.state = 'IDLE'
        elif self.state == 'RESP':
            self.resp.append((sample, level))
            self.last_edge = sample
            return
        self.request_edge(sample, level)
        self.last_edge = sample

    def expire(self, sample):
        # Timeouts which elapsed before this edge.
        if self.state == 'REQ' and self.last_edge is not None:
            if sample - self.last_edge > self.gap_limit():
                self.finish_burst()
        elif self.state == 'WAIT':
            if sample > self.req_end + self.samples(RESPONSE_WINDOW_USEC):
                if self.options['tlm'] == 'on':
                    self.putx(Ann.WARN, self.req_end,
                              self.req_end + self.samples(RESPONSE_WINDOW_USEC), [
                                  'No telemetry response', 'No response', 'No TLM'])
                self.state = 'IDLE'
        elif self.state == 'RESP':
            tb = self.period * TLM_BIT_RATIO
            if sample - self.resp[0][0] > (TLM_BITS + 4) * tb:
                self.finish_response()

    #-- Request frames -------------------------------------------------

    def gap_limit(self):
        # Within a frame the line keeps a level for at most one bit
        # period. Before two leading edges are known, allow the longest
        # supported period.
        lead = [s for s, lvl in self.burst[0::2]]
        if len(lead) >= 2:
            periods = sorted(b - a for a, b in zip(lead, lead[1:]))
            return 3 * periods[len(periods) // 2]
        return 3 * self.samples(PERIOD_MAX_USEC)

    def request_edge(self, sample, level):
        if self.state == 'IDLE':
            self.state = 'REQ'
            self.burst = []
            self.dropped = 0
        self.burst.append((sample, level))
        while len(self.burst) >= 32:
            if self.try_request():
                return
            # Not a frame yet: drop the oldest pulse and retry with the
            # next edges, which resynchronises in the middle of a burst.
            if not self.dropped:
                self.drop_ss = self.burst[0][0]
            self.burst = self.burst[2:]
            self.dropped += 1

    def finish_burst(self):
        pulses = len(self.burst) // 2 + self.dropped
        if pulses and self.burst:
            ss = self.burst[0][0] if not self.dropped else self.drop_ss
            es = self.burst[-1][0]
            self.putx(Ann.WARN, ss, es, [
                'Unexpected burst of {} pulses, no frame'.format(pulses),
                'Unexpected burst ({})'.format(pulses), 'Burst'])
        self.burst = []
        self.dropped = 0
        self.state = 'IDLE'

    def try_request(self):
        edges = self.burst[:32]
        active = edges[0][1]
        if any(lvl != active for _, lvl in edges[0::2]) or \
                any(lvl == active for _, lvl in edges[1::2]):
            return False
        lead = [s for s, _ in edges[0::2]]
        trail = [s for s, _ in edges[1::2]]
        periods = sorted(b - a for a, b in zip(lead, lead[1:]))
        period = periods[len(periods) // 2]
        if period <= 0 or self.usec(period) > PERIOD_MAX_USEC:
            return False
        if periods[0] < period * (1 - PERIOD_TOLERANCE) or \
                periods[-1] > period * (1 + PERIOD_TOLERANCE):
            return False
        duties = [(t - l) / period for l, t in zip(lead, trail)]
        if min(duties) <= 0.1 or max(duties) >= 0.92:
            return False
        if self.dropped:
            self.putx(Ann.WARN, self.drop_ss, lead[0], [
                'Unexpected burst of {} pulses, no frame'.format(self.dropped),
                'Unexpected burst ({})'.format(self.dropped), 'Burst'])
        self.burst = self.burst[32:]
        self.dropped = 0
        # The mean of the 15 periods is far more precise than any single
        # one, which matters at low sample rates.
        period = (lead[-1] - lead[0]) / 15.0
        duties = [(t - l) / period for l, t in zip(lead, trail)]
        self.handle_request(lead, duties, period, active == 0)
        return True

    def rate_name(self, period):
        # Name the symbol rate which matches the measured bit period.
        # Never silently pretend that a configured rate was measured.
        usec = self.usec(period)
        best = None
        for name, nominal in RATE_PERIODS:
            delta = abs(usec - nominal) / nominal
            if best is None or delta < best[1]:
                best = (name, delta)
        if best is None or best[1] > 0.25:
            return None
        return best[0]

    @staticmethod
    def crc4(value):
        value &= 0xfff
        return (value ^ (value >> 4) ^ (value >> 8)) & 0xf

    def handle_request(self, lead, duties, period, inverted):
        ss = lead[0]
        es = lead[0] + 16 * period
        bits = [1 if duty > DUTY_THRESHOLD else 0 for duty in duties]
        dubious = sum(1 for duty in duties
                      if DUTY_AMBIGUOUS[0] < duty < DUTY_AMBIGUOUS[1])
        value16 = 0
        for bit in bits:
            value16 = (value16 << 1) | bit
        value11 = value16 >> 5
        tlm_req = bool((value16 >> 4) & 1)
        crc_rx = value16 & 0xf
        crc_calc = self.crc4(value16 >> 4)
        if inverted:
            crc_calc ^= 0xf
        crc_ok = crc_rx == crc_calc
        rate = self.rate_name(period)
        motor = self.options['motor']

        if self.options['bits'] == 'on':
            for idx, (start, bit) in enumerate(zip(lead, bits)):
                width = self.usec(duties[idx] * period)
                self.putx(Ann.BIT, start, start + period, [
                    '{} ({:.2f} us)'.format(bit, width), '{}'.format(bit)])

        # The configured rate is a hint for the user, not a constraint.
        configured = self.options['rate']
        if configured != 'auto' and rate and configured != rate:
            self.putx(Ann.WARN, ss, es, [
                'Measured {} instead of configured {}'.format(rate, configured),
                'Measured {}, not {}'.format(rate, configured),
                'Rate mismatch'])
        if dubious:
            self.putx(Ann.WARN, ss, es, [
                '{} bit(s) with ambiguous duty cycle'.format(dubious),
                '{} ambiguous bit(s)'.format(dubious), 'Ambiguous timing'])

        rate_txt = rate if rate else '{:.0f} kbit/s'.format(1e3 / self.usec(period))
        mode = ' bidir' if inverted else ''
        self.putx(Ann.FRAME, ss, es, [
            'M{} {}{} 0x{:04x}'.format(motor, rate_txt, mode, value16),
            'M{} 0x{:04x}'.format(motor, value16), 'M{}'.format(motor)])
        self.putpy(ss, es, ['FRAME', (motor, value11, tlm_req, crc_ok, rate)])

        # Throttle values and command numbers share the same field.
        if value11 in COMMANDS:
            name = COMMANDS[value11]
            self.putx(Ann.COMMAND, ss, es, [
                'Command {} ({})'.format(value11, name),
                '{} ({})'.format(value11, name), name])
            self.putpy(ss, es, ['COMMAND', (value11, name)])
        elif value11 == 0:
            self.putx(Ann.THROTTLE, ss, es, ['Motor stop (0)', 'Stop'])
            self.putpy(ss, es, ['THROTTLE', (0, 0.0)])
        else:
            self.annotate_throttle(ss, es, value11)

        if crc_ok:
            crc_text = 'Checksum 0x{:x} correct'.format(crc_rx)
        else:
            crc_text = 'Checksum 0x{:x}, expected 0x{:x}'.format(crc_rx, crc_calc)
            self.putx(Ann.WARN, ss, es, [
                'Checksum mismatch, got 0x{:x} expected 0x{:x}'.format(
                    crc_rx, crc_calc),
                'Checksum mismatch', 'CRC error'])
        self.putx(Ann.CRC, ss, es, [crc_text, 'CRC {}'.format(
            'ok' if crc_ok else 'bad')])
        if tlm_req:
            self.putx(Ann.TLM, ss, es, ['Telemetry requested', 'TLM req'])

        self.period = period
        if inverted:
            # Bidirectional DShot: the ESC answers every frame.
            self.state = 'WAIT'
            self.req_end = es
        else:
            self.state = 'IDLE'

    def annotate_throttle(self, ss, es, value11):
        if value11 < THROTTLE_MIN:
            self.putx(Ann.THROTTLE, ss, es, [
                'Reserved value {}'.format(value11), 'Reserved'])
            self.putpy(ss, es, ['THROTTLE', (value11, None)])
            return
        if self.options['3d'] == 'on':
            if value11 < THROTTLE_3D_FORWARD_MIN:
                span = THROTTLE_3D_FORWARD_MIN - 1 - THROTTLE_MIN
                percent = 100.0 * (value11 - THROTTLE_MIN) / span
                text = 'Throttle {} (3D reverse {:.1f}%)'.format(
                    value11, percent)
                self.putx(Ann.THROTTLE, ss, es, [
                    text, '{} reverse {:.0f}%'.format(value11, percent)])
                self.putpy(ss, es, ['THROTTLE', (value11, -percent)])
                return
            span = THROTTLE_MAX - THROTTLE_3D_FORWARD_MIN
            percent = 100.0 * (value11 - THROTTLE_3D_FORWARD_MIN) / span
            text = 'Throttle {} (3D forward {:.1f}%)'.format(value11, percent)
            self.putx(Ann.THROTTLE, ss, es, [
                text, '{} forward {:.0f}%'.format(value11, percent)])
            self.putpy(ss, es, ['THROTTLE', (value11, percent)])
            return
        span = THROTTLE_MAX - THROTTLE_MIN
        percent = 100.0 * (value11 - THROTTLE_MIN) / span
        self.putx(Ann.THROTTLE, ss, es, [
            'Throttle {} ({:.1f}%)'.format(value11, percent),
            '{} ({:.0f}%)'.format(value11, percent),
            '{:.0f}%'.format(percent)])
        self.putpy(ss, es, ['THROTTLE', (value11, percent)])

    #-- Telemetry responses --------------------------------------------

    def finish_response(self):
        edges = self.resp
        self.resp = []
        self.state = 'IDLE'
        if self.options['tlm'] != 'on' or not edges:
            return
        self.handle_response(edges)

    def response_slots(self, edges, tb):
        # Bit slot of every edge: first from the rounded run lengths, then
        # once more from the bit time fitted to those slots, which follows
        # an ESC clock that is a few percent off.
        t0 = edges[0][0]
        slots = [0]
        for (a, _), (b, _) in zip(edges, edges[1:]):
            slots.append(slots[-1] + max(1, int(round((b - a) / tb))))
        den = sum(k * k for k in slots)
        if den:
            fit = sum(k * (s - t0) for k, (s, _) in zip(slots, edges)) / den
            if abs(fit / tb - 1) < 0.15:
                tb = fit
                refined = [0]
                for s, _ in edges[1:]:
                    refined.append(max(refined[-1] + 1, int(round((s - t0) / tb))))
                slots = refined
        return slots, tb

    def handle_response(self, edges):
        tb = self.period * TLM_BIT_RATIO
        slots, tb = self.response_slots(edges, tb)
        ss = edges[0][0]
        es = ss + TLM_BITS * tb
        if slots[-1] > TLM_BITS or edges[-1][1] != 1:
            self.putx(Ann.WARN, ss, max(es, edges[-1][0]), [
                'Telemetry response longer than {} bits'.format(TLM_BITS),
                'Telemetry too long', 'TLM error'])
            return
        levels = []
        idx = 0
        for slot in range(TLM_BITS):
            while idx + 1 < len(edges) and slots[idx + 1] <= slot:
                idx += 1
            levels.append(edges[idx][1])
        gcr = 0
        for slot in range(1, TLM_BITS):
            gcr = (gcr << 1) | (levels[slot] ^ levels[slot - 1])

        if self.options['bits'] == 'on':
            for slot in range(TLM_BITS):
                text = 'S' if slot == 0 else str(levels[slot] ^ levels[slot - 1])
                self.putx(Ann.BIT, ss + slot * tb, ss + (slot + 1) * tb,
                          [text])

        value16 = 0
        for idx in range(3, -1, -1):
            symbol = (gcr >> (5 * idx)) & 0x1f
            if symbol not in GCR_NIBBLES:
                self.putx(Ann.WARN, ss, es, [
                    'Invalid GCR symbol 0x{:02x} in telemetry response'.format(symbol),
                    'Invalid GCR symbol', 'GCR error'])
                return
            value16 = (value16 << 4) | GCR_NIBBLES[symbol]

        raw12 = value16 >> 4
        csum = value16 ^ (value16 >> 4) ^ (value16 >> 8) ^ (value16 >> 12)
        csum_ok = (csum & 0xf) == 0xf
        kind, value, texts = self.decode_tlm_value(raw12)
        if not csum_ok:
            texts = [texts[0] + ' (checksum error)'] + texts[1:]
        self.putx(Ann.TLM, ss, es, texts)
        self.putpy(ss, es, ['TLM', (kind, value, raw12)])
        if not csum_ok:
            self.putx(Ann.WARN, ss, es, [
                'Telemetry checksum mismatch, nibble XOR 0x{:x}'.format(csum & 0xf),
                'Telemetry checksum error', 'TLM CRC error'])

    def decode_tlm_value(self, raw12):
        # An even, nonzero upper nibble selects an extended telemetry
        # frame with the value in the lower 8 bits. Everything else is
        # the commutation period.
        tlm_type = (raw12 >> 8) & 0xf
        if tlm_type and not tlm_type & 1:
            name = TLM_TYPES[tlm_type >> 1]
            value = raw12 & 0xff
            if name == 'Temperature':
                text = '{} degC'.format(value)
            elif name == 'Voltage':
                text = '{:.2f} V'.format(value * 0.25)
            elif name == 'Current':
                text = '{} A'.format(value)
            else:
                text = '0x{:02x}'.format(value)
            return name, value, [
                '{}: {} (raw 0x{:03x})'.format(name, text, raw12),
                '{} {}'.format(name, text), text]
        period = (raw12 & 0x1ff) << (raw12 >> 9)
        if raw12 == 0xfff or not period:
            return 'eRPM', 0, ['eRPM 0, motor stopped (raw 0x{:03x})'.format(raw12),
                               'eRPM 0 (stopped)', '0 eRPM']
        erpm = (60000000 + period // 2) // period
        return 'eRPM', erpm, [
            'eRPM {} (period {} us, raw 0x{:03x})'.format(erpm, period, raw12),
            'eRPM {} ({} us)'.format(erpm, period), '{} eRPM'.format(erpm)]
