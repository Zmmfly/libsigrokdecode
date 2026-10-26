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
ExpressLRS (ELRS) uses the CRSF serial protocol for the link between
the receiver and the flight controller, and between the radio and the
transmitter module. CRSF frames consist of an address byte, a length
byte, a type byte, the payload, and a CRC-8 with polynomial 0xD5 which
covers the type and the payload.

Frames with a type in the 0x28..0x96 range carry the destination and
the origin address behind the type byte. ELRS uses several of them:
0x2E reports the packet counters and the warning flags, 0x29 identifies
a device, 0x2B..0x2D carry the settings which the LUA script shows, and
0x32 carries commands like binding. The 0x16 frame carries 16 eleven
bit channels, ELRS appends an extra byte which holds the armed state.

Note that only the serial side of the link is decoded. ExpressLRS
transmits LoRa chirps over the air, with FLRC or FSK in some of its
modes. Those need an RF receiver and demodulation, which a logic
analyzer cannot provide. Capturing the SPI bus between the MCU and the
radio chip is the closest a logic analyzer can get to the air interface,
and that data is whitened and FEC protected rather than being CRSF.

This decoder replaces the plain 'crsf' decoder for ELRS setups: it
verifies the checksum of every frame, names both addresses, and knows
the ELRS specific frame types. Ordinary CRSF frames are still decoded,
so the decoder also works on a plain Crossfire link.

OUTPUT_PYTHON format:

Packet:
(<ptype>, <pdata>)

This is the list of <ptype> codes and their respective <pdata> values:
 - 'FRAME': (<address>, <type>, <dest>, <orig>, <crc_ok>, <payload>)
 - 'CHANNELS': (<tuple of 16 channel values>, <armed or None>)
 - 'LINK': (<uplink RSSI 1 and 2>, <uplink LQ>, <uplink SNR>,
            <antenna>, <rf mode>, <TX power>, <downlink RSSI>,
            <downlink LQ>, <downlink SNR>)
 - 'STATUS': (<bad packets>, <good packets>, <flags>, <message>)
 - 'BIND': <True>
'''

import sigrokdecode as srd


CRC8_POLY = 0xd5
# The length byte counts the type, the payload and the checksum.
LENGTH_MIN = 2
LENGTH_MAX = 62
# Frame types in this range carry destination and origin addresses.
EXT_TYPE_MIN = 0x28
EXT_TYPE_MAX = 0x96

# Addresses which CRSF and ELRS assign to devices.
ADDRESSES = {
    0x00: 'Broadcast',
    0x0e: 'Cloud',
    0x10: 'USB',
    0x12: 'Bluetooth/WiFi module',
    0x13: 'WiFi receiver',
    0x14: 'Video receiver',
    0x80: 'OSD / TBS Core PNP Pro',
    0x8a: 'Reserved',
    0x90: 'ESC 1',
    0x91: 'ESC 2',
    0x92: 'ESC 3',
    0x93: 'ESC 4',
    0x94: 'ESC 5',
    0x95: 'ESC 6',
    0x96: 'ESC 7',
    0x97: 'ESC 8',
    0xc0: 'Current sensor',
    0xc2: 'GPS',
    0xc4: 'Blackbox',
    0xc8: 'Flight controller',
    0xca: 'Reserved',
    0xcc: 'Race tag',
    0xce: 'VTX',
    0xea: 'Radio transmitter',
    0xec: 'CRSF receiver',
    0xee: 'CRSF transmitter',
    0xef: 'ELRS LUA',
}

FRAME_TYPES = {
    0x02: 'GPS',
    0x03: 'GPS time',
    0x07: 'Vario',
    0x08: 'Battery sensor',
    0x09: 'Baro altitude',
    0x0a: 'Airspeed',
    0x0b: 'Heartbeat',
    0x0c: 'RPM',
    0x0d: 'Temperature',
    0x0e: 'Cells',
    0x14: 'Link statistics',
    0x16: 'RC channels',
    0x1e: 'Attitude',
    0x21: 'Flight mode',
    0x28: 'Device ping',
    0x29: 'Device info',
    0x2b: 'Parameter settings entry',
    0x2c: 'Parameter read',
    0x2d: 'Parameter write',
    0x2e: 'ELRS status',
    0x32: 'Command',
    0x3a: 'Handset',
    0x78: 'KISS request',
    0x79: 'KISS response',
    0x7a: 'MSP request',
    0x7b: 'MSP response',
    0x7c: 'MSP write',
    0x80: 'ArduPilot response',
}

# RF modes as used in the link statistics frame. The numbering is
# defined by the ExpressLRS rate list, 2.4 GHz entries start at 20.
RF_MODES = {
    0: '900 MHz 25 Hz',
    1: '900 MHz 50 Hz',
    2: '900 MHz 100 Hz',
    3: '900 MHz 100 Hz 8ch',
    4: '900 MHz 150 Hz',
    5: '900 MHz 200 Hz',
    6: '900 MHz 200 Hz 8ch',
    7: '900 MHz 250 Hz',
    8: '900 MHz 333 Hz 8ch',
    9: '900 MHz 500 Hz',
    10: '900 MHz 50 Hz DVDA',
    11: 'FSK 900 MHz 1000 Hz 8ch',
    20: '2.4 GHz 25 Hz',
    21: '2.4 GHz 50 Hz',
    22: '2.4 GHz 100 Hz',
    23: '2.4 GHz 100 Hz 8ch',
    24: '2.4 GHz 150 Hz',
    25: '2.4 GHz 200 Hz',
    26: '2.4 GHz 200 Hz 8ch',
    27: '2.4 GHz 250 Hz',
    28: '2.4 GHz 333 Hz 8ch',
    29: '2.4 GHz 500 Hz',
    30: 'FLRC 2.4 GHz 250 Hz DVDA',
    31: 'FLRC 2.4 GHz 500 Hz DVDA',
    32: 'FLRC 2.4 GHz 500 Hz',
    33: 'FLRC 2.4 GHz 1000 Hz',
    34: 'FSK 2.4 GHz 250 Hz DVDA',
    35: 'FSK 2.4 GHz 500 Hz DVDA',
    36: 'FSK 2.4 GHz 1000 Hz',
    100: '2.4 GHz dual 100 Hz 8ch',
    101: '2.4 GHz dual 150 Hz',
}

TX_POWERS = {
    0: '0 mW',
    1: '10 mW',
    2: '25 mW',
    3: '100 mW',
    4: '500 mW',
    5: '1000 mW',
    6: '2000 mW',
    7: '250 mW',
    8: '50 mW',
}

# Warning and status bits of the ELRS status frame.
STATUS_FLAGS = (
    (0x01, 'connected'),
    (0x02, 'status 1'),
    (0x04, 'model mismatch'),
    (0x08, 'armed'),
    (0x10, 'warning 1'),
    (0x20, 'not while connected'),
    (0x40, 'baud rate too low'),
    (0x80, 'critical warning 2'),
)

# Channel values in CRSF are 11 bit, and the usual endpoints are these.
CHANNEL_MIN = 172
CHANNEL_MID = 992
CHANNEL_MAX = 1811
USEC_MIN = 988
USEC_MID = 1500
USEC_MAX = 2012


class Ann:
    SYNC, CRC, FRAME, CHANNELS, LINK, ELRS, WARN = range(7)


class Decoder(srd.Decoder):
    api_version = 3
    id = 'elrs'
    name = 'ExpressLRS'
    longname = 'ExpressLRS (CRSF serial link)'
    desc = 'ExpressLRS link between receiver, flight controller and module'
    license = 'gplv2+'
    inputs = ['uart']
    outputs = ['elrs']
    tags = ['Radio/control/RC']
    options = (
        {'id': 'crc', 'desc': 'Decode frames with a wrong checksum',
            'default': 'off', 'values': ('off', 'on')},
        {'id': 'channels', 'desc': 'Annotate each channel separately',
            'default': 'off', 'values': ('off', 'on')},
        {'id': 'usec', 'desc': 'Show channel values in microseconds',
            'default': 'on', 'values': ('on', 'off')},
    )
    annotations = (
        ('address', 'Address'),
        ('crc', 'Checksum'),
        ('frame', 'Frame'),
        ('channels', 'Channels'),
        ('link', 'Link statistics'),
        ('elrs', 'ELRS information'),
        ('warn', 'Warning'),
    )
    annotation_rows = (
        ('framing', 'Framing', (Ann.SYNC, Ann.CRC)),
        ('frames', 'Frames', (Ann.FRAME,)),
        ('channels', 'Channels', (Ann.CHANNELS,)),
        ('link', 'Link', (Ann.LINK,)),
        ('elrs', 'ELRS', (Ann.ELRS,)),
        ('warnings', 'Warnings', (Ann.WARN,)),
    )

    def __init__(self):
        self.reset()

    def reset(self):
        self.state = 'SYNC'
        self.frame = []
        self.body_len = 0
        self.dropped = 0

    def start(self):
        self.out_ann = self.register(srd.OUTPUT_ANN)
        self.out_py = self.register(srd.OUTPUT_PYTHON)

    def putx(self, ann, ss, es, texts):
        self.put(ss, es, self.out_ann, [ann, texts])

    def putpy(self, ss, es, data):
        self.put(ss, es, self.out_py, data)

    #-- Frame assembly -------------------------------------------------

    def decode(self, ss, es, data):
        ptype, rxtx, pdata = data
        if ptype != 'FRAME':
            return
        value, valid = pdata
        if not valid:
            self.putx(Ann.WARN, ss, es, [
                'Invalid UART frame in the CRSF stream', 'UART error'])
            return
        self.handle_byte(value, ss, es)

    def handle_byte(self, value, ss, es):
        if self.state == 'SYNC':
            # Bytes in front of an address are noise on the wire, the
            # frame counter keeps track of how much was skipped.
            if value in ADDRESSES:
                self.frame = [(value, ss, es)]
                self.state = 'LEN'
            else:
                self.dropped += 1
            return
        if self.state == 'LEN':
            if LENGTH_MIN <= value <= LENGTH_MAX:
                self.frame.append((value, ss, es))
                self.body_len = value
                self.state = 'BODY'
                return
            # The byte was neither a length nor part of a frame. It may
            # still be the address of the next frame.
            self.putx(Ann.WARN, ss, es, [
                'Invalid frame length {}'.format(value),
                'Invalid length {}'.format(value), 'Bad length'])
            self.reset()
            self.handle_byte(value, ss, es)
            return
        self.frame.append((value, ss, es))
        if len(self.frame) >= 2 + self.body_len:
            self.handle_frame()
            self.reset()

    @staticmethod
    def crc8(data):
        crc = 0
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x80:
                    crc = ((crc << 1) ^ CRC8_POLY) & 0xff
                else:
                    crc = (crc << 1) & 0xff
        return crc

    @staticmethod
    def address_name(value):
        return ADDRESSES.get(value, 'Unknown address 0x{:02x}'.format(value))

    @staticmethod
    def frame_type_name(value):
        return FRAME_TYPES.get(value, 'Unknown type 0x{:02x}'.format(value))

    def handle_frame(self):
        sync = self.frame[0][0]
        body = [item[0] for item in self.frame[2:]]
        ss, es = self.frame[0][1], self.frame[-1][2]
        ftype = body[0]
        crc_rx = body[-1]
        payload = body[1:-1]
        crc_calc = self.crc8(body[:-1])
        crc_ok = crc_rx == crc_calc
        type_name = self.frame_type_name(ftype)

        self.putx(Ann.SYNC, self.frame[0][1], self.frame[0][2],
                  [self.address_name(sync), '0x{:02x}'.format(sync)])
        if crc_ok:
            self.putx(Ann.CRC, self.frame[-1][1], self.frame[-1][2],
                      ['Checksum 0x{:02x} correct'.format(crc_rx),
                       'CRC ok'])
        else:
            self.putx(Ann.WARN, self.frame[-1][1], self.frame[-1][2], [
                'Checksum 0x{:02x}, expected 0x{:02x}'.format(
                    crc_rx, crc_calc),
                'Checksum mismatch', 'CRC error'])

        # Extended frames carry destination and origin addresses.
        dest = orig = None
        ext_payload = payload
        if EXT_TYPE_MIN <= ftype <= EXT_TYPE_MAX and len(payload) >= 2:
            dest, orig = payload[0], payload[1]
            ext_payload = payload[2:]
            text = '{}: {} -> {}, {} byte(s)'.format(
                type_name, self.address_name(orig),
                self.address_name(dest), len(ext_payload))
        else:
            text = '{}: {} byte(s)'.format(type_name, len(payload))
        self.putx(Ann.FRAME, ss, es, [text, type_name])
        self.putpy(ss, es, ['FRAME', (sync, ftype, dest, orig, crc_ok,
                                      bytes(ext_payload))])

        if not crc_ok and self.options['crc'] != 'on':
            return
        self.handle_payload(ftype, ext_payload, ss, es)

    #-- Payload handlers -----------------------------------------------

    def handle_payload(self, ftype, payload, ss, es):
        if ftype == 0x16:
            self.handle_channels(payload, ss, es)
        elif ftype == 0x14:
            self.handle_link_stats(payload, ss, es)
        elif ftype == 0x2e:
            self.handle_status(payload, ss, es)
        elif ftype == 0x29:
            self.handle_device_info(payload, ss, es)
        elif ftype == 0x2b:
            self.handle_param_entry(payload, ss, es)
        elif ftype == 0x32:
            self.handle_command(payload, ss, es)
        elif ftype == 0x1e:
            self.handle_attitude(payload, ss, es)
        elif ftype == 0x02:
            self.handle_gps(payload, ss, es)
        elif ftype == 0x08:
            self.handle_battery(payload, ss, es)
        elif ftype == 0x21:
            self.handle_text(payload, ss, es, 'Flight mode')
        elif ftype in (0x7a, 0x7b, 0x7c):
            self.handle_msp(ftype, payload, ss, es)
        else:
            if payload:
                self.putx(Ann.FRAME, ss, es, [
                    'Payload: ' + ' '.join('{:02x}'.format(b) for b in payload)])

    @staticmethod
    def channel_to_usec(value):
        # Linear scaling between the CRSF channel range and the pulse
        # width range which the CRSF specification defines.
        span = CHANNEL_MAX - CHANNEL_MIN
        usec = USEC_MIN + (value - CHANNEL_MIN) * (USEC_MAX - USEC_MIN) / span
        return int(round(usec))

    def handle_channels(self, payload, ss, es):
        if len(payload) < 22:
            self.putx(Ann.WARN, ss, es, [
                'RC channels frame is too short ({} byte(s))'.format(
                    len(payload)), 'Short channel frame'])
            return
        bits = int.from_bytes(bytes(payload[:22]), byteorder='little')
        channels = [(bits >> (11 * idx)) & 0x7ff for idx in range(16)]
        armed = None
        if len(payload) > 22:
            # ELRS appends the armed state behind the channel data.
            armed = bool(payload[22] & 0x01)
            arm_ch5 = bool(payload[22] & 0x02)
            self.putx(Ann.ELRS, ss, es, [
                'Armed: {}, arm via CH5: {}'.format(
                    'yes' if armed else 'no', 'yes' if arm_ch5 else 'no'),
                'Armed {}, CH5 arm {}'.format(
                    'yes' if armed else 'no', 'yes' if arm_ch5 else 'no')])
        show_usec = self.options['usec'] == 'on'
        if show_usec:
            text = ' '.join('{}us'.format(self.channel_to_usec(ch))
                            for ch in channels)
        else:
            text = ' '.join(str(ch) for ch in channels)
        self.putx(Ann.CHANNELS, ss, es, [text])
        if self.options['channels'] == 'on':
            for idx, ch in enumerate(channels):
                if show_usec:
                    label = 'CH{}: {} ({} us)'.format(
                        idx + 1, ch, self.channel_to_usec(ch))
                else:
                    label = 'CH{}: {}'.format(idx + 1, ch)
                self.putx(Ann.CHANNELS, ss, es, [label])
        self.putpy(ss, es, ['CHANNELS', (tuple(channels), armed)])

    def handle_link_stats(self, payload, ss, es):
        if len(payload) < 10:
            self.putx(Ann.WARN, ss, es, [
                'Link statistics frame is too short ({} byte(s))'.format(
                    len(payload)), 'Short link frame'])
            return

        def signed(value):
            return value - 256 if value > 127 else value

        up_rssi1 = payload[0]
        up_rssi2 = payload[1]
        up_lq = payload[2]
        up_snr = signed(payload[3])
        antenna = payload[4]
        rf_mode = payload[5]
        tx_power = payload[6]
        dn_rssi = payload[7]
        dn_lq = payload[8]
        dn_snr = signed(payload[9])
        mode_name = RF_MODES.get(rf_mode, 'mode {}'.format(rf_mode))
        power_name = TX_POWERS.get(tx_power, 'level {}'.format(tx_power))
        self.putx(Ann.LINK, ss, es, [
            'Uplink: RSSI1 {} dBm, RSSI2 {} dBm, LQ {}%, SNR {} dB'.format(
                -up_rssi1, -up_rssi2, up_lq, up_snr),
            'Up {} / {} dBm, {}%, {} dB'.format(
                -up_rssi1, -up_rssi2, up_lq, up_snr)])
        self.putx(Ann.LINK, ss, es, [
            'Mode {}, antenna {}, TX power {}'.format(
                mode_name, antenna + 1, power_name),
            '{}, ant {}, {}'.format(mode_name, antenna + 1, power_name)])
        self.putx(Ann.LINK, ss, es, [
            'Downlink: RSSI {} dBm, LQ {}%, SNR {} dB'.format(
                -dn_rssi, dn_lq, dn_snr),
            'Down {} dBm, {}%, {} dB'.format(-dn_rssi, dn_lq, dn_snr)])
        self.putpy(ss, es, ['LINK', (up_rssi1, up_rssi2, up_lq, up_snr,
                                     antenna, rf_mode, tx_power,
                                     dn_rssi, dn_lq, dn_snr)])

    def handle_status(self, payload, ss, es):
        if len(payload) < 4:
            self.putx(Ann.WARN, ss, es, [
                'ELRS status frame is too short ({} byte(s))'.format(
                    len(payload)), 'Short status frame'])
            return
        bad = payload[0]
        good = (payload[1] << 8) | payload[2]
        flags = payload[3]
        message = bytes(payload[4:]).split(b'\x00')[0].decode('latin-1')
        names = [name for bit, name in STATUS_FLAGS if flags & bit]
        self.putx(Ann.ELRS, ss, es, [
            'ELRS status: {} bad, {} good packet(s), flags 0x{:02x}'.format(
                bad, good, flags),
            '{} bad / {} good, 0x{:02x}'.format(bad, good, flags)])
        self.putx(Ann.ELRS, ss, es, [
            'Status flags: {}'.format(', '.join(names) if names else 'none'),
            ', '.join(names) if names else 'no flags'])
        if message:
            self.putx(Ann.ELRS, ss, es, [
                'Message: {}'.format(message), message])
        self.putpy(ss, es, ['STATUS', (bad, good, flags, message)])

    def handle_device_info(self, payload, ss, es):
        raw = bytes(payload)
        nul = raw.find(b'\x00')
        if nul < 0:
            self.putx(Ann.WARN, ss, es, [
                'Device info without a device name', 'Short device info'])
            return
        name = raw[:nul].decode('latin-1')
        fields = raw[nul + 1:]
        if len(fields) < 13:
            self.putx(Ann.WARN, ss, es, [
                'Truncated device info frame', 'Short device info'])
            return
        serial = int.from_bytes(fields[0:4], byteorder='big')
        hardware = int.from_bytes(fields[4:8], byteorder='big')
        software = int.from_bytes(fields[8:12], byteorder='big')
        count = fields[12]
        param_version = fields[13] if len(fields) > 13 else None
        version = '{}.{}.{}'.format(
            (software >> 24) & 0xff, (software >> 16) & 0xff,
            (software >> 8) & 0xff)
        text = 'Device "{}", version {}, serial 0x{:08x}'.format(
            name, version, serial)
        self.putx(Ann.ELRS, ss, es, [text, '"{}" {}'.format(name, version)])
        extra = 'Hardware 0x{:08x}, {} parameter(s)'.format(hardware, count)
        if param_version is not None:
            extra += ', parameter version {}'.format(param_version)
        self.putx(Ann.ELRS, ss, es, [extra])

    def handle_param_entry(self, payload, ss, es):
        if len(payload) < 2:
            self.putx(Ann.WARN, ss, es, [
                'Parameter frame is too short', 'Short parameter frame'])
            return
        field = payload[0]
        remaining = payload[1]
        is_first = len(payload) >= 4 and (payload[3] & 0x7f) <= 0x0f
        name = None
        if is_first:
            parent = payload[2]
            value_type = payload[3]
            hidden = bool(value_type & 0x80)
            raw = bytes(payload[4:])
            nul = raw.find(b'\x00')
            if nul > 0:
                name = raw[:nul].decode('latin-1')
            text = 'Parameter {}, "{}", type {}'.format(
                field, name if name else '<no name>', value_type & 0x7f)
            if parent:
                text += ', parent {}'.format(parent)
            if hidden:
                text += ', hidden'
            if remaining:
                text += ', {} more chunk(s)'.format(remaining)
            self.putx(Ann.ELRS, ss, es, [text])
        else:
            text = 'Parameter {}, chunk with {} more chunk(s), {} byte(s)'.format(
                field, remaining, max(0, len(payload) - 2))
            self.putx(Ann.ELRS, ss, es, [text])

    def handle_command(self, payload, ss, es):
        if not payload:
            return
        command = payload[0]
        sub = payload[1] if len(payload) > 1 else None
        if command == 0x10:
            name = 'RX command'
        elif command == 0x05:
            name = 'Model select'
        else:
            name = 'command 0x{:02x}'.format(command)
        text = name
        if command == 0x10 and sub is not None:
            if sub == 0x01:
                text += ': bind'
                self.putpy(ss, es, ['BIND', True])
            elif sub == 0x05:
                text += ': model select'
            else:
                text += ': subcommand 0x{:02x}'.format(sub)
        self.putx(Ann.ELRS, ss, es, [text, name])

    def handle_attitude(self, payload, ss, es):
        if len(payload) < 6:
            return
        pitch = int.from_bytes(bytes(payload[0:2]), byteorder='big', signed=True)
        roll = int.from_bytes(bytes(payload[2:4]), byteorder='big', signed=True)
        yaw = int.from_bytes(bytes(payload[4:6]), byteorder='big', signed=True)
        scale = 180.0 / 3.141592653589793 / 10000.0
        self.putx(Ann.FRAME, ss, es, [
            'Attitude: pitch {:.2f} deg, roll {:.2f} deg, yaw {:.2f} deg'.format(
                pitch * scale, roll * scale, yaw * scale)])

    def handle_gps(self, payload, ss, es):
        if len(payload) < 15:
            return
        lat = int.from_bytes(bytes(payload[0:4]), byteorder='big', signed=True)
        lon = int.from_bytes(bytes(payload[4:8]), byteorder='big', signed=True)
        speed = int.from_bytes(bytes(payload[8:10]), byteorder='big')
        heading = int.from_bytes(bytes(payload[10:12]), byteorder='big')
        altitude = int.from_bytes(bytes(payload[12:14]), byteorder='big')
        sats = payload[14]
        self.putx(Ann.FRAME, ss, es, [
            'GPS: {:.7f}, {:.7f}, {:.1f} km/h, heading {:.2f} deg, '
            '{} m, {} satellite(s)'.format(
                lat / 1e7, lon / 1e7, speed / 10.0, heading / 100.0,
                altitude - 1000, sats)])

    def handle_battery(self, payload, ss, es):
        if len(payload) < 7:
            return
        voltage = int.from_bytes(bytes(payload[0:2]), byteorder='big')
        current = int.from_bytes(bytes(payload[2:4]), byteorder='big')
        capacity = int.from_bytes(bytes(payload[4:7]), byteorder='big')
        remaining = payload[7] if len(payload) > 7 else None
        text = 'Battery: {:.1f} V, {:.1f} A, {} mAh'.format(
            voltage / 10.0, current / 10.0, capacity)
        if remaining is not None:
            text += ', {}%'.format(remaining)
        self.putx(Ann.FRAME, ss, es, [text])

    def handle_text(self, payload, ss, es, label):
        text = bytes(payload).split(b'\x00')[0].decode('latin-1')
        if text:
            self.putx(Ann.FRAME, ss, es, ['{}: {}'.format(label, text)])

    def handle_msp(self, ftype, payload, ss, es):
        # Behind the addresses the MSP frame carries a status byte, the
        # 16 bit MSP function and a 16 bit payload length.
        if len(payload) < 5:
            self.putx(Ann.FRAME, ss, es, [
                'MSP frame with {} byte(s)'.format(len(payload))])
            return
        status = payload[0]
        function = (payload[1] << 8) | payload[2]
        length = (payload[3] << 8) | payload[4]
        kind = {0x7a: 'request', 0x7b: 'response', 0x7c: 'write'}.get(
            ftype, 'frame')
        self.putx(Ann.ELRS, ss, es, [
            'MSP {}: function 0x{:04x}, status 0x{:02x}, {} byte(s)'.format(
                kind, function, status, length),
            'MSP {} 0x{:04x}'.format(kind, function)])
