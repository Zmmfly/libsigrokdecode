##
## This file is part of the libsigrokdecode project.
##
## Copyright (C) 2023 James Cordell <james@cordell.org.uk>

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

# Decodes the RC channels and the link statistics field by field, and the
# GPS, vario, battery, barometric altitude, attitude and flight mode
# telemetry frames on one line each; other frame types show the size of
# their payload. The CRC of every frame is checked.
#
# c_decoders/crsf_c.c is a port of this decoder and has to produce the
# same annotations, tests/testdata/crsf_c holds the reference cases.
#
# TODO: More packet types need to be implemented, and the decoder needs
# testing with different CRSF modes.

import math
import sigrokdecode as srd

# Uplink TX power levels of the link statistics frame, in mW.
TX_POWERS = (0, 10, 25, 100, 500, 1000, 2000, 250, 50)

class Decoder(srd.Decoder):
    api_version = 3
    id = 'crsf'
    name = "CRSF"
    longname = "Crossfire rc protocol"
    desc = 'A protocol for radio control systems.'
    license = 'gplv2+'
    inputs = ['uart']
    outputs = []
    tags = ['radio','control', 'RC']
    annotations = (
        ('text-verbose', 'Human-readable text (verbose)'),
        ('text-error', 'Human-readable Error text'),
    )
    annotation_rows = (
        ('normal', 'Normal', (0, 1)),
    )

    sync_bytes = {
            0xEE: ['To Transmitter Module', 'To TX Module'],
            0xEA: ['To Handset', 'To HS'],
            0xC8: ['To Flight Controller', 'To FC' ],
            0xEC: ['To Receiver', 'To RX']
    }

    packet_type = { 
    0x16: ['CRSF_FRAMETYPE_RC_CHANNELS_PACKED'],  #16 channels of 11 bit each is 176 bits
    0x28: ['CRSF_FRAMETYPE_DEVICE_PING'],
    0x29: ['CRSF_FRAMETYPE_DEVICE_INFO'],
    0x2B: ['CRSF_FRAMETYPE_PARAMETER_SETTINGS_ENTRY'],
    0x2C: ['CRSF_FRAMETYPE_PARAMETER_READ'],
    0x2D: ['CRSF_FRAMETYPE_PARAMETER_WRITE'],
    0x32: ['CRSF_FRAMETYPE_COMMAND'],
    0x02: ['GPS'],
    0x07: ['Vario'],
    0x08: ['Battery sensor'],
    0x09: ['Baro altitude'],
    0x0B: ['Heartbeat'],
    0x10: ['OpenTX sync'],
    0x14: ['LINK_STATISTICS'],
    0x1E: ['Attitude'],
    0x21: ['Flight mode'],
    0x2A: ['Request settings'],
    0x3A: ['Radio'],
    }

    def __init__(self):
        self.reset()

    def reset(self):
        self.state = 'SYNC'
        self.len_byte = 0
        # (value, ss, es, data bits) of the type, payload and CRC bytes
        self.body = []

    @staticmethod
    def crc8(data):
        # CRC-8/DVB-S2, polynomial 0xD5, over the type and the payload.
        crc = 0
        for byte in data:
            crc ^= byte
            for _ in range(8):
                crc = ((crc << 1) ^ 0xD5) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
        return crc

    def start(self):
        self.out_ann = self.register(srd.OUTPUT_ANN)

    def putt(self, ss, es, cls, texts):
        self.put(ss, es, self.out_ann, [cls, texts])

    def put_unknown(self, ss, es, serial_byte):
        self.putt(ss, es, 0, ['Unknown packet type' + str(serial_byte)])

    def decode(self, ss, es, data):
        ptype, rxtx, pdata = data

        # For now, ignore all UART packets except the actual data packets.
        if ptype != 'DATA':
            return
        serial_byte, bits = pdata

        if self.state == 'SYNC': # Start of packet sync byte
            ann = self.sync_bytes.get(serial_byte)
            if ann:
                self.putt(ss, es, 0, ann)
                self.state = 'LEN'
            else:
                self.put_unknown(ss, es, serial_byte)
            return

        if self.state == 'LEN':
            # The length counts the type, the payload and the CRC byte.
            if 2 <= serial_byte <= 62: # Within range
                self.putt(ss, es, 0, ['Num of bytes succeeding: ' + str(serial_byte - 2)])
                self.len_byte = serial_byte
                self.state = 'TYPE'
            else:
                self.put_unknown(ss, es, serial_byte)
                self.reset()
            return

        if self.state == 'TYPE':
            frame_type = self.packet_type.get(serial_byte)
            if not frame_type:
                self.put_unknown(ss, es, serial_byte)
                self.reset()
                return
            self.putt(ss, es, 0, frame_type)
            self.state = 'BODY'

        # The uart decoder reuses its list of data bits, keep a copy.
        self.body.append((serial_byte, ss, es, list(bits)))
        if len(self.body) == self.len_byte:
            self.handle_frame()
            self.reset()

    def handle_frame(self):
        frame_type = self.body[0][0]
        payload = self.body[1:-1]
        values = [item[0] for item in payload]
        if frame_type == 0x16 and len(payload) >= 22:
            self.handle_channels(payload)
        elif frame_type == 0x14 and len(payload) >= 10:
            self.handle_link_statistics(payload, values)
        elif payload:
            # One line over the payload bytes.
            text = self.telemetry_text(frame_type, values)
            if text is None:
                text = '{} ({} bytes payload)'.format(
                    self.packet_type[frame_type][0], len(payload))
            self.putt(payload[0][1], payload[-1][2], 0, [text])
        self.put_checksum()

    def put_checksum(self):
        crc_byte, ss, es = self.body[-1][:3]
        crc = self.crc8([item[0] for item in self.body[:-1]])
        if crc == crc_byte:
            self.putt(ss, es, 0, ['Checksum crc8 poly 0xD5.'])
        else:
            self.putt(ss, es, 1, [
                'Checksum 0x{:02x}, expected 0x{:02x}'.format(crc_byte, crc),
                'Checksum error'])

    def handle_channels(self, payload):
        # 16 channels of 11 bits, LSB first. Each channel spans its bits.
        bits = [bit for item in payload for bit in item[3]]
        for chan in range(16):
            chan_bits = bits[chan * 11:chan * 11 + 11]
            value = sum(bit[0] << idx for idx, bit in enumerate(chan_bits))  # little edian 'RC' value
            self.putt(chan_bits[0][1], chan_bits[-1][2], 0,
                      ['Chan:' + str(chan) + ' Value:' + str(value)])

    def handle_link_statistics(self, payload, values):
        def signed(value):
            return value - 256 if value > 127 else value

        power = values[6]
        if power < len(TX_POWERS):
            power = '{} mW'.format(TX_POWERS[power])
        else:
            power = 'level {}'.format(power)
        texts = (
            'Uplink RSSI 1: -' + str(values[0]) + 'dB',
            'Uplink RSSI 2: -' + str(values[1]) + 'dB',
            'Uplink Link Quality: ' + str(values[2]),
            'Uplink SNR: ' + str(signed(values[3])) + 'dB',
            'Active Antenna: ' + str(values[4]),
            'RF Mode: ' + str(values[5]),
            'Uplink TX Power: ' + power,
            'Downlink RSSI: -' + str(values[7]) + 'dB',
            'Downlink Link Quality: ' + str(values[8]),
            'Downlink SNR: ' + str(signed(values[9])) + 'dB',
        )
        # Each field spans its byte.
        for item, text in zip(payload, texts):
            self.putt(item[1], item[2], 0, [text])

    @staticmethod
    def telemetry_text(frame_type, p):
        # Multi-byte fields are big-endian. None when the payload does not
        # fit the layout of the frame type.
        def be(data, signed=False):
            return int.from_bytes(bytes(data), byteorder='big', signed=signed)

        if frame_type == 0x02 and len(p) >= 15:
            # deg * 1e7, km/h * 10, deg * 100, m + 1000
            return ('GPS: {:.7f}, {:.7f}, {:.1f} km/h, heading {:.2f} deg, '
                    '{} m, {} satellite(s)').format(
                be(p[0:4], True) / 1e7, be(p[4:8], True) / 1e7, be(p[8:10]) / 10.0,
                be(p[10:12]) / 100.0, be(p[12:14]) - 1000, p[14])
        if frame_type == 0x07 and len(p) >= 2:
            # cm/s
            return 'Vario: {:.2f} m/s'.format(be(p[0:2], True) / 100.0)
        if frame_type == 0x08 and len(p) >= 7:
            # 0.1 V, 0.1 A, mAh, %
            text = 'Battery: {:.1f} V, {:.1f} A, {} mAh'.format(
                be(p[0:2]) / 10.0, be(p[2:4]) / 10.0, be(p[4:7]))
            if len(p) >= 8:
                text += ', {}%'.format(p[7])
            return text
        if frame_type == 0x09 and len(p) >= 2:
            altitude = be(p[0:2])
            if altitude & 0x8000:
                # m, from 2276.8 m on
                text = 'Baro altitude: {} m'.format(altitude & 0x7FFF)
            else:
                # dm + 10000
                text = 'Baro altitude: {:.1f} m'.format((altitude - 10000) / 10.0)
            if len(p) == 3:
                # Packed vertical speed, sign * (e^(|v| * 0.026) - 1) m/s.
                packed = p[2] - 256 if p[2] > 127 else p[2]
                speed = math.exp(abs(packed) * 0.026) - 1.0
                text += ', vertical speed {:.2f} m/s'.format(-speed if packed < 0 else speed)
            return text
        if frame_type == 0x1E and len(p) >= 6:
            # rad * 10000
            return 'Attitude: pitch {:.4f} rad, roll {:.4f} rad, yaw {:.4f} rad'.format(
                be(p[0:2], True) / 10000.0, be(p[2:4], True) / 10000.0,
                be(p[4:6], True) / 10000.0)
        if frame_type == 0x21:
            # Null-terminated string
            name = bytes(p).split(b'\x00')[0].decode('latin-1')
            if name:
                return 'Flight mode: ' + name
        return None
