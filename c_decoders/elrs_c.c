/*
 * This file is part of the libsigrokdecode project.
 *
 * Copyright (C) 2026 Zmmfly <shenqhy@qq.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, see <http://www.gnu.org/licenses/>.
 */

/*
 * ExpressLRS (CRSF serial link) decoder, native port of
 * decoders/elrs/pd.py.
 *
 * Stack it on top of the UART(C) decoder: it consumes the "FRAME"
 * output of uart_c, whose fields are (rxtx, value, valid). The
 * annotation classes, sample ranges and texts match the Python decoder,
 * tests/testdata/elrs_c holds the reference captures. See the Python
 * decoder for the description of the frames.
 *
 * Unlike the Python decoder, the RX and the TX line of the UART are
 * assembled separately, so a capture of both directions does not mix
 * the bytes of two frames. With a single line the output is the same.
 *
 * C protocol output (c_proto(), for stacked C decoders):
 *  - "FRAME":    C_U8(address), C_U8(type), C_I32(destination, -1 if none),
 *                C_I32(origin, -1 if none), C_U8(crc_ok), C_BYTES(payload)
 *  - "CHANNELS": 16 x C_U16(channel value), C_I32(armed, -1 if unknown)
 *  - "LINK":     C_U8(uplink RSSI 1), C_U8(uplink RSSI 2), C_U8(uplink LQ),
 *                C_I8(uplink SNR), C_U8(antenna), C_U8(RF mode),
 *                C_U8(TX power), C_U8(downlink RSSI), C_U8(downlink LQ),
 *                C_I8(downlink SNR)
 *  - "STATUS":   C_U8(bad packets), C_U16(good packets), C_U8(flags),
 *                C_STR(message)
 *  - "BIND":     C_U8(1)
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <glib.h>
#include "libsigrokdecode.h"

enum {
    ANN_SYNC = 0,
    ANN_CRC,
    ANN_FRAME,
    ANN_CHANNELS,
    ANN_LINK,
    ANN_ELRS,
    ANN_WARN,
    NUM_ANN,
};

enum {
    STATE_SYNC = 0,
    STATE_LEN,
    STATE_BODY,
};

#define CRC8_POLY 0xd5
/* The length byte counts the type, the payload and the checksum. */
#define LENGTH_MIN 2
#define LENGTH_MAX 62
#define FRAME_MAX (2 + LENGTH_MAX)
/* Frame types in this range carry destination and origin addresses. */
#define EXT_TYPE_MIN 0x28
#define EXT_TYPE_MAX 0x96

/* Channel values in CRSF are 11 bit, and the usual endpoints are these. */
#define CHANNEL_MIN 172
#define CHANNEL_MAX 1811
#define USEC_MIN 988
#define USEC_MAX 2012

struct elrs_name {
    unsigned value;
    const char *name;
};

/* Addresses which CRSF and ELRS assign to devices. */
static const struct elrs_name elrs_addresses[] = {
    {0x00, "Broadcast"},
    {0x0e, "Cloud"},
    {0x10, "USB"},
    {0x12, "Bluetooth/WiFi module"},
    {0x13, "WiFi receiver"},
    {0x14, "Video receiver"},
    {0x80, "OSD / TBS Core PNP Pro"},
    {0x8a, "Reserved"},
    {0x90, "ESC 1"},
    {0x91, "ESC 2"},
    {0x92, "ESC 3"},
    {0x93, "ESC 4"},
    {0x94, "ESC 5"},
    {0x95, "ESC 6"},
    {0x96, "ESC 7"},
    {0x97, "ESC 8"},
    {0xc0, "Current sensor"},
    {0xc2, "GPS"},
    {0xc4, "Blackbox"},
    {0xc8, "Flight controller"},
    {0xca, "Reserved"},
    {0xcc, "Race tag"},
    {0xce, "VTX"},
    {0xea, "Radio transmitter"},
    {0xec, "CRSF receiver"},
    {0xee, "CRSF transmitter"},
    {0xef, "ELRS LUA"},
};

static const struct elrs_name elrs_frame_types[] = {
    {0x02, "GPS"},
    {0x03, "GPS time"},
    {0x07, "Vario"},
    {0x08, "Battery sensor"},
    {0x09, "Baro altitude"},
    {0x0a, "Airspeed"},
    {0x0b, "Heartbeat"},
    {0x0c, "RPM"},
    {0x0d, "Temperature"},
    {0x0e, "Cells"},
    {0x14, "Link statistics"},
    {0x16, "RC channels"},
    {0x1e, "Attitude"},
    {0x21, "Flight mode"},
    {0x28, "Device ping"},
    {0x29, "Device info"},
    {0x2b, "Parameter settings entry"},
    {0x2c, "Parameter read"},
    {0x2d, "Parameter write"},
    {0x2e, "ELRS status"},
    {0x32, "Command"},
    {0x3a, "Handset"},
    {0x78, "KISS request"},
    {0x79, "KISS response"},
    {0x7a, "MSP request"},
    {0x7b, "MSP response"},
    {0x7c, "MSP write"},
    {0x80, "ArduPilot response"},
};

/* RF modes as used in the link statistics frame. */
static const struct elrs_name elrs_rf_modes[] = {
    {0, "900 MHz 25 Hz"},
    {1, "900 MHz 50 Hz"},
    {2, "900 MHz 100 Hz"},
    {3, "900 MHz 100 Hz 8ch"},
    {4, "900 MHz 150 Hz"},
    {5, "900 MHz 200 Hz"},
    {6, "900 MHz 200 Hz 8ch"},
    {7, "900 MHz 250 Hz"},
    {8, "900 MHz 333 Hz 8ch"},
    {9, "900 MHz 500 Hz"},
    {10, "900 MHz 50 Hz DVDA"},
    {11, "FSK 900 MHz 1000 Hz 8ch"},
    {20, "2.4 GHz 25 Hz"},
    {21, "2.4 GHz 50 Hz"},
    {22, "2.4 GHz 100 Hz"},
    {23, "2.4 GHz 100 Hz 8ch"},
    {24, "2.4 GHz 150 Hz"},
    {25, "2.4 GHz 200 Hz"},
    {26, "2.4 GHz 200 Hz 8ch"},
    {27, "2.4 GHz 250 Hz"},
    {28, "2.4 GHz 333 Hz 8ch"},
    {29, "2.4 GHz 500 Hz"},
    {30, "FLRC 2.4 GHz 250 Hz DVDA"},
    {31, "FLRC 2.4 GHz 500 Hz DVDA"},
    {32, "FLRC 2.4 GHz 500 Hz"},
    {33, "FLRC 2.4 GHz 1000 Hz"},
    {34, "FSK 2.4 GHz 250 Hz DVDA"},
    {35, "FSK 2.4 GHz 500 Hz DVDA"},
    {36, "FSK 2.4 GHz 1000 Hz"},
    {100, "2.4 GHz dual 100 Hz 8ch"},
    {101, "2.4 GHz dual 150 Hz"},
};

static const struct elrs_name elrs_tx_powers[] = {
    {0, "0 mW"},
    {1, "10 mW"},
    {2, "25 mW"},
    {3, "100 mW"},
    {4, "500 mW"},
    {5, "1000 mW"},
    {6, "2000 mW"},
    {7, "250 mW"},
    {8, "50 mW"},
};

/* Warning and status bits of the ELRS status frame. */
static const struct elrs_name elrs_status_flags[] = {
    {0x01, "connected"},
    {0x02, "status 1"},
    {0x04, "model mismatch"},
    {0x08, "armed"},
    {0x10, "warning 1"},
    {0x20, "not while connected"},
    {0x40, "baud rate too low"},
    {0x80, "critical warning 2"},
};

#define ELRS_COUNT(table) ((int)(sizeof(table) / sizeof((table)[0])))

/* Frame assembly state of one UART direction. */
typedef struct {
    int state;
    int len;
    int body_len;
    uint8_t value[FRAME_MAX];
    uint64_t ss[FRAME_MAX];
    uint64_t es[FRAME_MAX];
} elrs_link;

typedef struct {
    int out_ann;
    int out_proto;
    int opt_crc;
    int opt_channels;
    int opt_usec;
    elrs_link link[2];
} elrs_state;

static struct srd_decoder_option elrs_options[] = {
    {"crc", NULL, "Decode frames with a wrong checksum", NULL, NULL},
    {"channels", NULL, "Annotate each channel separately", NULL, NULL},
    {"usec", NULL, "Show channel values in microseconds", NULL, NULL},
};
#define NUM_OPTIONS ELRS_COUNT(elrs_options)

static const char *elrs_ann_labels[][3] = {
    {"", "address", "Address"},
    {"", "crc", "Checksum"},
    {"", "frame", "Frame"},
    {"", "channels", "Channels"},
    {"", "link", "Link statistics"},
    {"", "elrs", "ELRS information"},
    {"", "warn", "Warning"},
};

static const int elrs_row_framing[] = {ANN_SYNC, ANN_CRC};
static const int elrs_row_frames[] = {ANN_FRAME};
static const int elrs_row_channels[] = {ANN_CHANNELS};
static const int elrs_row_link[] = {ANN_LINK};
static const int elrs_row_elrs[] = {ANN_ELRS};
static const int elrs_row_warnings[] = {ANN_WARN};

static const struct srd_c_ann_row elrs_ann_rows[] = {
    {"framing", "Framing", elrs_row_framing, 2},
    {"frames", "Frames", elrs_row_frames, 1},
    {"channels", "Channels", elrs_row_channels, 1},
    {"link", "Link", elrs_row_link, 1},
    {"elrs", "ELRS", elrs_row_elrs, 1},
    {"warnings", "Warnings", elrs_row_warnings, 1},
};

static const char *elrs_inputs[] = {"uart", NULL};
static const char *elrs_outputs[] = {"elrs", NULL};
static const char *elrs_tags[] = {"Radio/control/RC", NULL};

/*-- Helpers ----------------------------------------------------------*/

static void elrs_put(struct srd_decoder_inst *di, const elrs_state *s,
                     uint64_t ss, uint64_t es, int cls,
                     const char *t0, const char *t1, const char *t2)
{
    const char *texts[4] = {t0, t1, t2, NULL};
    struct srd_c_annotation ann = {cls, 0, (char **)texts, "", 0};

    c_decoder_put(di, ss, es, s->out_ann, &ann);
}

static const char *elrs_lookup(const struct elrs_name *table, int count,
                               unsigned value)
{
    for (int idx = 0; idx < count; idx++) {
        if (table[idx].value == value)
            return table[idx].name;
    }
    return NULL;
}

static const char *elrs_address_name(unsigned value, char *buf, size_t size)
{
    const char *name = elrs_lookup(elrs_addresses, ELRS_COUNT(elrs_addresses), value);

    if (name)
        return name;
    snprintf(buf, size, "Unknown address 0x%02x", value);
    return buf;
}

static const char *elrs_type_name(unsigned value, char *buf, size_t size)
{
    const char *name = elrs_lookup(elrs_frame_types, ELRS_COUNT(elrs_frame_types), value);

    if (name)
        return name;
    snprintf(buf, size, "Unknown type 0x%02x", value);
    return buf;
}

static uint8_t elrs_crc8(const uint8_t *data, int len)
{
    uint8_t crc = 0;

    for (int idx = 0; idx < len; idx++) {
        crc ^= data[idx];
        for (int bit = 0; bit < 8; bit++)
            crc = (crc & 0x80) ? (uint8_t)((crc << 1) ^ CRC8_POLY) : (uint8_t)(crc << 1);
    }
    return crc;
}

static int64_t elrs_field_int(const c_field *field)
{
    switch (field->type) {
    case C_FIELD_U8: return field->u8;
    case C_FIELD_U16: return field->u16;
    case C_FIELD_U32: return field->u32;
    case C_FIELD_U64: return (int64_t)field->u64;
    case C_FIELD_I8: return field->i8;
    case C_FIELD_I16: return field->i16;
    case C_FIELD_I32: return field->i32;
    case C_FIELD_I64: return field->i64;
    case C_FIELD_F64: return (int64_t)field->f64;
    default: return 0;
    }
}

/* Text up to the first NUL byte, Latin-1 decoded into UTF-8. */
static void elrs_latin1(char *out, size_t size, const uint8_t *data, int len)
{
    size_t pos = 0;

    for (int idx = 0; idx < len && data[idx]; idx++) {
        uint8_t ch = data[idx];
        if (ch < 0x80) {
            if (pos + 1 >= size)
                break;
            out[pos++] = (char)ch;
        } else {
            if (pos + 2 >= size)
                break;
            out[pos++] = (char)(0xc0 | (ch >> 6));
            out[pos++] = (char)(0x80 | (ch & 0x3f));
        }
    }
    out[pos] = '\0';
}

static int elrs_find_nul(const uint8_t *data, int len)
{
    for (int idx = 0; idx < len; idx++) {
        if (!data[idx])
            return idx;
    }
    return -1;
}

static uint32_t elrs_be(const uint8_t *data, int len)
{
    uint32_t value = 0;

    for (int idx = 0; idx < len; idx++)
        value = (value << 8) | data[idx];
    return value;
}

/* Linear scaling between the CRSF channel and the pulse width range. */
static int elrs_channel_to_usec(unsigned value)
{
    int span = CHANNEL_MAX - CHANNEL_MIN;
    double usec = USEC_MIN +
        (double)(((int)value - CHANNEL_MIN) * (USEC_MAX - USEC_MIN)) / (double)span;

    /* Round half to even, like Python's round(). */
    return (int)rint(usec);
}

/*-- Payload handlers -------------------------------------------------*/

static void elrs_handle_channels(struct srd_decoder_inst *di, elrs_state *s,
                                 const uint8_t *payload, int len,
                                 uint64_t ss, uint64_t es)
{
    char text[192], t0[64], t1[64];
    unsigned ch[16];
    int armed = -1;

    if (len < 22) {
        snprintf(t0, sizeof(t0), "RC channels frame is too short (%d byte(s))", len);
        elrs_put(di, s, ss, es, ANN_WARN, t0, "Short channel frame", NULL);
        return;
    }
    for (int idx = 0; idx < 16; idx++) {
        unsigned value = 0;
        for (int bit = 0; bit < 11; bit++) {
            int pos = idx * 11 + bit;
            if ((payload[pos >> 3] >> (pos & 7)) & 1)
                value |= 1u << bit;
        }
        ch[idx] = value;
    }
    if (len > 22) {
        /* ELRS appends the armed state behind the channel data. */
        armed = payload[22] & 0x01;
        int arm_ch5 = (payload[22] & 0x02) != 0;
        snprintf(t0, sizeof(t0), "Armed: %s, arm via CH5: %s",
                 armed ? "yes" : "no", arm_ch5 ? "yes" : "no");
        snprintf(t1, sizeof(t1), "Armed %s, CH5 arm %s",
                 armed ? "yes" : "no", arm_ch5 ? "yes" : "no");
        elrs_put(di, s, ss, es, ANN_ELRS, t0, t1, NULL);
    }
    size_t pos = 0;
    for (int idx = 0; idx < 16; idx++) {
        if (s->opt_usec)
            pos += (size_t)snprintf(text + pos, sizeof(text) - pos, "%s%dus",
                                    idx ? " " : "", elrs_channel_to_usec(ch[idx]));
        else
            pos += (size_t)snprintf(text + pos, sizeof(text) - pos, "%s%u",
                                    idx ? " " : "", ch[idx]);
    }
    elrs_put(di, s, ss, es, ANN_CHANNELS, text, NULL, NULL);
    if (s->opt_channels) {
        for (int idx = 0; idx < 16; idx++) {
            if (s->opt_usec)
                snprintf(t0, sizeof(t0), "CH%d: %u (%d us)", idx + 1, ch[idx],
                         elrs_channel_to_usec(ch[idx]));
            else
                snprintf(t0, sizeof(t0), "CH%d: %u", idx + 1, ch[idx]);
            elrs_put(di, s, ss, es, ANN_CHANNELS, t0, NULL, NULL);
        }
    }
    c_proto(di, ss, es, s->out_proto, "CHANNELS",
            C_U16(ch[0]), C_U16(ch[1]), C_U16(ch[2]), C_U16(ch[3]),
            C_U16(ch[4]), C_U16(ch[5]), C_U16(ch[6]), C_U16(ch[7]),
            C_U16(ch[8]), C_U16(ch[9]), C_U16(ch[10]), C_U16(ch[11]),
            C_U16(ch[12]), C_U16(ch[13]), C_U16(ch[14]), C_U16(ch[15]),
            C_I32(armed), C_END);
}

static void elrs_handle_link_stats(struct srd_decoder_inst *di, elrs_state *s,
                                   const uint8_t *payload, int len,
                                   uint64_t ss, uint64_t es)
{
    char t0[128], t1[96], mode_buf[24], power_buf[24];

    if (len < 10) {
        snprintf(t0, sizeof(t0), "Link statistics frame is too short (%d byte(s))", len);
        elrs_put(di, s, ss, es, ANN_WARN, t0, "Short link frame", NULL);
        return;
    }
    int up_rssi1 = payload[0];
    int up_rssi2 = payload[1];
    int up_lq = payload[2];
    int up_snr = (int8_t)payload[3];
    int antenna = payload[4];
    int rf_mode = payload[5];
    int tx_power = payload[6];
    int dn_rssi = payload[7];
    int dn_lq = payload[8];
    int dn_snr = (int8_t)payload[9];
    const char *mode_name = elrs_lookup(elrs_rf_modes, ELRS_COUNT(elrs_rf_modes), (unsigned)rf_mode);
    const char *power_name = elrs_lookup(elrs_tx_powers, ELRS_COUNT(elrs_tx_powers), (unsigned)tx_power);

    if (!mode_name) {
        snprintf(mode_buf, sizeof(mode_buf), "mode %d", rf_mode);
        mode_name = mode_buf;
    }
    if (!power_name) {
        snprintf(power_buf, sizeof(power_buf), "level %d", tx_power);
        power_name = power_buf;
    }
    snprintf(t0, sizeof(t0), "Uplink: RSSI1 %d dBm, RSSI2 %d dBm, LQ %d%%, SNR %d dB",
             -up_rssi1, -up_rssi2, up_lq, up_snr);
    snprintf(t1, sizeof(t1), "Up %d / %d dBm, %d%%, %d dB",
             -up_rssi1, -up_rssi2, up_lq, up_snr);
    elrs_put(di, s, ss, es, ANN_LINK, t0, t1, NULL);
    snprintf(t0, sizeof(t0), "Mode %s, antenna %d, TX power %s",
             mode_name, antenna + 1, power_name);
    snprintf(t1, sizeof(t1), "%s, ant %d, %s", mode_name, antenna + 1, power_name);
    elrs_put(di, s, ss, es, ANN_LINK, t0, t1, NULL);
    snprintf(t0, sizeof(t0), "Downlink: RSSI %d dBm, LQ %d%%, SNR %d dB",
             -dn_rssi, dn_lq, dn_snr);
    snprintf(t1, sizeof(t1), "Down %d dBm, %d%%, %d dB", -dn_rssi, dn_lq, dn_snr);
    elrs_put(di, s, ss, es, ANN_LINK, t0, t1, NULL);
    c_proto(di, ss, es, s->out_proto, "LINK", C_U8(up_rssi1), C_U8(up_rssi2),
            C_U8(up_lq), C_I8(up_snr), C_U8(antenna), C_U8(rf_mode),
            C_U8(tx_power), C_U8(dn_rssi), C_U8(dn_lq), C_I8(dn_snr), C_END);
}

static void elrs_handle_status(struct srd_decoder_inst *di, elrs_state *s,
                               const uint8_t *payload, int len,
                               uint64_t ss, uint64_t es)
{
    char t0[192], t1[128], names[160], message[128];

    if (len < 4) {
        snprintf(t0, sizeof(t0), "ELRS status frame is too short (%d byte(s))", len);
        elrs_put(di, s, ss, es, ANN_WARN, t0, "Short status frame", NULL);
        return;
    }
    int bad = payload[0];
    int good = (payload[1] << 8) | payload[2];
    int flags = payload[3];
    elrs_latin1(message, sizeof(message), payload + 4, len - 4);

    size_t pos = 0;
    names[0] = '\0';
    for (int idx = 0; idx < ELRS_COUNT(elrs_status_flags); idx++) {
        if (!(flags & (int)elrs_status_flags[idx].value))
            continue;
        pos += (size_t)snprintf(names + pos, sizeof(names) - pos, "%s%s",
                                pos ? ", " : "", elrs_status_flags[idx].name);
    }
    snprintf(t0, sizeof(t0), "ELRS status: %d bad, %d good packet(s), flags 0x%02x",
             bad, good, flags);
    snprintf(t1, sizeof(t1), "%d bad / %d good, 0x%02x", bad, good, flags);
    elrs_put(di, s, ss, es, ANN_ELRS, t0, t1, NULL);
    snprintf(t0, sizeof(t0), "Status flags: %s", pos ? names : "none");
    elrs_put(di, s, ss, es, ANN_ELRS, t0, pos ? names : "no flags", NULL);
    if (message[0]) {
        snprintf(t0, sizeof(t0), "Message: %s", message);
        elrs_put(di, s, ss, es, ANN_ELRS, t0, message, NULL);
    }
    c_proto(di, ss, es, s->out_proto, "STATUS", C_U8(bad), C_U16(good),
            C_U8(flags), C_STR(message), C_END);
}

static void elrs_handle_device_info(struct srd_decoder_inst *di, elrs_state *s,
                                    const uint8_t *payload, int len,
                                    uint64_t ss, uint64_t es)
{
    char name[128], version[16], t0[192], t1[160];
    int nul = elrs_find_nul(payload, len);

    if (nul < 0) {
        elrs_put(di, s, ss, es, ANN_WARN, "Device info without a device name",
                 "Short device info", NULL);
        return;
    }
    elrs_latin1(name, sizeof(name), payload, nul);
    const uint8_t *fields = payload + nul + 1;
    int nfields = len - nul - 1;
    if (nfields < 13) {
        elrs_put(di, s, ss, es, ANN_WARN, "Truncated device info frame",
                 "Short device info", NULL);
        return;
    }
    uint32_t serial = elrs_be(fields, 4);
    uint32_t hardware = elrs_be(fields + 4, 4);
    uint32_t software = elrs_be(fields + 8, 4);
    int count = fields[12];
    snprintf(version, sizeof(version), "%u.%u.%u", (software >> 24) & 0xff,
             (software >> 16) & 0xff, (software >> 8) & 0xff);
    snprintf(t0, sizeof(t0), "Device \"%s\", version %s, serial 0x%08x",
             name, version, serial);
    snprintf(t1, sizeof(t1), "\"%s\" %s", name, version);
    elrs_put(di, s, ss, es, ANN_ELRS, t0, t1, NULL);
    int pos = snprintf(t0, sizeof(t0), "Hardware 0x%08x, %d parameter(s)", hardware, count);
    if (nfields > 13)
        snprintf(t0 + pos, sizeof(t0) - (size_t)pos, ", parameter version %d", fields[13]);
    elrs_put(di, s, ss, es, ANN_ELRS, t0, NULL, NULL);
}

static void elrs_handle_param_entry(struct srd_decoder_inst *di, elrs_state *s,
                                    const uint8_t *payload, int len,
                                    uint64_t ss, uint64_t es)
{
    char name[128], text[224];

    if (len < 2) {
        elrs_put(di, s, ss, es, ANN_WARN, "Parameter frame is too short",
                 "Short parameter frame", NULL);
        return;
    }
    int field = payload[0];
    int remaining = payload[1];
    if (len >= 4 && (payload[3] & 0x7f) <= 0x0f) {
        int parent = payload[2];
        int value_type = payload[3];
        int nul = elrs_find_nul(payload + 4, len - 4);
        if (nul > 0)
            elrs_latin1(name, sizeof(name), payload + 4, nul);
        int pos = snprintf(text, sizeof(text), "Parameter %d, \"%s\", type %d",
                           field, nul > 0 ? name : "<no name>", value_type & 0x7f);
        if (parent)
            pos += snprintf(text + pos, sizeof(text) - (size_t)pos, ", parent %d", parent);
        if (value_type & 0x80)
            pos += snprintf(text + pos, sizeof(text) - (size_t)pos, ", hidden");
        if (remaining)
            snprintf(text + pos, sizeof(text) - (size_t)pos, ", %d more chunk(s)", remaining);
    } else {
        snprintf(text, sizeof(text), "Parameter %d, chunk with %d more chunk(s), %d byte(s)",
                 field, remaining, len - 2);
    }
    elrs_put(di, s, ss, es, ANN_ELRS, text, NULL, NULL);
}

static void elrs_handle_command(struct srd_decoder_inst *di, elrs_state *s,
                                const uint8_t *payload, int len,
                                uint64_t ss, uint64_t es)
{
    char name[32], text[64];

    if (len < 1)
        return;
    int command = payload[0];
    if (command == 0x10)
        snprintf(name, sizeof(name), "RX command");
    else if (command == 0x05)
        snprintf(name, sizeof(name), "Model select");
    else
        snprintf(name, sizeof(name), "command 0x%02x", command);
    if (command == 0x10 && len > 1) {
        int sub = payload[1];
        if (sub == 0x01) {
            snprintf(text, sizeof(text), "%s: bind", name);
            c_proto(di, ss, es, s->out_proto, "BIND", C_U8(1), C_END);
        } else if (sub == 0x05) {
            snprintf(text, sizeof(text), "%s: model select", name);
        } else {
            snprintf(text, sizeof(text), "%s: subcommand 0x%02x", name, sub);
        }
    } else {
        snprintf(text, sizeof(text), "%s", name);
    }
    elrs_put(di, s, ss, es, ANN_ELRS, text, name, NULL);
}

static void elrs_handle_attitude(struct srd_decoder_inst *di, elrs_state *s,
                                 const uint8_t *payload, int len,
                                 uint64_t ss, uint64_t es)
{
    char text[128];

    if (len < 6)
        return;
    int pitch = (int16_t)elrs_be(payload, 2);
    int roll = (int16_t)elrs_be(payload + 2, 2);
    int yaw = (int16_t)elrs_be(payload + 4, 2);
    double scale = 180.0 / 3.141592653589793 / 10000.0;
    snprintf(text, sizeof(text), "Attitude: pitch %.2f deg, roll %.2f deg, yaw %.2f deg",
             pitch * scale, roll * scale, yaw * scale);
    elrs_put(di, s, ss, es, ANN_FRAME, text, NULL, NULL);
}

static void elrs_handle_gps(struct srd_decoder_inst *di, elrs_state *s,
                            const uint8_t *payload, int len,
                            uint64_t ss, uint64_t es)
{
    char text[192];

    if (len < 15)
        return;
    int32_t lat = (int32_t)elrs_be(payload, 4);
    int32_t lon = (int32_t)elrs_be(payload + 4, 4);
    unsigned speed = elrs_be(payload + 8, 2);
    unsigned heading = elrs_be(payload + 10, 2);
    int altitude = (int)elrs_be(payload + 12, 2);
    int sats = payload[14];
    snprintf(text, sizeof(text),
             "GPS: %.7f, %.7f, %.1f km/h, heading %.2f deg, %d m, %d satellite(s)",
             lat / 1e7, lon / 1e7, speed / 10.0, heading / 100.0, altitude - 1000, sats);
    elrs_put(di, s, ss, es, ANN_FRAME, text, NULL, NULL);
}

static void elrs_handle_battery(struct srd_decoder_inst *di, elrs_state *s,
                                const uint8_t *payload, int len,
                                uint64_t ss, uint64_t es)
{
    char text[128];

    if (len < 7)
        return;
    unsigned voltage = elrs_be(payload, 2);
    unsigned current = elrs_be(payload + 2, 2);
    unsigned capacity = elrs_be(payload + 4, 3);
    int pos = snprintf(text, sizeof(text), "Battery: %.1f V, %.1f A, %u mAh",
                       voltage / 10.0, current / 10.0, capacity);
    if (len > 7)
        snprintf(text + pos, sizeof(text) - (size_t)pos, ", %d%%", payload[7]);
    elrs_put(di, s, ss, es, ANN_FRAME, text, NULL, NULL);
}

/* The vertical speed in cm/s. */
static void elrs_handle_vario(struct srd_decoder_inst *di, elrs_state *s,
                              const uint8_t *payload, int len,
                              uint64_t ss, uint64_t es)
{
    char text[64];

    if (len < 2)
        return;
    int speed = (int16_t)elrs_be(payload, 2);
    snprintf(text, sizeof(text), "Vario: %.2f m/s", speed / 100.0);
    elrs_put(di, s, ss, es, ANN_FRAME, text, NULL, NULL);
}

/* The altitude is in dm + 10000, or in m when bit 15 is set. A third byte
 * carries the vertical speed, packed as sign * (e^(|v| * 0.026) - 1) m/s.
 * The four byte frame of TBS has no vertical speed. */
static void elrs_handle_baro(struct srd_decoder_inst *di, elrs_state *s,
                             const uint8_t *payload, int len,
                             uint64_t ss, uint64_t es)
{
    char text[96];
    int pos;

    if (len < 2)
        return;
    unsigned altitude = elrs_be(payload, 2);
    if (altitude & 0x8000)
        pos = snprintf(text, sizeof(text), "Baro altitude: %u m", altitude & 0x7fff);
    else
        pos = snprintf(text, sizeof(text), "Baro altitude: %.1f m",
                       ((int)altitude - 10000) / 10.0);
    if (len == 3 && pos > 0 && (size_t)pos < sizeof(text)) {
        int packed = (int8_t)payload[2];
        double speed = exp((packed < 0 ? -packed : packed) * 0.026) - 1.0;
        snprintf(text + pos, sizeof(text) - (size_t)pos, ", vertical speed %.2f m/s",
                 packed < 0 ? -speed : speed);
    }
    elrs_put(di, s, ss, es, ANN_FRAME, text, NULL, NULL);
}

static void elrs_handle_text(struct srd_decoder_inst *di, elrs_state *s,
                             const uint8_t *payload, int len,
                             uint64_t ss, uint64_t es, const char *label)
{
    char value[128], text[160];

    elrs_latin1(value, sizeof(value), payload, len);
    if (!value[0])
        return;
    snprintf(text, sizeof(text), "%s: %s", label, value);
    elrs_put(di, s, ss, es, ANN_FRAME, text, NULL, NULL);
}

/* Behind the addresses MSP frames carry a status byte, the 16 bit MSP
 * function and a 16 bit payload length. */
static void elrs_handle_msp(struct srd_decoder_inst *di, elrs_state *s,
                            unsigned ftype, const uint8_t *payload, int len,
                            uint64_t ss, uint64_t es)
{
    char t0[128], t1[64];

    if (len < 5) {
        snprintf(t0, sizeof(t0), "MSP frame with %d byte(s)", len);
        elrs_put(di, s, ss, es, ANN_FRAME, t0, NULL, NULL);
        return;
    }
    const char *kind = ftype == 0x7a ? "request" : ftype == 0x7b ? "response"
                     : ftype == 0x7c ? "write" : "frame";
    int status = payload[0];
    unsigned function = elrs_be(payload + 1, 2);
    unsigned length = elrs_be(payload + 3, 2);
    snprintf(t0, sizeof(t0), "MSP %s: function 0x%04x, status 0x%02x, %u byte(s)",
             kind, function, status, length);
    snprintf(t1, sizeof(t1), "MSP %s 0x%04x", kind, function);
    elrs_put(di, s, ss, es, ANN_ELRS, t0, t1, NULL);
}

static void elrs_handle_payload(struct srd_decoder_inst *di, elrs_state *s,
                                unsigned ftype, const uint8_t *payload, int len,
                                uint64_t ss, uint64_t es)
{
    switch (ftype) {
    case 0x16:
        elrs_handle_channels(di, s, payload, len, ss, es);
        break;
    case 0x14:
        elrs_handle_link_stats(di, s, payload, len, ss, es);
        break;
    case 0x2e:
        elrs_handle_status(di, s, payload, len, ss, es);
        break;
    case 0x29:
        elrs_handle_device_info(di, s, payload, len, ss, es);
        break;
    case 0x2b:
        elrs_handle_param_entry(di, s, payload, len, ss, es);
        break;
    case 0x32:
        elrs_handle_command(di, s, payload, len, ss, es);
        break;
    case 0x1e:
        elrs_handle_attitude(di, s, payload, len, ss, es);
        break;
    case 0x02:
        elrs_handle_gps(di, s, payload, len, ss, es);
        break;
    case 0x07:
        elrs_handle_vario(di, s, payload, len, ss, es);
        break;
    case 0x08:
        elrs_handle_battery(di, s, payload, len, ss, es);
        break;
    case 0x09:
        elrs_handle_baro(di, s, payload, len, ss, es);
        break;
    case 0x21:
        elrs_handle_text(di, s, payload, len, ss, es, "Flight mode");
        break;
    case 0x7a:
    case 0x7b:
    case 0x7c:
        elrs_handle_msp(di, s, ftype, payload, len, ss, es);
        break;
    default:
        if (len > 0) {
            char text[16 + 3 * FRAME_MAX];
            size_t pos = (size_t)snprintf(text, sizeof(text), "Payload:");
            for (int idx = 0; idx < len; idx++)
                pos += (size_t)snprintf(text + pos, sizeof(text) - pos, " %02x", payload[idx]);
            elrs_put(di, s, ss, es, ANN_FRAME, text, NULL, NULL);
        }
        break;
    }
}

/*-- Frame assembly ---------------------------------------------------*/

static void elrs_handle_frame(struct srd_decoder_inst *di, elrs_state *s,
                              const elrs_link *link)
{
    char t0[160], t1[64], type_buf[24], addr_buf[24], orig_buf[24], dest_buf[24];
    int last = link->len - 1;
    unsigned sync = link->value[0];
    const uint8_t *body = link->value + 2;
    int body_n = link->len - 2;
    uint64_t ss = link->ss[0], es = link->es[last];
    unsigned ftype = body[0];
    unsigned crc_rx = body[body_n - 1];
    const uint8_t *payload = body + 1;
    int payload_n = body_n - 2;
    unsigned crc_calc = elrs_crc8(body, body_n - 1);
    int crc_ok = crc_rx == crc_calc;
    const char *type_name = elrs_type_name(ftype, type_buf, sizeof(type_buf));

    snprintf(t1, sizeof(t1), "0x%02x", sync);
    elrs_put(di, s, link->ss[0], link->es[0], ANN_SYNC,
             elrs_address_name(sync, addr_buf, sizeof(addr_buf)), t1, NULL);
    if (crc_ok) {
        snprintf(t0, sizeof(t0), "Checksum 0x%02x correct", crc_rx);
        elrs_put(di, s, link->ss[last], link->es[last], ANN_CRC, t0, "CRC ok", NULL);
    } else {
        snprintf(t0, sizeof(t0), "Checksum 0x%02x, expected 0x%02x", crc_rx, crc_calc);
        elrs_put(di, s, link->ss[last], link->es[last], ANN_WARN, t0,
                 "Checksum mismatch", "CRC error");
    }

    /* Extended frames carry destination and origin addresses. */
    int dest = -1, orig = -1;
    const uint8_t *ext = payload;
    int ext_n = payload_n;
    if (ftype >= EXT_TYPE_MIN && ftype <= EXT_TYPE_MAX && payload_n >= 2) {
        dest = payload[0];
        orig = payload[1];
        ext = payload + 2;
        ext_n = payload_n - 2;
        snprintf(t0, sizeof(t0), "%s: %s -> %s, %d byte(s)", type_name,
                 elrs_address_name((unsigned)orig, orig_buf, sizeof(orig_buf)),
                 elrs_address_name((unsigned)dest, dest_buf, sizeof(dest_buf)), ext_n);
    } else {
        snprintf(t0, sizeof(t0), "%s: %d byte(s)", type_name, payload_n);
    }
    elrs_put(di, s, ss, es, ANN_FRAME, t0, type_name, NULL);
    c_proto(di, ss, es, s->out_proto, "FRAME", C_U8(sync), C_U8(ftype),
            C_I32(dest), C_I32(orig), C_U8(crc_ok),
            C_BYTES(ext, (uint32_t)ext_n), C_END);

    if (!crc_ok && !s->opt_crc)
        return;
    elrs_handle_payload(di, s, ftype, ext, ext_n, ss, es);
}

static void elrs_link_reset(elrs_link *link)
{
    link->state = STATE_SYNC;
    link->len = 0;
    link->body_len = 0;
}

static void elrs_link_push(elrs_link *link, unsigned value, uint64_t ss, uint64_t es)
{
    if (link->len >= FRAME_MAX)
        return;
    link->value[link->len] = (uint8_t)value;
    link->ss[link->len] = ss;
    link->es[link->len] = es;
    link->len++;
}

static void elrs_handle_byte(struct srd_decoder_inst *di, elrs_state *s,
                             elrs_link *link, unsigned value,
                             uint64_t ss, uint64_t es)
{
    char t0[48], t1[48];

    switch (link->state) {
    case STATE_SYNC:
        /* Bytes in front of an address are noise on the wire. */
        if (elrs_lookup(elrs_addresses, ELRS_COUNT(elrs_addresses), value)) {
            link->len = 0;
            elrs_link_push(link, value, ss, es);
            link->state = STATE_LEN;
        }
        return;
    case STATE_LEN:
        if (value >= LENGTH_MIN && value <= LENGTH_MAX) {
            elrs_link_push(link, value, ss, es);
            link->body_len = (int)value;
            link->state = STATE_BODY;
            return;
        }
        /* Neither a length nor part of a frame, but it may still be the
         * address of the next frame. */
        snprintf(t0, sizeof(t0), "Invalid frame length %u", value);
        snprintf(t1, sizeof(t1), "Invalid length %u", value);
        elrs_put(di, s, ss, es, ANN_WARN, t0, t1, "Bad length");
        elrs_link_reset(link);
        elrs_handle_byte(di, s, link, value, ss, es);
        return;
    default:
        elrs_link_push(link, value, ss, es);
        if (link->len >= 2 + link->body_len) {
            elrs_handle_frame(di, s, link);
            elrs_link_reset(link);
        }
        return;
    }
}

/*-- Decoder callbacks ------------------------------------------------*/

static void elrs_decode_upper(struct srd_decoder_inst *di,
                              uint64_t start_sample, uint64_t end_sample,
                              const char *cmd, const c_field *fields, int n_fields)
{
    elrs_state *s = (elrs_state *)c_decoder_get_private(di);

    if (!s || !cmd || strcmp(cmd, "FRAME") != 0 || n_fields < 3)
        return;
    int rxtx = elrs_field_int(&fields[0]) == 1 ? 1 : 0;
    unsigned value = (unsigned)(elrs_field_int(&fields[1]) & 0xff);
    if (!elrs_field_int(&fields[2])) {
        elrs_put(di, s, start_sample, end_sample, ANN_WARN,
                 "Invalid UART frame in the CRSF stream", "UART error", NULL);
        return;
    }
    elrs_handle_byte(di, s, &s->link[rxtx], value, start_sample, end_sample);
}

static void elrs_reset(struct srd_decoder_inst *di)
{
    if (!c_decoder_get_private(di))
        c_decoder_set_private(di, g_malloc0(sizeof(elrs_state)));
    elrs_state *s = (elrs_state *)c_decoder_get_private(di);
    memset(s, 0, sizeof(*s));
    s->out_ann = -1;
    s->out_proto = -1;
    elrs_link_reset(&s->link[0]);
    elrs_link_reset(&s->link[1]);
}

static void elrs_start(struct srd_decoder_inst *di)
{
    elrs_state *s = (elrs_state *)c_decoder_get_private(di);

    s->out_ann = c_reg_out(di, SRD_OUTPUT_ANN, "elrs");
    s->out_proto = c_reg_out(di, SRD_OUTPUT_PROTO, "elrs");
    s->opt_crc = !strcmp(c_opt_str(di, "crc", "off"), "on");
    s->opt_channels = !strcmp(c_opt_str(di, "channels", "off"), "on");
    s->opt_usec = !strcmp(c_opt_str(di, "usec", "on"), "on");
}

static void elrs_decode(struct srd_decoder_inst *di)
{
    (void)di;
}

static void elrs_destroy(struct srd_decoder_inst *di)
{
    void *priv = c_decoder_get_private(di);

    if (priv) {
        g_free(priv);
        c_decoder_set_private(di, NULL);
    }
}

static struct srd_c_decoder elrs_c_decoder = {
    .id = "elrs_c",
    .name = "ExpressLRS(C)",
    .longname = "ExpressLRS (CRSF serial link) (C)",
    .desc = "ExpressLRS link between receiver, flight controller and module (C implementation)",
    .license = "gplv2+",
    .channels = NULL,
    .num_channels = 0,
    .optional_channels = NULL,
    .num_optional_channels = 0,
    .options = elrs_options,
    .num_options = NUM_OPTIONS,
    .num_annotations = NUM_ANN,
    .ann_labels = elrs_ann_labels,
    .num_annotation_rows = 6,
    .annotation_rows = elrs_ann_rows,
    .inputs = elrs_inputs,
    .num_inputs = 1,
    .outputs = elrs_outputs,
    .num_outputs = 1,
    .binary = NULL,
    .num_binary = 0,
    .tags = elrs_tags,
    .num_tags = 1,
    .state_size = sizeof(elrs_state),
    .reset = elrs_reset,
    .start = elrs_start,
    .decode = elrs_decode,
    .end = NULL,
    .metadata = NULL,
    .destroy = elrs_destroy,
    .decode_upper = elrs_decode_upper,
};

static GSList *elrs_string_values(const char *first, const char *second)
{
    GSList *list = NULL;

    list = g_slist_append(list, g_variant_new_string(first));
    list = g_slist_append(list, g_variant_new_string(second));
    return list;
}

SRD_C_DECODER_EXPORT struct srd_c_decoder *srd_c_decoder_entry(void)
{
    elrs_options[0].def = g_variant_new_string("off");
    elrs_options[0].values = elrs_string_values("off", "on");
    elrs_options[1].def = g_variant_new_string("off");
    elrs_options[1].values = elrs_string_values("off", "on");
    elrs_options[2].def = g_variant_new_string("on");
    elrs_options[2].values = elrs_string_values("on", "off");
    return &elrs_c_decoder;
}

SRD_C_DECODER_EXPORT int srd_c_decoder_api_version(void)
{
    return SRD_C_DECODER_API_VERSION;
}
