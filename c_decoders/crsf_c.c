/*
 * This file is part of the libsigrokdecode project.
 *
 * Copyright (C) 2023 James Cordell <james@cordell.org.uk>
 * Copyright (C) 2025 C port contributors
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

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <glib.h>
#include "libsigrokdecode.h"

/*
 * Port of decoders/crsf/pd.py, which is the reference: both have to produce
 * the same annotations (see tests/testdata/crsf_c).
 */

enum {
    ANN_TEXT = 0,
    ANN_ERROR,
    NUM_ANN,
};

enum crsf_state {
    CRSF_WAIT_SYNC,
    CRSF_WAIT_LEN,
    CRSF_WAIT_TYPE,
    CRSF_WAIT_BODY,
};

/* The length byte counts the type, the payload and the CRC byte. */
#define CRSF_LEN_MIN 2
#define CRSF_LEN_MAX 62

/* CRSF sync byte definitions */
static const struct {
    uint8_t byte;
    const char *name;
    const char *short_name;
} crsf_sync_bytes[] = {
    {0xEE, "To Transmitter Module", "To TX Module"},
    {0xEA, "To Handset", "To HS"},
    {0xC8, "To Flight Controller", "To FC"},
    {0xEC, "To Receiver", "To RX"},
    {0, NULL, NULL}
};

/* CRSF frame type definitions */
static const struct {
    uint8_t type;
    const char *name;
} crsf_frame_types[] = {
    {0x02, "GPS"},
    {0x07, "Vario"},
    {0x08, "Battery sensor"},
    {0x09, "Baro altitude"},
    {0x0B, "Heartbeat"},
    {0x10, "OpenTX sync"},
    {0x14, "LINK_STATISTICS"},
    {0x16, "CRSF_FRAMETYPE_RC_CHANNELS_PACKED"},
    {0x1E, "Attitude"},
    {0x21, "Flight mode"},
    {0x28, "CRSF_FRAMETYPE_DEVICE_PING"},
    {0x29, "CRSF_FRAMETYPE_DEVICE_INFO"},
    {0x2A, "Request settings"},
    {0x2B, "CRSF_FRAMETYPE_PARAMETER_SETTINGS_ENTRY"},
    {0x2C, "CRSF_FRAMETYPE_PARAMETER_READ"},
    {0x2D, "CRSF_FRAMETYPE_PARAMETER_WRITE"},
    {0x32, "CRSF_FRAMETYPE_COMMAND"},
    {0x3A, "Radio"},
    {0, NULL}
};

/* Uplink TX power levels of the link statistics frame, in mW. */
static const int crsf_tx_powers[] = {0, 10, 25, 100, 500, 1000, 2000, 250, 50};

typedef struct {
    int out_ann;
    enum crsf_state state;
    int len_byte;
    /* Value and data bit span of the type, payload and CRC bytes. */
    int body_len;
    uint8_t value[CRSF_LEN_MAX];
    uint64_t ss[CRSF_LEN_MAX];
    uint64_t es[CRSF_LEN_MAX];
} crsf_state;

static const char *crsf_inputs[] = {"uart", NULL};
static const char *crsf_outputs[] = {"crsf", NULL};
static const char *crsf_tags[] = {"Radio/control/RC", NULL};

static const char *crsf_ann_labels[][3] = {
    {"", "text-verbose", "Human-readable text (verbose)"},
    {"", "text-error", "Human-readable Error text"},
};

static const int crsf_row_normal_classes[] = {ANN_TEXT, ANN_ERROR, -1};

static const struct srd_c_ann_row crsf_ann_rows[] = {
    {"normal", "Normal", crsf_row_normal_classes, 2},
};

static const char *crsf_get_frame_type_name(unsigned type)
{
    for (int i = 0; crsf_frame_types[i].name; i++) {
        if (crsf_frame_types[i].type == type)
            return crsf_frame_types[i].name;
    }
    return NULL;
}

/* CRC-8/DVB-S2 (polynomial 0xD5) of one more byte. */
static uint8_t crsf_crc8(uint8_t byte, uint8_t crc)
{
    crc ^= byte;
    for (int i = 0; i < 8; i++)
        crc = (crc & 0x80) ? (uint8_t)((crc << 1) ^ 0xD5) : (uint8_t)(crc << 1);
    return crc;
}

/* Big-endian unsigned value of `n` bytes. */
static uint32_t crsf_be(const uint8_t *p, int n)
{
    uint32_t v = 0;
    for (int i = 0; i < n; i++)
        v = (v << 8) | p[i];
    return v;
}

/* Text up to the first NUL byte, Latin-1 decoded into UTF-8. */
static void crsf_latin1(char *out, size_t size, const uint8_t *data, int len)
{
    size_t pos = 0;

    for (int i = 0; i < len && data[i]; i++) {
        uint8_t ch = data[i];
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

static void crsf_restart(crsf_state *s)
{
    s->state = CRSF_WAIT_SYNC;
    s->len_byte = 0;
    s->body_len = 0;
}

static void crsf_put_unknown(struct srd_decoder_inst *di, crsf_state *s,
                             uint64_t ss, uint64_t es, unsigned value)
{
    char t[32];
    snprintf(t, sizeof(t), "Unknown packet type%u", value);
    c_put(di, ss, es, s->out_ann, ANN_TEXT, t);
}

/*
 * Span of bit `bit` of the payload. uart_c reports the span of the data
 * bits only, split it evenly into the bit slots (the Python uart rounds
 * every bit boundary, at most one sample apart).
 */
static void crsf_bit_span(const crsf_state *s, int bit, uint64_t *ss, uint64_t *es)
{
    int idx = 1 + bit / 8, pos = bit % 8;
    double width = (double)(s->es[idx] - s->ss[idx]) / 8.0;

    *ss = s->ss[idx] + (uint64_t)llround(pos * width);
    *es = s->ss[idx] + (uint64_t)llround((pos + 1) * width);
}

static void crsf_handle_rc_channels(struct srd_decoder_inst *di, crsf_state *s)
{
    /* 16 channels of 11 bits, LSB first. Each channel spans its bits. */
    const uint8_t *payload = s->value + 1;

    for (int chan = 0; chan < 16; chan++) {
        unsigned raw = 0;
        for (int b = 0; b < 11; b++) {
            int bit = chan * 11 + b;
            if (payload[bit / 8] & (1u << (bit % 8)))
                raw |= 1u << b;
        }
        uint64_t ss, es, unused;
        crsf_bit_span(s, chan * 11, &ss, &unused);
        crsf_bit_span(s, chan * 11 + 10, &unused, &es);
        char t[32];
        snprintf(t, sizeof(t), "Chan:%d Value:%u", chan, raw);
        c_put(di, ss, es, s->out_ann, ANN_TEXT, t);
    }
}

static void crsf_handle_link_statistics(struct srd_decoder_inst *di, crsf_state *s)
{
    const uint8_t *p = s->value + 1;
    char t[10][48], power[24];

    if ((size_t)p[6] < sizeof(crsf_tx_powers) / sizeof(crsf_tx_powers[0]))
        snprintf(power, sizeof(power), "%d mW", crsf_tx_powers[p[6]]);
    else
        snprintf(power, sizeof(power), "level %u", p[6]);
    snprintf(t[0], sizeof(t[0]), "Uplink RSSI 1: -%udB", p[0]);
    snprintf(t[1], sizeof(t[1]), "Uplink RSSI 2: -%udB", p[1]);
    snprintf(t[2], sizeof(t[2]), "Uplink Link Quality: %u", p[2]);
    snprintf(t[3], sizeof(t[3]), "Uplink SNR: %ddB", (int8_t)p[3]);
    snprintf(t[4], sizeof(t[4]), "Active Antenna: %u", p[4]);
    snprintf(t[5], sizeof(t[5]), "RF Mode: %u", p[5]);
    snprintf(t[6], sizeof(t[6]), "Uplink TX Power: %s", power);
    snprintf(t[7], sizeof(t[7]), "Downlink RSSI: -%udB", p[7]);
    snprintf(t[8], sizeof(t[8]), "Downlink Link Quality: %u", p[8]);
    snprintf(t[9], sizeof(t[9]), "Downlink SNR: %ddB", (int8_t)p[9]);
    /* Each field spans its byte. */
    for (int i = 0; i < 10; i++)
        c_put(di, s->ss[1 + i], s->es[1 + i], s->out_ann, ANN_TEXT, t[i]);
}

/*
 * One line for the telemetry frames with a known layout. Multi-byte fields
 * are big-endian. Returns 0 when the payload does not fit the layout.
 */
static int crsf_telemetry_text(char *t, size_t size, unsigned type,
                               const uint8_t *p, int len)
{
    switch (type) {
    case 0x02:
        if (len < 15)
            return 0;
        /* deg * 1e7, km/h * 10, deg * 100, m + 1000 */
        snprintf(t, size, "GPS: %.7f, %.7f, %.1f km/h, heading %.2f deg, %d m, %u satellite(s)",
                 (int32_t)crsf_be(p, 4) / 1e7, (int32_t)crsf_be(p + 4, 4) / 1e7,
                 crsf_be(p + 8, 2) / 10.0, crsf_be(p + 10, 2) / 100.0,
                 (int)crsf_be(p + 12, 2) - 1000, p[14]);
        return 1;
    case 0x07:
        if (len < 2)
            return 0;
        /* cm/s */
        snprintf(t, size, "Vario: %.2f m/s", (int16_t)crsf_be(p, 2) / 100.0);
        return 1;
    case 0x08: {
        if (len < 7)
            return 0;
        /* 0.1 V, 0.1 A, mAh, % */
        int pos = snprintf(t, size, "Battery: %.1f V, %.1f A, %u mAh",
                           crsf_be(p, 2) / 10.0, crsf_be(p + 2, 2) / 10.0, crsf_be(p + 4, 3));
        if (len >= 8 && pos > 0 && (size_t)pos < size)
            snprintf(t + pos, size - (size_t)pos, ", %u%%", p[7]);
        return 1;
    }
    case 0x09: {
        if (len < 2)
            return 0;
        unsigned altitude = crsf_be(p, 2);
        int pos;
        if (altitude & 0x8000)  /* m, from 2276.8 m on */
            pos = snprintf(t, size, "Baro altitude: %u m", altitude & 0x7FFF);
        else  /* dm + 10000 */
            pos = snprintf(t, size, "Baro altitude: %.1f m", ((int)altitude - 10000) / 10.0);
        if (len == 3 && pos > 0 && (size_t)pos < size) {
            /* Packed vertical speed, sign * (e^(|v| * 0.026) - 1) m/s. */
            int packed = (int8_t)p[2];
            double speed = exp(abs(packed) * 0.026) - 1.0;
            snprintf(t + pos, size - (size_t)pos, ", vertical speed %.2f m/s",
                     packed < 0 ? -speed : speed);
        }
        return 1;
    }
    case 0x1E:
        if (len < 6)
            return 0;
        /* rad * 10000 */
        snprintf(t, size, "Attitude: pitch %.4f rad, roll %.4f rad, yaw %.4f rad",
                 (int16_t)crsf_be(p, 2) / 10000.0, (int16_t)crsf_be(p + 2, 2) / 10000.0,
                 (int16_t)crsf_be(p + 4, 2) / 10000.0);
        return 1;
    case 0x21: {
        /* Null-terminated string */
        char name[2 * CRSF_LEN_MAX + 1];
        crsf_latin1(name, sizeof(name), p, len);
        if (!name[0])
            return 0;
        snprintf(t, size, "Flight mode: %s", name);
        return 1;
    }
    }
    return 0;
}

static void crsf_put_checksum(struct srd_decoder_inst *di, crsf_state *s)
{
    int last = s->body_len - 1;
    uint8_t crc = 0;

    for (int i = 0; i < last; i++)
        crc = crsf_crc8(s->value[i], crc);
    if (crc == s->value[last]) {
        c_put(di, s->ss[last], s->es[last], s->out_ann, ANN_TEXT,
              "Checksum crc8 poly 0xD5.");
    } else {
        char t[48];
        snprintf(t, sizeof(t), "Checksum 0x%02x, expected 0x%02x", s->value[last], crc);
        c_put(di, s->ss[last], s->es[last], s->out_ann, ANN_ERROR, t, "Checksum error");
    }
}

static void crsf_handle_frame(struct srd_decoder_inst *di, crsf_state *s)
{
    unsigned type = s->value[0];
    /* The payload sits between the type and the CRC byte. */
    int len = s->body_len - 2;

    if (type == 0x16 && len >= 22) {
        crsf_handle_rc_channels(di, s);
    } else if (type == 0x14 && len >= 10) {
        crsf_handle_link_statistics(di, s);
    } else if (len > 0) {
        /* One line over the payload bytes. */
        char t[256];
        if (!crsf_telemetry_text(t, sizeof(t), type, s->value + 1, len))
            snprintf(t, sizeof(t), "%s (%d bytes payload)", crsf_get_frame_type_name(type), len);
        c_put(di, s->ss[1], s->es[len], s->out_ann, ANN_TEXT, t);
    }
    crsf_put_checksum(di, s);
}

static void crsf_recv_proto(struct srd_decoder_inst *di, uint64_t start_sample, uint64_t end_sample, const char *cmd, const c_field *fields, int n_fields)
{
    crsf_state *s = (crsf_state *)c_decoder_get_private(di);
    if (!s)
        return;

    /* For now, ignore all UART packets except the actual data packets. */
    if (strcmp(cmd, "DATA") != 0 || n_fields < 1)
        return;

    unsigned byte_val = fields[0].u8;

    switch (s->state) {
    case CRSF_WAIT_SYNC:
        for (int i = 0; crsf_sync_bytes[i].name; i++) {
            if (crsf_sync_bytes[i].byte == byte_val) {
                c_put(di, start_sample, end_sample, s->out_ann, ANN_TEXT,
                      crsf_sync_bytes[i].name, crsf_sync_bytes[i].short_name);
                s->state = CRSF_WAIT_LEN;
                return;
            }
        }
        crsf_put_unknown(di, s, start_sample, end_sample, byte_val);
        return;

    case CRSF_WAIT_LEN:
        if (byte_val >= CRSF_LEN_MIN && byte_val <= CRSF_LEN_MAX) {
            char t[48];
            snprintf(t, sizeof(t), "Num of bytes succeeding: %u", byte_val - 2);
            c_put(di, start_sample, end_sample, s->out_ann, ANN_TEXT, t);
            s->len_byte = (int)byte_val;
            s->state = CRSF_WAIT_TYPE;
        } else {
            crsf_put_unknown(di, s, start_sample, end_sample, byte_val);
            crsf_restart(s);
        }
        return;

    case CRSF_WAIT_TYPE: {
        const char *name = crsf_get_frame_type_name(byte_val);
        if (!name) {
            crsf_put_unknown(di, s, start_sample, end_sample, byte_val);
            crsf_restart(s);
            return;
        }
        c_put(di, start_sample, end_sample, s->out_ann, ANN_TEXT, name);
        s->state = CRSF_WAIT_BODY;
        break;
    }

    case CRSF_WAIT_BODY:
        break;
    }

    s->value[s->body_len] = (uint8_t)byte_val;
    s->ss[s->body_len] = start_sample;
    s->es[s->body_len] = end_sample;
    s->body_len++;
    if (s->body_len == s->len_byte) {
        crsf_handle_frame(di, s);
        crsf_restart(s);
    }
}

static void crsf_reset(struct srd_decoder_inst *di)
{
    if (!c_decoder_get_private(di))
        c_decoder_set_private(di, g_malloc0(sizeof(crsf_state)));
    crsf_state *s = (crsf_state *)c_decoder_get_private(di);
    memset(s, 0, sizeof(crsf_state));
    s->state = CRSF_WAIT_SYNC;
}

static void crsf_start(struct srd_decoder_inst *di)
{
    crsf_state *s = (crsf_state *)c_decoder_get_private(di);
    s->out_ann = c_reg_out(di, SRD_OUTPUT_ANN, "crsf");
}

static void crsf_decode(struct srd_decoder_inst *di)
{
    (void)di;
}

static void crsf_destroy(struct srd_decoder_inst *di)
{
    void *priv = c_decoder_get_private(di);
    if (priv) {
        g_free(priv);
        c_decoder_set_private(di, NULL);
    }
}

struct srd_c_decoder crsf_c_decoder = {
    .id = "crsf_c",
    .name = "CRSF(C)",
    .longname = "Crossfire RC protocol (C)",
    .desc = "A protocol for radio control systems. (C implementation)",
    .license = "gplv2+",
    .channels = NULL,
    .num_channels = 0,
    .optional_channels = NULL,
    .num_optional_channels = 0,
    .options = NULL,
    .num_options = 0,
    .num_annotations = NUM_ANN,
    .ann_labels = crsf_ann_labels,
    .num_annotation_rows = 1,
    .annotation_rows = crsf_ann_rows,
    .inputs = crsf_inputs,
    .num_inputs = 1,
    .outputs = crsf_outputs,
    .num_outputs = 1,
    .binary = NULL,
    .num_binary = 0,
    .tags = crsf_tags,
    .num_tags = 1,
    .reset = crsf_reset,
    .start = crsf_start,
    .decode = crsf_decode,
    .destroy = crsf_destroy,
    .decode_upper = crsf_recv_proto,
    .state_size = 0,
};

SRD_C_DECODER_EXPORT struct srd_c_decoder *srd_c_decoder_entry(void)
{
    return &crsf_c_decoder;
}

SRD_C_DECODER_EXPORT int srd_c_decoder_api_version(void)
{
    return SRD_C_DECODER_API_VERSION;
}