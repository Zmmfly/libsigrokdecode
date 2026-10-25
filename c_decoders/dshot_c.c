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
 * DShot decoder, native port of decoders/dshot/pd.py.
 *
 * The annotation classes, sample ranges and texts match the Python
 * decoder, tests/testdata/dshot_c holds the reference captures. See the
 * Python decoder for the description of the protocol and of the burst
 * classification. As there, a burst ends when the line stays idle for
 * longer than five microseconds, so the last frame of a capture is only
 * annotated once another edge follows it.
 *
 * Instead of one wait() call per edge, this decoder scans the packed
 * sample buffers of libsigrokdecode 64 samples at a time, which keeps
 * the long idle periods between frames cheap on large captures.
 *
 * C protocol output (c_proto(), for stacked C decoders):
 *  - "FRAME":    C_STR(motor), C_U16(value11), C_U8(tlm_req), C_U8(crc_ok),
 *                C_STR(rate name, "" when the rate is unknown)
 *  - "COMMAND":  C_U16(code), C_STR(name)
 *  - "THROTTLE": C_U16(value11), C_F64(percent, NaN for reserved values)
 *  - "TLM":      C_STR(kind), C_I64(value), C_U16(raw12)
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <glib.h>
#include "libsigrokdecode.h"

enum {
    ANN_FRAME = 0,
    ANN_THROTTLE,
    ANN_COMMAND,
    ANN_CRC,
    ANN_TLM,
    ANN_BIT,
    ANN_WARN,
    NUM_ANN,
};

/* A gap longer than this separates two bursts. */
#define DSHOT_IDLE_USEC 5.0

/*
 * Request frames have up to 40 edges and telemetry responses up to 44.
 * Longer bursts can only be reported as unexpected ones, for which the
 * pulse counts and the position of the last edge suffice.
 */
#define DSHOT_MAX_EDGES 64

/* Samples per c_fetch_packed_multi() call, the input chunk limits it. */
#define DSHOT_FETCH_BLOCK (1ULL << 24)

#define THROTTLE_MIN 48
#define THROTTLE_MAX 2047
#define THROTTLE_3D_FORWARD_MIN 1048

/* Bit period in microseconds of the four DShot variants. */
static const struct {
    const char *name;
    double usec;
} dshot_rates[] = {
    {"DShot150", 1e6 / 150e3},
    {"DShot300", 1e6 / 300e3},
    {"DShot600", 1e6 / 600e3},
    {"DShot1200", 1e6 / 1200e3},
};
#define NUM_RATES ((int)(sizeof(dshot_rates) / sizeof(dshot_rates[0])))

/* 4b/5b GCR code, listed in nibble order. */
static const uint8_t gcr_symbols[16] = {
    0x19, 0x1b, 0x12, 0x13, 0x1c, 0x15, 0x16, 0x17,
    0x1a, 0x09, 0x0a, 0x0b, 0x1e, 0x0d, 0x0e, 0x0f,
};

/* Command numbers as defined by BLHeli, NULL marks unused numbers. */
static const char *const dshot_commands[THROTTLE_MIN] = {
    [1] = "Beep 1",
    [2] = "Beep 2",
    [3] = "Beep 3",
    [4] = "Beep 4",
    [5] = "Beep 5",
    [6] = "ESC info",
    [7] = "Spin direction 1",
    [8] = "Spin direction 2",
    [9] = "3D mode off",
    [10] = "3D mode on",
    [11] = "Settings request",
    [12] = "Save settings",
    [13] = "Extended telemetry enable",
    [14] = "Extended telemetry disable",
    [20] = "Spin direction normal",
    [21] = "Spin direction reversed",
    [22] = "LED 0 on",
    [23] = "LED 1 on",
    [24] = "LED 2 on",
    [25] = "LED 3 on",
    [26] = "LED 0 off",
    [27] = "LED 1 off",
    [28] = "LED 2 off",
    [29] = "LED 3 off",
    [30] = "Audio stream mode on/off",
    [31] = "Silent mode on/off",
};

/* Extended telemetry types, indexed by half the type nibble. */
static const char *const tlm_types[8] = {
    NULL,
    "Temperature",
    "Voltage",
    "Current",
    "Debug 1",
    "Debug 2",
    "Debug 3",
    "State",
};

typedef struct {
    uint64_t sample;
    int level;
} dshot_edge;

typedef struct {
    uint64_t ss;
    uint64_t es;
} dshot_pulse;

typedef struct {
    uint64_t samplerate;
    uint64_t idle_samples;
    int out_ann;
    int out_proto;

    /* Options. */
    char rate[16];
    char motor[16];
    int tlm;
    int mode_3d;
    int bits;

    /* Level of the most recently scanned sample. */
    int have_level;
    int level;

    /* The current burst. */
    dshot_edge edges[DSHOT_MAX_EDGES];
    uint64_t num_edges;
    uint64_t num_high;
    uint64_t num_low;
    uint64_t last_sample;
    int last_level;
} dshot_state;

static struct srd_channel dshot_channels[] = {
    {"data", "Data", "DShot signal", 0, SRD_CHANNEL_SDATA, NULL},
};

static struct srd_decoder_option dshot_options[] = {
    {"rate", NULL, "Bit rate", NULL, NULL},
    {"motor", NULL, "Motor number", NULL, NULL},
    {"tlm", NULL, "Decode ESC telemetry response", NULL, NULL},
    {"3d", NULL, "3D mode throttle ranges", NULL, NULL},
    {"bits", NULL, "Show individual symbol bits", NULL, NULL},
};
#define NUM_OPTIONS ((int)(sizeof(dshot_options) / sizeof(dshot_options[0])))

static const char *dshot_ann_labels[][3] = {
    {"", "frame", "Frame"},
    {"", "throttle", "Throttle"},
    {"", "command", "Command"},
    {"", "crc", "Checksum"},
    {"", "tlm", "Telemetry"},
    {"", "bits", "Symbol bits"},
    {"", "warn", "Warning"},
};

static const int dshot_row_frames[] = {ANN_FRAME, ANN_THROTTLE, ANN_COMMAND, ANN_CRC};
static const int dshot_row_telemetry[] = {ANN_TLM};
static const int dshot_row_symbols[] = {ANN_BIT};
static const int dshot_row_warnings[] = {ANN_WARN};

static const struct srd_c_ann_row dshot_ann_rows[] = {
    {"frames", "Frames", dshot_row_frames, 4},
    {"telemetry", "Telemetry", dshot_row_telemetry, 1},
    {"symbols", "Symbols", dshot_row_symbols, 1},
    {"warnings", "Warnings", dshot_row_warnings, 1},
};

static const char *dshot_inputs[] = {"logic", NULL};
static const char *dshot_outputs[] = {"dshot", NULL};
static const char *dshot_tags[] = {"Radio/control/RC", NULL};

/*-- Helpers ----------------------------------------------------------*/

static void dshot_put(struct srd_decoder_inst *di, const dshot_state *s,
                      uint64_t ss, uint64_t es, int cls,
                      const char *t0, const char *t1, const char *t2)
{
    const char *texts[4] = {t0, t1, t2, NULL};
    struct srd_c_annotation ann = {cls, 0, (char **)texts, "", 0};

    c_decoder_put(di, ss, es, s->out_ann, &ann);
}

static unsigned dshot_crc4(unsigned value)
{
    value &= 0xfff;
    return (value ^ (value >> 4) ^ (value >> 8)) & 0xf;
}

static int gcr_nibble(unsigned symbol)
{
    for (int idx = 0; idx < 16; idx++) {
        if (gcr_symbols[idx] == symbol)
            return idx;
    }
    return -1;
}

/*
 * Consecutive symbols start exactly one bit period apart, which makes
 * the first edge of each symbol the most reliable reference.
 */
static double dshot_sample_period(const dshot_pulse *pulses, int count)
{
    if (count >= 2)
        return (double)(pulses[count - 1].ss - pulses[0].ss) / (double)(count - 1);
    if (count == 1)
        return (double)(pulses[0].es - pulses[0].ss);
    return 0.0;
}

/* A '1' keeps the active level for 75% of the period, a '0' for 37.5%. */
static int dshot_pulses_to_bits(const dshot_pulse *pulses, int count,
                                double period, int *bits)
{
    int dubious = 0;

    for (int idx = 0; idx < count; idx++) {
        double duty = period != 0.0
            ? (double)(pulses[idx].es - pulses[idx].ss) / period : 0.0;
        bits[idx] = duty > 0.5625 ? 1 : 0;
        if (duty > 0.45 && duty < 0.65)
            dubious++;
    }
    return dubious;
}

/* Index of the rate which matches the measured bit period, or -1. */
static int dshot_rate_index(const dshot_state *s, double period)
{
    int best = -1;
    double best_delta = 0.0;

    if (!s->samplerate || period == 0.0)
        return -1;
    double usec = period * 1e6 / (double)s->samplerate;
    for (int idx = 0; idx < NUM_RATES; idx++) {
        double nominal = dshot_rates[idx].usec;
        double delta = fabs(usec - nominal) / nominal;
        if (best < 0 || delta < best_delta) {
            best = idx;
            best_delta = delta;
        }
    }
    if (best < 0 || best_delta > 0.25)
        return -1;
    return best;
}

/*
 * Decode 20 GCR coded bits. Returns -1 for an invalid symbol, otherwise
 * whether the checksum matched. *order is 0 when the first symbol holds
 * the high nibble, 1 when it holds the low nibble, and -1 when neither
 * order produces a valid checksum.
 */
static int dshot_gcr_to_value(const int *bits, unsigned *value, int *order)
{
    int nibbles[4];

    for (int sym = 0; sym < 4; sym++) {
        unsigned symbol = 0;
        for (int idx = 0; idx < 5; idx++)
            symbol = (symbol << 1) | (unsigned)bits[sym * 5 + idx];
        nibbles[sym] = gcr_nibble(symbol);
        if (nibbles[sym] < 0)
            return -1;
    }
    for (int pass = 0; pass < 2; pass++) {
        unsigned val = 0;
        for (int idx = 0; idx < 4; idx++)
            val = (val << 4) | (unsigned)nibbles[pass ? 3 - idx : idx];
        if ((val & 0xf) == dshot_crc4(val >> 4)) {
            *value = val;
            *order = pass;
            return 1;
        }
    }
    *value = ((unsigned)nibbles[0] << 12) | ((unsigned)nibbles[1] << 8) |
             ((unsigned)nibbles[2] << 4) | (unsigned)nibbles[3];
    *order = -1;
    return 0;
}

static void dshot_annotate_bits(struct srd_decoder_inst *di,
                                const dshot_state *s,
                                const dshot_pulse *pulses, const int *bits,
                                int count)
{
    char text[48], bit[8];

    if (!s->bits)
        return;
    for (int idx = 0; idx < count; idx++) {
        double usec = s->samplerate
            ? (double)(pulses[idx].es - pulses[idx].ss) * 1e6 / (double)s->samplerate
            : 0.0;
        snprintf(text, sizeof(text), "%d (%.2f us)", bits[idx], usec);
        snprintf(bit, sizeof(bit), "%d", bits[idx]);
        dshot_put(di, s, pulses[idx].ss, pulses[idx].es, ANN_BIT, text, bit, NULL);
    }
}

/*-- Request frames ---------------------------------------------------*/

static void dshot_annotate_throttle(struct srd_decoder_inst *di,
                                    const dshot_state *s,
                                    uint64_t ss, uint64_t es, unsigned value11)
{
    char t0[64], t1[48], t2[16];
    double percent;

    if (value11 < THROTTLE_MIN) {
        snprintf(t0, sizeof(t0), "Reserved value %u", value11);
        dshot_put(di, s, ss, es, ANN_THROTTLE, t0, "Reserved", NULL);
        c_proto(di, ss, es, s->out_proto, "THROTTLE", C_U16(value11),
                C_F64(NAN), C_END);
        return;
    }
    if (s->mode_3d) {
        if (value11 < THROTTLE_3D_FORWARD_MIN) {
            int span = THROTTLE_3D_FORWARD_MIN - 1 - THROTTLE_MIN;
            percent = 100.0 * (double)(value11 - THROTTLE_MIN) / (double)span;
            snprintf(t0, sizeof(t0), "Throttle %u (3D reverse %.1f%%)", value11, percent);
            snprintf(t1, sizeof(t1), "%u reverse %.0f%%", value11, percent);
            dshot_put(di, s, ss, es, ANN_THROTTLE, t0, t1, NULL);
            c_proto(di, ss, es, s->out_proto, "THROTTLE", C_U16(value11),
                    C_F64(-percent), C_END);
            return;
        }
        int span = THROTTLE_MAX - THROTTLE_3D_FORWARD_MIN;
        percent = 100.0 * (double)(value11 - THROTTLE_3D_FORWARD_MIN) / (double)span;
        snprintf(t0, sizeof(t0), "Throttle %u (3D forward %.1f%%)", value11, percent);
        snprintf(t1, sizeof(t1), "%u forward %.0f%%", value11, percent);
        dshot_put(di, s, ss, es, ANN_THROTTLE, t0, t1, NULL);
        c_proto(di, ss, es, s->out_proto, "THROTTLE", C_U16(value11),
                C_F64(percent), C_END);
        return;
    }
    int span = THROTTLE_MAX - THROTTLE_MIN;
    percent = 100.0 * (double)(value11 - THROTTLE_MIN) / (double)span;
    snprintf(t0, sizeof(t0), "Throttle %u (%.1f%%)", value11, percent);
    snprintf(t1, sizeof(t1), "%u (%.0f%%)", value11, percent);
    snprintf(t2, sizeof(t2), "%.0f%%", percent);
    dshot_put(di, s, ss, es, ANN_THROTTLE, t0, t1, t2);
    c_proto(di, ss, es, s->out_proto, "THROTTLE", C_U16(value11),
            C_F64(percent), C_END);
}

static void dshot_handle_request(struct srd_decoder_inst *di, dshot_state *s,
                                 const dshot_pulse *pulses, int count)
{
    char t0[96], t1[64], t2[64];
    int bits[20] = {0};
    unsigned value16 = 0;
    int crc_ok, gcr_order = -1;
    double period = dshot_sample_period(pulses, count);
    uint64_t ss = pulses[0].ss, es = pulses[count - 1].es;
    int dubious = dshot_pulses_to_bits(pulses, count, period, bits);

    dshot_annotate_bits(di, s, pulses, bits, count);
    if (count == 20) {
        crc_ok = dshot_gcr_to_value(bits, &value16, &gcr_order);
        if (crc_ok < 0) {
            dshot_put(di, s, ss, es, ANN_WARN,
                      "Invalid GCR symbol in DShot1200 frame",
                      "Invalid GCR symbol", "GCR error");
            return;
        }
    } else {
        for (int idx = 0; idx < count; idx++)
            value16 = (value16 << 1) | (unsigned)bits[idx];
        crc_ok = (value16 & 0xf) == dshot_crc4(value16 >> 4);
    }

    unsigned value11 = value16 >> 5;
    int tlm_req = (value16 >> 4) & 1;
    unsigned crc_rx = value16 & 0xf;
    unsigned crc_calc = dshot_crc4(value16 >> 4);
    int rate = dshot_rate_index(s, period);
    const char *rate_name = rate >= 0 ? dshot_rates[rate].name : NULL;

    /* The configured rate is a hint, a mismatch only raises a warning. */
    if (strcmp(s->rate, "auto") != 0 && rate_name && strcmp(s->rate, rate_name) != 0) {
        snprintf(t0, sizeof(t0), "Measured %s instead of configured %s", rate_name, s->rate);
        snprintf(t1, sizeof(t1), "Measured %s, not %s", rate_name, s->rate);
        dshot_put(di, s, ss, es, ANN_WARN, t0, t1, "Rate mismatch");
    }
    if (dubious) {
        snprintf(t0, sizeof(t0), "%d symbol(s) with ambiguous duty cycle", dubious);
        snprintf(t1, sizeof(t1), "%d ambiguous symbol(s)", dubious);
        dshot_put(di, s, ss, es, ANN_WARN, t0, t1, "Ambiguous timing");
    }

    snprintf(t0, sizeof(t0), "M%s %s 0x%04x%s", s->motor,
             rate_name ? rate_name : "unknown rate", value16,
             gcr_order == 1 ? " (low first)" : "");
    snprintf(t1, sizeof(t1), "M%s", s->motor);
    dshot_put(di, s, ss, es, ANN_FRAME, t0, t1, NULL);
    c_proto(di, ss, es, s->out_proto, "FRAME", C_STR(s->motor), C_U16(value11),
            C_U8(tlm_req), C_U8(crc_ok), C_STR(rate_name ? rate_name : ""), C_END);

    /* Throttle values and command numbers share the same field. */
    if (value11 < THROTTLE_MIN && dshot_commands[value11]) {
        const char *name = dshot_commands[value11];
        snprintf(t0, sizeof(t0), "Command %u (%s)", value11, name);
        snprintf(t1, sizeof(t1), "%u (%s)", value11, name);
        dshot_put(di, s, ss, es, ANN_COMMAND, t0, t1, name);
        c_proto(di, ss, es, s->out_proto, "COMMAND", C_U16(value11),
                C_STR(name), C_END);
    } else if (value11 == 0) {
        dshot_put(di, s, ss, es, ANN_THROTTLE, "Motor stop (0)", "Stop", NULL);
        c_proto(di, ss, es, s->out_proto, "THROTTLE", C_U16(0), C_F64(0.0), C_END);
    } else {
        dshot_annotate_throttle(di, s, ss, es, value11);
    }

    if (crc_ok) {
        snprintf(t2, sizeof(t2), "Checksum 0x%x correct", crc_rx);
    } else {
        snprintf(t2, sizeof(t2), "Checksum 0x%x, expected 0x%x", crc_rx, crc_calc);
        snprintf(t0, sizeof(t0), "Checksum mismatch, got 0x%x expected 0x%x", crc_rx, crc_calc);
        dshot_put(di, s, ss, es, ANN_WARN, t0, "Checksum mismatch", "CRC error");
    }
    dshot_put(di, s, ss, es, ANN_CRC, t2, crc_ok ? "CRC ok" : "CRC bad", NULL);
    if (tlm_req)
        dshot_put(di, s, ss, es, ANN_TLM, "Telemetry requested", "TLM req", NULL);
}

/*-- Telemetry responses ----------------------------------------------*/

/*
 * Bits 11..8 hold the telemetry type, bits 7..0 the value. An odd type
 * (or type zero) selects the commutation period which every ESC reports.
 */
static const char *dshot_decode_tlm(unsigned raw12, long long *value,
                                    char *text, size_t text_size)
{
    unsigned tlm_type = (raw12 >> 8) & 0xf;

    if (!(tlm_type & 1) && tlm_type) {
        const char *name = tlm_types[tlm_type >> 1];
        unsigned val = raw12 & 0xff;
        *value = val;
        if (!strcmp(name, "Temperature"))
            snprintf(text, text_size, "%u degC", val);
        else if (!strcmp(name, "Voltage"))
            snprintf(text, text_size, "%.2f V", val * 0.25);
        else if (!strcmp(name, "Current"))
            snprintf(text, text_size, "%u A", val);
        else
            snprintf(text, text_size, "%s %u", name, val);
        return name;
    }
    unsigned exponent = (raw12 >> 9) & 0x7;
    unsigned mantissa = raw12 & 0x1ff;
    unsigned period = mantissa << exponent;
    if (!period) {
        *value = 0;
        snprintf(text, text_size, "0 eRPM");
        return "eRPM";
    }
    /* The conversion which the flight controller firmware applies. */
    unsigned erpm = (600000 + period / 2) / period;
    *value = erpm;
    snprintf(text, text_size, "%u eRPM", erpm);
    return "eRPM";
}

/* The response has a start bit and 20 GCR coded payload bits. */
static int dshot_handle_response(struct srd_decoder_inst *di, dshot_state *s,
                                 const dshot_pulse *pulses, uint64_t count)
{
    char t0[96], t1[64], text[48];
    int bits[21] = {0};
    unsigned value16;
    int order;
    long long value;

    if (count != 21)
        return 0;
    double period = dshot_sample_period(pulses, 21);
    int dubious = dshot_pulses_to_bits(pulses, 21, period, bits);
    if (bits[0] != 0)
        return 0;
    int crc_ok = dshot_gcr_to_value(bits + 1, &value16, &order);
    if (crc_ok < 0)
        return 0;
    uint64_t ss = pulses[0].ss, es = pulses[20].es;
    dshot_annotate_bits(di, s, pulses, bits, 21);

    unsigned raw12 = value16 & 0xfff;
    unsigned crc_rx = value16 >> 12;
    unsigned crc_calc = dshot_crc4(raw12);
    const char *kind = dshot_decode_tlm(raw12, &value, text, sizeof(text));
    if (!strcmp(kind, "eRPM")) {
        snprintf(t0, sizeof(t0), "Telemetry: %lld eRPM (raw 0x%03x)", value, raw12);
        snprintf(t1, sizeof(t1), "%lld eRPM", value);
        dshot_put(di, s, ss, es, ANN_TLM, t0, t1, "eRPM");
    } else {
        snprintf(t0, sizeof(t0), "Telemetry: %s (raw 0x%03x)", text, raw12);
        dshot_put(di, s, ss, es, ANN_TLM, t0, text, kind);
    }
    c_proto(di, ss, es, s->out_proto, "TLM", C_STR(kind), C_I64(value),
            C_U16(raw12), C_END);

    if (!crc_ok) {
        snprintf(t0, sizeof(t0), "Telemetry checksum 0x%x, expected 0x%x", crc_rx, crc_calc);
        dshot_put(di, s, ss, es, ANN_WARN, t0, "Telemetry checksum error", "TLM CRC error");
    }
    if (dubious) {
        snprintf(t0, sizeof(t0), "Telemetry with %d ambiguous symbol(s)", dubious);
        snprintf(t1, sizeof(t1), "%d ambiguous TLM symbol(s)", dubious);
        dshot_put(di, s, ss, es, ANN_WARN, t0, t1, "Ambiguous timing");
    }
    return 1;
}

/*-- Burst segmentation -----------------------------------------------*/

/* Collect the (start, end) of the segments which have the given level. */
static int dshot_collect_pulses(const dshot_state *s, int level,
                                dshot_pulse *pulses, int max_pulses)
{
    int count = 0;

    for (uint64_t idx = 0; idx + 1 < s->num_edges && idx + 1 < DSHOT_MAX_EDGES; idx++) {
        if (s->edges[idx].level != level)
            continue;
        if (count == max_pulses)
            break;
        pulses[count].ss = s->edges[idx].sample;
        pulses[count].es = s->edges[idx + 1].sample;
        count++;
    }
    return count;
}

static void dshot_handle_burst(struct srd_decoder_inst *di, dshot_state *s)
{
    char t0[96], t1[64];
    dshot_pulse high[22], low[21];

    /* The trailing idle period behind the last edge is not part of it. */
    if (s->num_edges < 3)
        return;
    uint64_t ss = s->edges[0].sample, es = s->last_sample;
    uint64_t num_high = s->num_high, num_low = s->num_low;

    /* A request frame starts with a high pulse and carries one pulse per
     * symbol. Twenty pulses are DShot1200 frames. */
    if (s->edges[0].level == 1 && (num_high == 16 || num_high == 20) &&
        num_low == num_high - 1) {
        int count = dshot_collect_pulses(s, 1, high, 20);
        if ((uint64_t)count == num_high)
            dshot_handle_request(di, s, high, count);
        return;
    }
    /* The inverted telemetry response has one low pulse per symbol. */
    if (num_low == 21 && s->tlm) {
        int nlow = dshot_collect_pulses(s, 0, low, 21);
        int nhigh = dshot_collect_pulses(s, 1, high, 22);
        if (dshot_handle_response(di, s, low, (uint64_t)nlow) ||
            dshot_handle_response(di, s, high, (uint64_t)nhigh))
            return;
        dshot_put(di, s, ss, es, ANN_WARN,
                  "Telemetry response without valid symbols or checksum",
                  "Undecodable telemetry response", "TLM error");
        return;
    }
    snprintf(t0, sizeof(t0), "Unexpected burst, %llu high and %llu low pulses",
             (unsigned long long)num_high, (unsigned long long)num_low);
    snprintf(t1, sizeof(t1), "Unexpected burst, %llu / %llu pulses",
             (unsigned long long)num_high, (unsigned long long)num_low);
    dshot_put(di, s, ss, es, ANN_WARN, t0, t1, "Unexpected burst");
}

static void dshot_clear_burst(dshot_state *s)
{
    s->num_edges = 0;
    s->num_high = 0;
    s->num_low = 0;
}

/* A level which does not change for a while separates two bursts. */
static void dshot_handle_edge(struct srd_decoder_inst *di, dshot_state *s,
                              uint64_t sample, int level)
{
    if (s->num_edges) {
        if (sample - s->last_sample > s->idle_samples) {
            dshot_handle_burst(di, s);
            dshot_clear_burst(s);
        } else if (s->last_level) {
            s->num_high++;
        } else {
            s->num_low++;
        }
    }
    if (s->num_edges < DSHOT_MAX_EDGES) {
        s->edges[s->num_edges].sample = sample;
        s->edges[s->num_edges].level = level;
    }
    s->num_edges++;
    s->last_sample = sample;
    s->last_level = level;
}

/*-- Packed sample scanner --------------------------------------------*/

static inline unsigned dshot_ctz64(uint64_t value)
{
#if defined(_MSC_VER) && defined(_M_X64)
    unsigned long index = 0;
    _BitScanForward64(&index, value);
    return (unsigned)index;
#else
    return (unsigned)__builtin_ctzll(value);
#endif
}

static inline uint64_t dshot_low_mask(unsigned bits)
{
    return bits >= 64 ? UINT64_MAX : ((1ULL << bits) - 1ULL);
}

/* Load 'width' samples which start at a byte boundary. */
static inline uint64_t dshot_load(const uint8_t *buf, uint8_t const_value,
                                  uint64_t bit_index, unsigned width)
{
    uint64_t word = 0;

    if (!buf)
        return const_value == 1 ? dshot_low_mask(width) : 0;
    const uint8_t *src = buf + (bit_index >> 3);
    unsigned bytes = (width + 7) >> 3;
    for (unsigned idx = 0; idx < bytes; idx++)
        word |= (uint64_t)src[idx] << (8 * idx);
    return word & dshot_low_mask(width);
}

static inline int dshot_bit(const uint8_t *buf, uint8_t const_value, uint64_t bit_index)
{
    if (!buf)
        return const_value == 1;
    return (buf[bit_index >> 3] >> (bit_index & 7)) & 1;
}

/* Report every level change within one word of samples. */
static inline void dshot_scan_word(struct srd_decoder_inst *di, dshot_state *s,
                                   uint64_t word, unsigned width, uint64_t base)
{
    uint64_t changes = (word ^ ((word << 1) | (uint64_t)s->level)) & dshot_low_mask(width);

    while (changes) {
        unsigned bit = dshot_ctz64(changes);
        dshot_handle_edge(di, s, base + bit, (int)((word >> bit) & 1));
        changes &= changes - 1;
    }
    s->level = (int)((word >> (width - 1)) & 1);
}

static void dshot_scan(struct srd_decoder_inst *di, dshot_state *s,
                       const uint8_t *buf, uint8_t const_value,
                       uint64_t count, uint64_t base, uint8_t bit_offset)
{
    uint64_t pos = 0;

    /* Like an edge condition, the first sample only sets the level. */
    if (!s->have_level) {
        s->level = dshot_bit(buf, const_value, bit_offset);
        s->have_level = 1;
    }
    /* Advance sample by sample to the next byte boundary. */
    while (pos < count && ((bit_offset + pos) & 7)) {
        int level = dshot_bit(buf, const_value, bit_offset + pos);
        if (level != s->level) {
            s->level = level;
            dshot_handle_edge(di, s, base + pos, level);
        }
        pos++;
    }
    while (count - pos >= 64) {
        uint64_t word = dshot_load(buf, const_value, bit_offset + pos, 64);
        if (word != (s->level ? UINT64_MAX : 0))
            dshot_scan_word(di, s, word, 64, base + pos);
        pos += 64;
    }
    if (pos < count) {
        unsigned width = (unsigned)(count - pos);
        uint64_t word = dshot_load(buf, const_value, bit_offset + pos, width);
        dshot_scan_word(di, s, word, width, base + pos);
    }
}

/*-- Decoder callbacks ------------------------------------------------*/

static void dshot_reset(struct srd_decoder_inst *di)
{
    if (!c_decoder_get_private(di))
        c_decoder_set_private(di, g_malloc0(sizeof(dshot_state)));
    dshot_state *s = (dshot_state *)c_decoder_get_private(di);
    memset(s, 0, sizeof(*s));
    s->out_ann = -1;
    s->out_proto = -1;
}

static void dshot_copy_option(struct srd_decoder_inst *di, const char *key,
                              const char *defval, char *dst, size_t dst_size)
{
    const char *val = c_opt_str(di, key, defval);
    g_strlcpy(dst, val ? val : defval, dst_size);
}

static void dshot_start(struct srd_decoder_inst *di)
{
    dshot_state *s = (dshot_state *)c_decoder_get_private(di);

    s->out_ann = c_reg_out(di, SRD_OUTPUT_ANN, "dshot");
    s->out_proto = c_reg_out(di, SRD_OUTPUT_PROTO, "dshot");
    dshot_copy_option(di, "rate", "auto", s->rate, sizeof(s->rate));
    dshot_copy_option(di, "motor", "1", s->motor, sizeof(s->motor));
    s->tlm = !strcmp(c_opt_str(di, "tlm", "on"), "on");
    s->mode_3d = !strcmp(c_opt_str(di, "3d", "off"), "on");
    s->bits = !strcmp(c_opt_str(di, "bits", "off"), "on");
}

static void dshot_metadata(struct srd_decoder_inst *di, int key, uint64_t value)
{
    dshot_state *s = (dshot_state *)c_decoder_get_private(di);

    if (s && key == SRD_CONF_SAMPLERATE)
        s->samplerate = value;
}

static void dshot_decode(struct srd_decoder_inst *di)
{
    dshot_state *s = (dshot_state *)c_decoder_get_private(di);

    if (!s)
        return;
    if (!s->samplerate)
        s->samplerate = c_samplerate(di);
    /* Without a samplerate or a signal there is nothing to decode. */
    if (!s->samplerate || !c_has_ch(di, 0))
        return;
    /* Round half to even, like Python's round(). */
    double idle = rint((double)s->samplerate * DSHOT_IDLE_USEC * 1e-6);
    s->idle_samples = idle < 1.0 ? 1 : (uint64_t)idle;

    for (;;) {
        const int channels[1] = {0};
        const uint8_t *buffers[1] = {NULL};
        uint8_t const_values[1] = {0};
        uint64_t base = 0;
        uint8_t bit_offset = 0;
        uint64_t count = c_fetch_packed_multi(di, channels, buffers,
                                              const_values, 1, DSHOT_FETCH_BLOCK,
                                              &base, &bit_offset);
        if (!count)
            break;
        dshot_scan(di, s, buffers[0], const_values[0], count, base, bit_offset);
        if (c_consume_samples(di, count) != SRD_OK)
            break;
    }
}

static void dshot_destroy(struct srd_decoder_inst *di)
{
    void *priv = c_decoder_get_private(di);

    if (priv) {
        g_free(priv);
        c_decoder_set_private(di, NULL);
    }
}

static struct srd_c_decoder dshot_c_decoder = {
    .id = "dshot_c",
    .name = "DShot(C)",
    .longname = "DShot (digital ESC protocol) (C)",
    .desc = "Digital throttle and telemetry protocol for brushless ESCs (C implementation)",
    .license = "gplv2+",
    .channels = dshot_channels,
    .num_channels = 1,
    .optional_channels = NULL,
    .num_optional_channels = 0,
    .options = dshot_options,
    .num_options = NUM_OPTIONS,
    .num_annotations = NUM_ANN,
    .ann_labels = dshot_ann_labels,
    .num_annotation_rows = 4,
    .annotation_rows = dshot_ann_rows,
    .inputs = dshot_inputs,
    .num_inputs = 1,
    .outputs = dshot_outputs,
    .num_outputs = 1,
    .binary = NULL,
    .num_binary = 0,
    .tags = dshot_tags,
    .num_tags = 1,
    .state_size = sizeof(dshot_state),
    .reset = dshot_reset,
    .start = dshot_start,
    .decode = dshot_decode,
    .end = NULL,
    .metadata = dshot_metadata,
    .destroy = dshot_destroy,
    .decode_upper = NULL,
};

static GSList *dshot_string_values(const char *const *values, int count)
{
    GSList *list = NULL;

    for (int idx = 0; idx < count; idx++)
        list = g_slist_append(list, g_variant_new_string(values[idx]));
    return list;
}

SRD_C_DECODER_EXPORT struct srd_c_decoder *srd_c_decoder_entry(void)
{
    static const char *const rates[] = {"auto", "DShot150", "DShot300", "DShot600", "DShot1200"};
    static const char *const motors[] = {"1", "2", "3", "4", "5", "6", "7", "8"};
    static const char *const on_off[] = {"on", "off"};
    static const char *const off_on[] = {"off", "on"};

    dshot_options[0].def = g_variant_new_string("auto");
    dshot_options[0].values = dshot_string_values(rates, 5);
    dshot_options[1].def = g_variant_new_string("1");
    dshot_options[1].values = dshot_string_values(motors, 8);
    dshot_options[2].def = g_variant_new_string("on");
    dshot_options[2].values = dshot_string_values(on_off, 2);
    dshot_options[3].def = g_variant_new_string("off");
    dshot_options[3].values = dshot_string_values(off_on, 2);
    dshot_options[4].def = g_variant_new_string("off");
    dshot_options[4].values = dshot_string_values(off_on, 2);
    return &dshot_c_decoder;
}

SRD_C_DECODER_EXPORT int srd_c_decoder_api_version(void)
{
    return SRD_C_DECODER_API_VERSION;
}
