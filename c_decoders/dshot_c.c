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
 * decoder, tests/testdata/dshot_c holds the reference cases. See the
 * Python decoder for the description of the protocol: a request is 16
 * pulses with one bit period, an idle-high line marks bidirectional
 * DShot, whose inverted requests the ESC answers with a GCR coded
 * response at 5/4 of the bit rate. The floating point expressions keep
 * the operand order of the Python decoder, so both compute the same
 * sample positions.
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

/* Longest accepted bit period (DShot150 plus margin), in us. */
#define PERIOD_MAX_USEC 8.0
/* Largest deviation of a bit from the median period of the frame. */
#define PERIOD_TOLERANCE 0.2
/* Share of the bit period above which the active level is a '1'. */
#define DUTY_THRESHOLD 0.53
#define DUTY_AMBIGUOUS_LO 0.46
#define DUTY_AMBIGUOUS_HI 0.60
/* The ESC starts its response within this time after a request ends. */
#define RESPONSE_WINDOW_USEC 60.0
/* Telemetry bits per request bit period, telemetry bits per response. */
#define TLM_BIT_RATIO 0.8
#define TLM_BITS 21

/* Edges of one request frame, and the most edges kept of a response. */
#define REQ_EDGES 32
#define RESP_MAX_EDGES 256

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

/* 4b/5b GCR code of the telemetry response, listed in nibble order. */
static const uint8_t gcr_symbols[16] = {
    0x19, 0x1b, 0x12, 0x13, 0x1d, 0x15, 0x16, 0x17,
    0x1a, 0x09, 0x0a, 0x0b, 0x1e, 0x0d, 0x0e, 0x0f,
};

/* Command numbers as defined by BLHeli and Betaflight, NULL marks unused numbers. */
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
    [32] = "Signal line telemetry disable",
    [33] = "Signal line telemetry enable",
    [34] = "Continuous eRPM telemetry",
    [35] = "Continuous eRPM period telemetry",
    [42] = "Temperature telemetry",
    [43] = "Voltage telemetry",
    [44] = "Current telemetry",
    [45] = "Consumption telemetry",
    [46] = "eRPM telemetry",
    [47] = "eRPM period telemetry",
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

enum {
    ST_IDLE = 0, /* waiting for the first edge of a burst */
    ST_REQ,      /* collecting the edges of a request */
    ST_WAIT,     /* inverted request done, waiting for the response */
    ST_RESP,     /* collecting the edges of a response */
};

typedef struct {
    uint64_t samplerate;
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

    int state;
    uint64_t last_edge;
    /* Edges of the current burst which may still start a request. */
    dshot_edge burst[REQ_EDGES + 2];
    int num_burst;
    /* Pulses dropped from the burst while looking for a request. */
    uint64_t dropped;
    uint64_t drop_ss;
    /* Mean bit period and end of the last request. */
    double period;
    double req_end;
    /* Edges of the current response. */
    dshot_edge resp[RESP_MAX_EDGES];
    int num_resp;
} dshot_state;

static struct srd_channel dshot_channels[] = {
    {"data", "Data", "DShot signal", 0, SRD_CHANNEL_SDATA, NULL},
};

static struct srd_decoder_option dshot_options[] = {
    {"rate", NULL, "Bit rate", NULL, NULL},
    {"motor", NULL, "Motor number", NULL, NULL},
    {"tlm", NULL, "Decode ESC telemetry response", NULL, NULL},
    {"3d", NULL, "3D mode throttle ranges", NULL, NULL},
    {"bits", NULL, "Show individual bits", NULL, NULL},
};
#define NUM_OPTIONS ((int)(sizeof(dshot_options) / sizeof(dshot_options[0])))

static const char *dshot_ann_labels[][3] = {
    {"", "frame", "Frame"},
    {"", "throttle", "Throttle"},
    {"", "command", "Command"},
    {"", "crc", "Checksum"},
    {"", "tlm", "Telemetry"},
    {"", "bits", "Bits"},
    {"", "warn", "Warning"},
};

static const int dshot_row_frames[] = {ANN_FRAME, ANN_THROTTLE, ANN_COMMAND, ANN_CRC};
static const int dshot_row_telemetry[] = {ANN_TLM};
static const int dshot_row_symbols[] = {ANN_BIT};
static const int dshot_row_warnings[] = {ANN_WARN};

static const struct srd_c_ann_row dshot_ann_rows[] = {
    {"frames", "Frames", dshot_row_frames, 4},
    {"telemetry", "Telemetry", dshot_row_telemetry, 1},
    {"symbols", "Bits", dshot_row_symbols, 1},
    {"warnings", "Warnings", dshot_row_warnings, 1},
};

static const char *dshot_inputs[] = {"logic", NULL};
static const char *dshot_outputs[] = {"dshot", NULL};
static const char *dshot_tags[] = {"Radio/control/RC", NULL};

/*-- Helpers ----------------------------------------------------------*/

/* Sample positions are doubles; truncate like Python's int(). */
static void dshot_put(struct srd_decoder_inst *di, const dshot_state *s,
                      double ss, double es, int cls,
                      const char *t0, const char *t1, const char *t2)
{
    const char *texts[4] = {t0, t1, t2, NULL};
    struct srd_c_annotation ann = {cls, 0, (char **)texts, "", 0};

    c_decoder_put(di, (uint64_t)ss, (uint64_t)es, s->out_ann, &ann);
}

static double dshot_usec(const dshot_state *s, double samples)
{
    return samples * 1e6 / (double)s->samplerate;
}

static double dshot_samples(const dshot_state *s, double usec)
{
    return usec * (double)s->samplerate / 1e6;
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

static int cmp_u64(const void *a, const void *b)
{
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : (x > y);
}

/* Name the rate which matches the measured bit period, or NULL. */
static const char *dshot_rate_name(const dshot_state *s, double period)
{
    double usec = dshot_usec(s, period);
    int best = -1;
    double best_delta = 0.0;

    for (int idx = 0; idx < NUM_RATES; idx++) {
        double delta = fabs(usec - dshot_rates[idx].usec) / dshot_rates[idx].usec;
        if (best < 0 || delta < best_delta) {
            best = idx;
            best_delta = delta;
        }
    }
    if (best < 0 || best_delta > 0.25)
        return NULL;
    return dshot_rates[best].name;
}

/*-- Request frames ---------------------------------------------------*/

static void dshot_annotate_throttle(struct srd_decoder_inst *di,
                                    const dshot_state *s,
                                    double ss, double es, unsigned value11)
{
    char t0[64], t1[48], t2[16];
    double percent;

    if (value11 < THROTTLE_MIN) {
        snprintf(t0, sizeof(t0), "Reserved value %u", value11);
        dshot_put(di, s, ss, es, ANN_THROTTLE, t0, "Reserved", NULL);
        c_proto(di, (uint64_t)ss, (uint64_t)es, s->out_proto, "THROTTLE", C_U16(value11),
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
            c_proto(di, (uint64_t)ss, (uint64_t)es, s->out_proto, "THROTTLE", C_U16(value11),
                    C_F64(-percent), C_END);
            return;
        }
        int span = THROTTLE_MAX - THROTTLE_3D_FORWARD_MIN;
        percent = 100.0 * (double)(value11 - THROTTLE_3D_FORWARD_MIN) / (double)span;
        snprintf(t0, sizeof(t0), "Throttle %u (3D forward %.1f%%)", value11, percent);
        snprintf(t1, sizeof(t1), "%u forward %.0f%%", value11, percent);
        dshot_put(di, s, ss, es, ANN_THROTTLE, t0, t1, NULL);
        c_proto(di, (uint64_t)ss, (uint64_t)es, s->out_proto, "THROTTLE", C_U16(value11),
                C_F64(percent), C_END);
        return;
    }
    int span = THROTTLE_MAX - THROTTLE_MIN;
    percent = 100.0 * (double)(value11 - THROTTLE_MIN) / (double)span;
    snprintf(t0, sizeof(t0), "Throttle %u (%.1f%%)", value11, percent);
    snprintf(t1, sizeof(t1), "%u (%.0f%%)", value11, percent);
    snprintf(t2, sizeof(t2), "%.0f%%", percent);
    dshot_put(di, s, ss, es, ANN_THROTTLE, t0, t1, t2);
    c_proto(di, (uint64_t)ss, (uint64_t)es, s->out_proto, "THROTTLE", C_U16(value11),
            C_F64(percent), C_END);
}

static void dshot_handle_request(struct srd_decoder_inst *di, dshot_state *s,
                                 const uint64_t *lead, const double *duties,
                                 double period, int inverted)
{
    char t0[96], t1[64], t2[64], rate_txt[32];
    int bits[16];
    unsigned value16 = 0;
    int dubious = 0;
    double ss = (double)lead[0];
    double es = (double)lead[0] + 16 * period;

    for (int idx = 0; idx < 16; idx++) {
        bits[idx] = duties[idx] > DUTY_THRESHOLD;
        if (duties[idx] > DUTY_AMBIGUOUS_LO && duties[idx] < DUTY_AMBIGUOUS_HI)
            dubious++;
        value16 = (value16 << 1) | (unsigned)bits[idx];
    }
    unsigned value11 = value16 >> 5;
    int tlm_req = (value16 >> 4) & 1;
    unsigned crc_rx = value16 & 0xf;
    unsigned crc_calc = dshot_crc4(value16 >> 4);
    if (inverted)
        crc_calc ^= 0xf;
    int crc_ok = crc_rx == crc_calc;
    const char *rate = dshot_rate_name(s, period);

    if (s->bits) {
        for (int idx = 0; idx < 16; idx++) {
            double width = dshot_usec(s, duties[idx] * period);
            snprintf(t0, sizeof(t0), "%d (%.2f us)", bits[idx], width);
            snprintf(t1, sizeof(t1), "%d", bits[idx]);
            dshot_put(di, s, (double)lead[idx], (double)lead[idx] + period, ANN_BIT, t0, t1, NULL);
        }
    }

    /* The configured rate is a hint, a mismatch only raises a warning. */
    if (strcmp(s->rate, "auto") != 0 && rate && strcmp(s->rate, rate) != 0) {
        snprintf(t0, sizeof(t0), "Measured %s instead of configured %s", rate, s->rate);
        snprintf(t1, sizeof(t1), "Measured %s, not %s", rate, s->rate);
        dshot_put(di, s, ss, es, ANN_WARN, t0, t1, "Rate mismatch");
    }
    if (dubious) {
        snprintf(t0, sizeof(t0), "%d bit(s) with ambiguous duty cycle", dubious);
        snprintf(t1, sizeof(t1), "%d ambiguous bit(s)", dubious);
        dshot_put(di, s, ss, es, ANN_WARN, t0, t1, "Ambiguous timing");
    }

    if (rate)
        g_strlcpy(rate_txt, rate, sizeof(rate_txt));
    else
        snprintf(rate_txt, sizeof(rate_txt), "%.0f kbit/s", 1e3 / dshot_usec(s, period));
    snprintf(t0, sizeof(t0), "M%s %s%s 0x%04x", s->motor, rate_txt,
             inverted ? " bidir" : "", value16);
    snprintf(t1, sizeof(t1), "M%s 0x%04x", s->motor, value16);
    snprintf(t2, sizeof(t2), "M%s", s->motor);
    dshot_put(di, s, ss, es, ANN_FRAME, t0, t1, t2);
    c_proto(di, (uint64_t)ss, (uint64_t)es, s->out_proto, "FRAME", C_STR(s->motor),
            C_U16(value11), C_U8(tlm_req), C_U8(crc_ok), C_STR(rate ? rate : ""), C_END);

    /* Throttle values and command numbers share the same field. */
    if (value11 < THROTTLE_MIN && dshot_commands[value11]) {
        const char *name = dshot_commands[value11];
        snprintf(t0, sizeof(t0), "Command %u (%s)", value11, name);
        snprintf(t1, sizeof(t1), "%u (%s)", value11, name);
        dshot_put(di, s, ss, es, ANN_COMMAND, t0, t1, name);
        c_proto(di, (uint64_t)ss, (uint64_t)es, s->out_proto, "COMMAND", C_U16(value11),
                C_STR(name), C_END);
    } else if (value11 == 0) {
        dshot_put(di, s, ss, es, ANN_THROTTLE, "Motor stop (0)", "Stop", NULL);
        c_proto(di, (uint64_t)ss, (uint64_t)es, s->out_proto, "THROTTLE", C_U16(0),
                C_F64(0.0), C_END);
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

    s->period = period;
    if (inverted) {
        /* Bidirectional DShot: the ESC answers every frame. */
        s->state = ST_WAIT;
        s->req_end = es;
    } else {
        s->state = ST_IDLE;
    }
}

/* Within a frame the line keeps a level for at most one bit period. */
static double dshot_gap_limit(const dshot_state *s)
{
    uint64_t periods[REQ_EDGES / 2];
    int n = 0;

    for (int idx = 2; idx < s->num_burst; idx += 2)
        periods[n++] = s->burst[idx].sample - s->burst[idx - 2].sample;
    if (n >= 1) {
        qsort(periods, (size_t)n, sizeof(periods[0]), cmp_u64);
        return 3.0 * (double)periods[n / 2];
    }
    return 3 * dshot_samples(s, PERIOD_MAX_USEC);
}

static void dshot_warn_burst(struct srd_decoder_inst *di, const dshot_state *s,
                             double ss, double es, uint64_t pulses)
{
    char t0[64], t1[48];

    snprintf(t0, sizeof(t0), "Unexpected burst of %llu pulses, no frame",
             (unsigned long long)pulses);
    snprintf(t1, sizeof(t1), "Unexpected burst (%llu)", (unsigned long long)pulses);
    dshot_put(di, s, ss, es, ANN_WARN, t0, t1, "Burst");
}

static void dshot_finish_burst(struct srd_decoder_inst *di, dshot_state *s)
{
    uint64_t pulses = (uint64_t)(s->num_burst / 2) + s->dropped;

    if (pulses && s->num_burst) {
        double ss = s->dropped ? (double)s->drop_ss : (double)s->burst[0].sample;
        dshot_warn_burst(di, s, ss, (double)s->burst[s->num_burst - 1].sample, pulses);
    }
    s->num_burst = 0;
    s->dropped = 0;
    s->state = ST_IDLE;
}

/* Check whether the first 32 edges of the burst are a request frame. */
static int dshot_try_request(struct srd_decoder_inst *di, dshot_state *s)
{
    const dshot_edge *e = s->burst;
    int active = e[0].level;
    uint64_t lead[16], trail[16], periods[15];
    double duties[16];

    for (int idx = 0; idx < 16; idx++) {
        if (e[2 * idx].level != active || e[2 * idx + 1].level == active)
            return 0;
        lead[idx] = e[2 * idx].sample;
        trail[idx] = e[2 * idx + 1].sample;
    }
    for (int idx = 0; idx < 15; idx++)
        periods[idx] = lead[idx + 1] - lead[idx];
    qsort(periods, 15, sizeof(periods[0]), cmp_u64);
    double period = (double)periods[7];
    if (period <= 0 || dshot_usec(s, period) > PERIOD_MAX_USEC)
        return 0;
    if ((double)periods[0] < period * (1 - PERIOD_TOLERANCE) ||
            (double)periods[14] > period * (1 + PERIOD_TOLERANCE))
        return 0;
    double dmin = 1.0, dmax = 0.0;
    for (int idx = 0; idx < 16; idx++) {
        double duty = (double)(trail[idx] - lead[idx]) / period;
        dmin = duty < dmin ? duty : dmin;
        dmax = duty > dmax ? duty : dmax;
    }
    if (dmin <= 0.1 || dmax >= 0.92)
        return 0;
    if (s->dropped)
        dshot_warn_burst(di, s, (double)s->drop_ss, (double)lead[0], s->dropped);
    s->num_burst -= REQ_EDGES;
    memmove(s->burst, s->burst + REQ_EDGES, (size_t)s->num_burst * sizeof(s->burst[0]));
    s->dropped = 0;
    /* The mean of the 15 periods is far more precise than any single one. */
    period = (double)(lead[15] - lead[0]) / 15.0;
    for (int idx = 0; idx < 16; idx++)
        duties[idx] = (double)(trail[idx] - lead[idx]) / period;
    dshot_handle_request(di, s, lead, duties, period, active == 0);
    return 1;
}

static void dshot_request_edge(struct srd_decoder_inst *di, dshot_state *s,
                               uint64_t sample, int level)
{
    if (s->state == ST_IDLE) {
        s->state = ST_REQ;
        s->num_burst = 0;
        s->dropped = 0;
    }
    s->burst[s->num_burst].sample = sample;
    s->burst[s->num_burst].level = level;
    s->num_burst++;
    while (s->num_burst >= REQ_EDGES) {
        if (dshot_try_request(di, s))
            return;
        /* Drop the oldest pulse and retry with the next edges. */
        if (!s->dropped)
            s->drop_ss = s->burst[0].sample;
        s->num_burst -= 2;
        memmove(s->burst, s->burst + 2, (size_t)s->num_burst * sizeof(s->burst[0]));
        s->dropped++;
    }
}

/*-- Telemetry responses ----------------------------------------------*/

/*
 * An even, nonzero upper nibble selects an extended telemetry frame with
 * the value in the lower 8 bits, everything else is the commutation
 * period. Fills the three annotation texts, returns the kind.
 */
static const char *dshot_decode_tlm(unsigned raw12, long long *value,
                                    char t[3][96])
{
    unsigned tlm_type = (raw12 >> 8) & 0xf;

    if (tlm_type && !(tlm_type & 1)) {
        const char *name = tlm_types[tlm_type >> 1];
        unsigned v = raw12 & 0xff;
        char text[32];
        if (!strcmp(name, "Temperature"))
            snprintf(text, sizeof(text), "%u degC", v);
        else if (!strcmp(name, "Voltage"))
            snprintf(text, sizeof(text), "%.2f V", v * 0.25);
        else if (!strcmp(name, "Current"))
            snprintf(text, sizeof(text), "%u A", v);
        else
            snprintf(text, sizeof(text), "0x%02x", v);
        snprintf(t[0], 96, "%s: %s (raw 0x%03x)", name, text, raw12);
        snprintf(t[1], 96, "%s %s", name, text);
        snprintf(t[2], 96, "%s", text);
        *value = v;
        return name;
    }
    unsigned period = (raw12 & 0x1ff) << (raw12 >> 9);
    if (raw12 == 0xfff || !period) {
        snprintf(t[0], 96, "eRPM 0, motor stopped (raw 0x%03x)", raw12);
        snprintf(t[1], 96, "eRPM 0 (stopped)");
        snprintf(t[2], 96, "0 eRPM");
        *value = 0;
        return "eRPM";
    }
    unsigned erpm = (60000000u + period / 2) / period;
    snprintf(t[0], 96, "eRPM %u (period %u us, raw 0x%03x)", erpm, period, raw12);
    snprintf(t[1], 96, "eRPM %u (%u us)", erpm, period);
    snprintf(t[2], 96, "%u eRPM", erpm);
    *value = erpm;
    return "eRPM";
}

static void dshot_handle_response(struct srd_decoder_inst *di, dshot_state *s,
                                  const dshot_edge *e, int n)
{
    static int slots[RESP_MAX_EDGES];
    char t[3][96], w0[96];
    double tb = s->period * TLM_BIT_RATIO;
    uint64_t t0 = e[0].sample;

    /*
     * Bit slot of every edge: first from the rounded run lengths, then
     * once more from the bit time fitted to those slots.
     */
    slots[0] = 0;
    for (int idx = 1; idx < n; idx++) {
        double steps = rint((double)(e[idx].sample - e[idx - 1].sample) / tb);
        slots[idx] = slots[idx - 1] + (steps < 1.0 ? 1 : (int)steps);
    }
    int64_t den = 0, num = 0;
    for (int idx = 0; idx < n; idx++) {
        den += (int64_t)slots[idx] * slots[idx];
        num += (int64_t)slots[idx] * (int64_t)(e[idx].sample - t0);
    }
    if (den) {
        double fit = (double)num / (double)den;
        if (fabs(fit / tb - 1) < 0.15) {
            tb = fit;
            for (int idx = 1; idx < n; idx++) {
                double k = rint((double)(e[idx].sample - t0) / tb);
                int slot = (int)k;
                slots[idx] = slot > slots[idx - 1] + 1 ? slot : slots[idx - 1] + 1;
            }
        }
    }

    double ss = (double)t0;
    double es = (double)t0 + TLM_BITS * tb;
    if (slots[n - 1] > TLM_BITS || e[n - 1].level != 1) {
        double end = es > (double)e[n - 1].sample ? es : (double)e[n - 1].sample;
        snprintf(w0, sizeof(w0), "Telemetry response longer than %d bits", TLM_BITS);
        dshot_put(di, s, ss, end, ANN_WARN, w0, "Telemetry too long", "TLM error");
        return;
    }
    int levels[TLM_BITS];
    int idx = 0;
    for (int slot = 0; slot < TLM_BITS; slot++) {
        while (idx + 1 < n && slots[idx + 1] <= slot)
            idx++;
        levels[slot] = e[idx].level;
    }
    unsigned gcr = 0;
    for (int slot = 1; slot < TLM_BITS; slot++)
        gcr = (gcr << 1) | (unsigned)(levels[slot] ^ levels[slot - 1]);

    if (s->bits) {
        for (int slot = 0; slot < TLM_BITS; slot++) {
            char text[4];
            if (slot == 0)
                g_strlcpy(text, "S", sizeof(text));
            else
                snprintf(text, sizeof(text), "%d", levels[slot] ^ levels[slot - 1]);
            dshot_put(di, s, ss + slot * tb, ss + (slot + 1) * tb, ANN_BIT, text, NULL, NULL);
        }
    }

    unsigned value16 = 0;
    for (int nib = 3; nib >= 0; nib--) {
        unsigned symbol = (gcr >> (5 * nib)) & 0x1f;
        int nibble = gcr_nibble(symbol);
        if (nibble < 0) {
            snprintf(w0, sizeof(w0), "Invalid GCR symbol 0x%02x in telemetry response", symbol);
            dshot_put(di, s, ss, es, ANN_WARN, w0, "Invalid GCR symbol", "GCR error");
            return;
        }
        value16 = (value16 << 4) | (unsigned)nibble;
    }

    unsigned raw12 = value16 >> 4;
    unsigned csum = value16 ^ (value16 >> 4) ^ (value16 >> 8) ^ (value16 >> 12);
    int csum_ok = (csum & 0xf) == 0xf;
    long long value = 0;
    const char *kind = dshot_decode_tlm(raw12, &value, t);
    if (!csum_ok)
        g_strlcat(t[0], " (checksum error)", sizeof(t[0]));
    dshot_put(di, s, ss, es, ANN_TLM, t[0], t[1], t[2]);
    c_proto(di, (uint64_t)ss, (uint64_t)es, s->out_proto, "TLM", C_STR(kind),
            C_I64(value), C_U16(raw12), C_END);
    if (!csum_ok) {
        snprintf(w0, sizeof(w0), "Telemetry checksum mismatch, nibble XOR 0x%x", csum & 0xf);
        dshot_put(di, s, ss, es, ANN_WARN, w0, "Telemetry checksum error", "TLM CRC error");
    }
}

static void dshot_finish_response(struct srd_decoder_inst *di, dshot_state *s)
{
    int n = s->num_resp;

    s->num_resp = 0;
    s->state = ST_IDLE;
    if (!s->tlm || !n)
        return;
    dshot_handle_response(di, s, s->resp, n);
}

/*-- Edge stream ------------------------------------------------------*/

/* Timeouts which elapsed before an edge at `sample`. */
static void dshot_expire(struct srd_decoder_inst *di, dshot_state *s, uint64_t sample)
{
    if (s->state == ST_REQ) {
        if ((double)(sample - s->last_edge) > dshot_gap_limit(s))
            dshot_finish_burst(di, s);
    } else if (s->state == ST_WAIT) {
        double window = dshot_samples(s, RESPONSE_WINDOW_USEC);
        if ((double)sample > s->req_end + window) {
            if (s->tlm)
                dshot_put(di, s, s->req_end, s->req_end + window, ANN_WARN,
                          "No telemetry response", "No response", "No TLM");
            s->state = ST_IDLE;
        }
    } else if (s->state == ST_RESP) {
        double tb = s->period * TLM_BIT_RATIO;
        if ((double)(sample - s->resp[0].sample) > (TLM_BITS + 4) * tb)
            dshot_finish_response(di, s);
    }
}

static void dshot_handle_edge(struct srd_decoder_inst *di, dshot_state *s,
                              uint64_t sample, int level)
{
    dshot_expire(di, s, sample);
    if (s->state == ST_WAIT) {
        if (level == 0) {
            /* The ESC pulls the idle-high line low: a response. */
            s->state = ST_RESP;
            s->resp[0].sample = sample;
            s->resp[0].level = level;
            s->num_resp = 1;
            s->last_edge = sample;
            return;
        }
        s->state = ST_IDLE;
    } else if (s->state == ST_RESP) {
        int idx = s->num_resp < RESP_MAX_EDGES ? s->num_resp++ : RESP_MAX_EDGES - 1;
        s->resp[idx].sample = sample;
        s->resp[idx].level = level;
        s->last_edge = sample;
        return;
    }
    dshot_request_edge(di, s, sample, level);
    s->last_edge = sample;
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
        s->last_edge = base;
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

/* Complete what the end of the capture interrupted. */
static void dshot_end(struct srd_decoder_inst *di)
{
    dshot_state *s = (dshot_state *)c_decoder_get_private(di);

    if (!s || !s->samplerate)
        return;
    if (s->state == ST_RESP)
        dshot_finish_response(di, s);
    else if (s->state == ST_REQ)
        dshot_finish_burst(di, s);
    s->state = ST_IDLE;
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
    .end = dshot_end,
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
