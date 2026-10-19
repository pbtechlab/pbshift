#pragma once
#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "pb_voice.h"
// Isolate the modified backend from applications using upstream Signalsmith.
#pragma push_macro("signalsmith")
#undef signalsmith
#define signalsmith pbshift_speech_vendor
#include "pb_speech_spectrum.h"
#pragma pop_macro("signalsmith")
namespace pbshift {
class SpeechPitchProcessor {
    static constexpr int packet = 32, lookahead = 16;
    std::unique_ptr<pbshift_speech_vendor::stretch::SignalsmithStretch<float>> engine;
    int channels = 0, window = 0, fill = 0, sampleRate = 48000;
    long long received = 0, produced = 0, rawWrite = 0, rawRead = 0, write = 0, readPos = 0, cap = 65536, target = 0;
    bool done = false;
    double inputMaximum = 0, gain = 1;
    std::vector<std::vector<float>> input, output;
    std::vector<const float*> ip;
    std::vector<float*> op;
    std::vector<Ring<float>> ring, raw;
    Ring<float> peakPrefix, ceiling;
    void publish() {
        const double release = std::exp(-1.0 / (.02 * sampleRate));
        while (rawWrite - rawRead > lookahead) {
            if (done && write >= target) {
                ++rawRead;
                continue;
            }
            double required = 1;
            for (int k = 0; k <= lookahead; ++k) {
                double peak = 0;
                for (int c = 0; c < channels; ++c) peak = std::max(peak, std::abs(double(raw[c].at(rawRead + k))));
                const double desired = peak > 1e-20 ? std::min(1.0, double(ceiling.at(rawRead + k)) / peak) : 1;
                required = std::min(required, desired + (1 - desired) * k / lookahead);
            }
            gain = std::min(required, 1 - (1 - gain) * release);
            if (write - readPos >= cap) {
                for (auto& r : ring) r.grow(cap * 2, readPos, write);
                cap *= 2;
            }
            for (int c = 0; c < channels; ++c) ring[c].at(write) = float(raw[c].at(rawRead) * gain);
            ++rawRead;
            ++write;
        }
    }
    void append(int count) {
        for (int i = 0; i < count; ++i, ++produced) {
            if (produced < window || (done && produced - window >= target + lookahead)) continue;
            for (int c = 0; c < channels; ++c) raw[c].at(rawWrite) = output[c][i];
            const long long position = std::clamp(rawWrite + window / 2, 0LL, std::max(0LL, received - 1));
            ceiling.at(rawWrite) = float(std::min(.999, std::sqrt(2.0) * peakPrefix.at(position)));
            ++rawWrite;
            publish();
        }
    }
    void process() {
        engine->process(ip.data(), packet, op.data(), packet);
        append(packet);
        fill = 0;
    }

   public:
    void configure(int sr, int ch, int w) {
        sampleRate = sr;
        channels = ch;
        window = 4 * std::max(4, int(std::lround((w * 5.0 / 4) * sr / 48000.0)) / 4);
        engine.reset(new pbshift_speech_vendor::stretch::SignalsmithStretch<float>(0));
        // Dense overlap stabilizes the spectrum during phonetic transitions.
        engine->configure(ch, window, window / 16, false);
        engine->setFormantFactor(1, true);
        engine->setFormantBase(0);
        engine->setPitchDetectionRange(float(60.0 / sr), float(1100.0 / sr));
        input.assign(ch, std::vector<float>(packet));
        output.assign(ch, std::vector<float>(std::max(packet, window / 2)));
        ip.resize(ch);
        op.resize(ch);
        ring.resize(ch);
        raw.resize(ch);
        cap = 65536;
        for (int c = 0; c < ch; ++c) {
            ip[c] = input[c].data();
            op[c] = output[c].data();
            ring[c].init(cap);
            raw[c].init(256);
        }
        peakPrefix.init(65536);
        ceiling.init(256);
        reset();
    }
    void reset() {
        engine->reset();
        fill = 0;
        received = produced = rawWrite = rawRead = write = readPos = target = 0;
        inputMaximum = 0;
        gain = 1;
        done = false;
        for (auto& r : ring) r.clear();
        for (auto& r : raw) r.clear();
        peakPrefix.clear();
        ceiling.clear();
    }
    void setPitch(double p) { engine->setTransposeSemitones(float(p)); }
    void setStretch(double) {}
    void allowExternalExtrapolation() {}
    void feed(const float* const* in, int n) {
        for (int i = 0; i < n; ++i) {
            for (int c = 0; c < channels; ++c) {
                input[c][fill] = in[c][i];
                inputMaximum = std::max(inputMaximum, std::abs(double(in[c][i])));
            }
            peakPrefix.at(received++) = float(inputMaximum);
            if (++fill == packet) process();
        }
    }
    void finish(long long exact) {
        target = exact;
        done = true;
        long long pad = window / 2 + lookahead;
        while (pad > 0 || fill > 0) {
            for (int c = 0; c < channels; ++c) input[c][fill] = 0;
            peakPrefix.at(received++) = float(inputMaximum);
            if (++fill == packet) process();
            --pad;
        }
        engine->flush(op.data(), window / 2);
        append(window / 2);
    }
    int analysisWindow() const {return window;}
    int available() const { return int(std::min<long long>(write - readPos, 1 << 30)); }
    int read(float* const* out, int n) {
        n = std::min(n, available());
        for (int c = 0; c < channels; ++c)
            for (int i = 0; i < n; ++i) out[c][i] = ring[c].at(readPos + i);
        readPos += n;
        return n;
    }
};
}  // namespace pbshift
