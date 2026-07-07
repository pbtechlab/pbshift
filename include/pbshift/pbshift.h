// pbshift — real-time pitch shift / time stretch library
// Public API. See docs/ARCHITECTURE.md.
#pragma once
#include <memory>

namespace pbshift {

struct Config {
    int sampleRate = 48000;
    int channels = 2;
    enum class Mode { Auto, Voice, Rhythm, Music };
    Mode mode = Mode::Auto;
    enum class Tier { Live, StudioRT, Offline };
    Tier tier = Tier::StudioRT;
};

// Pull-model streaming stretcher:
//   feed(input) -> available() -> read(output); finish() flushes the tail.
// Deterministic: same input + settings => bit-identical output.
class Stretcher {
public:
    Stretcher();
    ~Stretcher();
    Stretcher(const Stretcher&) = delete;
    Stretcher& operator=(const Stretcher&) = delete;

    void configure(const Config& cfg);
    void reset();

    // Time-stretch ratio = output_duration / input_duration, [0.25, 4].
    void setTimeStretch(double ratio);
    // Pitch shift in semitones, [-24, +24].
    void setPitchSemitones(double semitones);
    // Formant handling (M6).
    void setFormantPreserve(bool enable);

    // Push input samples ([channel][frame], non-interleaved).
    void feed(const float* const* in, int frames);
    // Signal end of input; remaining tail becomes available.
    void finish();
    // Finalized output frames ready to read.
    int available() const;
    // Read up to `frames` output frames; returns frames actually written.
    int read(float* const* out, int frames);

    // Latency in samples (for host PDC when used as a real-time insert).
    int inputLatency() const;
    int outputLatency() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace pbshift
