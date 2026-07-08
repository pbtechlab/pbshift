// pbshift offline CLI (benchmark harness convention):
//   pbshift in.wav out.wav --pitch <semitones> --stretch <out/in ratio> [--formant]
//
// Pitch = front resampler (Kaiser-sinc, offline quality) + engine stretch
// (resampler-then-stretch order) until the streaming resampler moves inside
// the engine (M3).
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "pbshift/pbshift.h"
#include "../common/wav_io.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: pbshift in.wav out.wav [--pitch st] [--stretch r] [--formant]\n");
        return 1;
    }
    std::string inPath = argv[1], outPath = argv[2];
    double pitch = 0.0, stretch = 1.0;
    bool formant = false;
    bool voiceMode = false;
    bool multiMode = false;
    std::string tier;
    for (int i = 3; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--pitch") && i + 1 < argc) pitch = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--stretch") && i + 1 < argc) stretch = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--formant")) formant = true;
        else if (!std::strcmp(argv[i], "--tier") && i + 1 < argc) tier = argv[++i];
        else if (!std::strcmp(argv[i], "--voice")) voiceMode = true;
        else if (!std::strcmp(argv[i], "--multi")) multiMode = true;
    }

    pbwav::AudioFile wav;
    std::string err;
    if (!pbwav::readWav(inPath, wav, &err)) {
        std::fprintf(stderr, "cannot read %s: %s\n", inPath.c_str(), err.c_str());
        return 1;
    }
    const int ch = wav.channels;
    const long long n = static_cast<long long>(wav.frames());

    // The engine owns the whole pitch/stretch pipeline (internal streaming
    // resamplers, placed pre- or post-engine depending on direction).
    std::vector<std::vector<float>>& x = wav.samples;
    const long long nIn = n;
    const long long outTarget = llround(n * stretch);

    pbshift::Config cfg;
    cfg.sampleRate = wav.sampleRate;
    cfg.channels = ch;
    if (tier == "live") cfg.tier = pbshift::Config::Tier::Live;
    else if (tier == "offline") cfg.tier = pbshift::Config::Tier::Offline;
    if (voiceMode) cfg.mode = pbshift::Config::Mode::Voice;
    else if (multiMode) cfg.mode = pbshift::Config::Mode::Music;  // general transient-adaptive
    pbshift::Stretcher st;
    st.configure(cfg);
    st.setTimeStretch(stretch);
    st.setPitchSemitones(pitch);
    st.setFormantPreserve(formant);

    std::vector<std::vector<float>> y(ch, std::vector<float>(outTarget, 0.0f));
    std::vector<const float*> inPtr(ch);
    std::vector<float*> outPtr(ch);
    long long fed = 0, got = 0;
    const int CHUNK = 8192;
    while (fed < nIn) {
        const int k = static_cast<int>(std::min<long long>(CHUNK, nIn - fed));
        for (int c = 0; c < ch; ++c) inPtr[c] = x[c].data() + fed;
        st.feed(inPtr.data(), k);
        fed += k;
        const int avail = st.available();
        if (avail > 0 && got < outTarget) {
            const int want = static_cast<int>(std::min<long long>(avail, outTarget - got));
            for (int c = 0; c < ch; ++c) outPtr[c] = y[c].data() + got;
            got += st.read(outPtr.data(), want);
        }
    }
    st.finish();
    while (got < outTarget) {
        const int avail = st.available();
        if (avail <= 0) break;
        const int want = static_cast<int>(std::min<long long>(avail, outTarget - got));
        for (int c = 0; c < ch; ++c) outPtr[c] = y[c].data() + got;
        const int r = st.read(outPtr.data(), want);
        if (r <= 0) break;
        got += r;
    }

    pbwav::AudioFile out;
    out.sampleRate = wav.sampleRate;
    out.channels = ch;
    out.samples = y;
    if (!pbwav::writeWavFloat32(outPath, out, &err)) {
        std::fprintf(stderr, "cannot write %s: %s\n", outPath.c_str(), err.c_str());
        return 1;
    }
    return 0;
}
