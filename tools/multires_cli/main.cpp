// Offline multi-resolution time-stretch test CLI.
//   multires in.wav out.wav --stretch <ratio> [--voice] [--nopin]
//                           [--scales "N:lo:hi,N:lo:hi,..."]
// --scales / --nopin / --voice also read from env PBMR_SCALES / PBMR_NOPIN /
// PBMR_VOICE so a single build can be swept over many layouts.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "pb_multires.h"
#include "../common/wav_io.h"

static std::vector<pbshift::MultiResStretch::Scale> parseScales(const char* s) {
    std::vector<pbshift::MultiResStretch::Scale> out;
    std::string str(s);
    size_t i = 0;
    while (i < str.size()) {
        size_t comma = str.find(',', i);
        std::string tok = str.substr(i, comma - i);
        int n = 0;
        double lo = 0, hi = 0;
        if (std::sscanf(tok.c_str(), "%d:%lf:%lf", &n, &lo, &hi) == 3)
            out.push_back({n, lo, hi});
        if (comma == std::string::npos) break;
        i = comma + 1;
    }
    return out;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: multires in.wav out.wav --stretch r "
                     "[--voice] [--nopin] [--scales N:lo:hi,...]\n");
        return 1;
    }
    double stretch = 1.0;
    bool nopin = std::getenv("PBMR_NOPIN") != nullptr;
    bool voice = std::getenv("PBMR_VOICE") != nullptr;
    const char* scalesArg = std::getenv("PBMR_SCALES");
    int hopDiv = std::getenv("PBMR_HOPDIV") ? std::atoi(std::getenv("PBMR_HOPDIV")) : 4;
    for (int i = 3; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--stretch") && i + 1 < argc)
            stretch = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--nopin"))
            nopin = true;
        else if (!std::strcmp(argv[i], "--voice"))
            voice = true;
        else if (!std::strcmp(argv[i], "--scales") && i + 1 < argc)
            scalesArg = argv[++i];
        else if (!std::strcmp(argv[i], "--hopdiv") && i + 1 < argc)
            hopDiv = std::atoi(argv[++i]);
    }

    pbwav::AudioFile in;
    std::string err;
    if (!pbwav::readWav(argv[1], in, &err)) {
        std::fprintf(stderr, "read %s: %s\n", argv[1], err.c_str());
        return 1;
    }

    std::vector<pbshift::MultiResStretch::Scale> scales;
    bool pin = !nopin;
    if (scalesArg && *scalesArg) {
        scales = parseScales(scalesArg);
    } else if (voice) {
        scales = pbshift::MultiResStretch::voicedScales(stretch);
        pin = false;                       // voiced: no transient pinning
    } else {
        // content-adaptive: percussion -> single window, tone -> long window,
        // tonal mix -> multi-resolution split (detected on channel 0)
        bool autoPin = true;
        scales = pbshift::MultiResStretch::autoScales(
            in.samples.empty() ? std::vector<float>() : in.samples[0],
            in.sampleRate, &autoPin, stretch);
        pin = autoPin && !nopin;
    }

    pbshift::MultiResStretch mr(in.sampleRate, scales, pin, hopDiv);
    pbwav::AudioFile out;
    out.sampleRate = in.sampleRate;
    out.channels = in.channels;
    out.samples = mr.process(in.samples, stretch);
    if (!pbwav::writeWavFloat32(argv[2], out, &err)) {
        std::fprintf(stderr, "write %s: %s\n", argv[2], err.c_str());
        return 1;
    }
    return 0;
}
