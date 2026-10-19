#pragma once
#include "pb_speech_pitch.h"
#include "pb_speech_time.h"
namespace pbshift {
class SpeechProcessor {
    SpeechPitchProcessor pitch;
    SpeechTimeProcessor time;
    bool pitched = false;

   public:
    void configure(int sr, int ch, int budget) {
        pitch.configure(sr, ch, budget);
        time.configure(sr, ch, budget);
        pitched = false;
    }
    void reset() {
        pitch.reset();
        time.reset();
        pitched = false;
    }
    void setPitch(double p) {
        pitched = std::abs(p) > 1e-12;
        pitch.setPitch(p);
    }
    void setStretch(double a) {
        pitch.setStretch(a);
        time.setStretch(a);
    }
    void feed(const float* const* in, int n) {
        if (pitched)
            pitch.feed(in, n);
        else
            time.feed(in, n);
    }
    void finish(long long n) {
        if (pitched)
            pitch.finish(n);
        else
            time.finish(n);
    }
    int pitchWindowSamples() const {return pitch.analysisWindow();}
    int available() const { return pitched ? pitch.available() : time.available(); }
    int read(float* const* out, int n) { return pitched ? pitch.read(out, n) : time.read(out, n); }
    void allowExternalExtrapolation() {
        if (pitched)
            pitch.allowExternalExtrapolation();
        else
            time.allowExternalExtrapolation();
    }
    void releaseFundamentalBefore(long long source) {
        if (!pitched) time.releaseFundamentalBefore(source);
    }
    double inputFundamental(long long source) const { return pitched ? 0 : time.inputFundamental(source); }
};
}  // namespace pbshift
