// pbshift engine core: streaming scheduler around the shared analysis
// front-end, RTPGHI phase engine and WOLA synthesis.
//
// M2 scope: time-stretch via RTPGHI with stereo reference-channel phase
// locking (inter-channel phase deltas copied verbatim from the input).
// Pitch shifting is done by the CLI via the front resampler (M3) until the
// resampler moves inside the engine.
#include "pbshift/pbshift.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstring>
#include <deque>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <vector>

#include "pb_envelope.h"
#include "pb_speech_periodicity.h"
#include "pb_fft.h"
#include "pb_multires.h"
#include "pb_voice.h"
#include "pb_prepared_queue.h"
#include "pb_speech_processor.h"
#include "pb_pghi.h"
#include "pb_resampler.h"
#include "pb_lpc.h"
#include "pb_stft.h"
#include "pb_window.h"

namespace pbshift {

namespace {
int nextPow2(int v) {
    int n = 32;
    while (n < v) n <<= 1;
    return n;
}
}  // namespace

struct Stretcher::Impl {
    Config cfg;
    int N = 4096;      // FFT / window size
    int hs = 1024;     // synthesis hop (output samples)
    int baseHs = 1024; // configured hop before deep-compression refinement
    double alpha = 1.0;      // engine (WOLA) stretch = alphaUser * pitchFactor
    double alphaUser = 1.0;  // user-facing time stretch
    double pitchSemi = 0.0;
    bool formant = false;

    // pitch resampler: pre-stage for pitch down (slows articulation into the
    // engine), post-stage for pitch up (avoids compressing articulation)
    std::unique_ptr<SincResampler> resIn, resOut;
    std::vector<std::vector<float>> resBuf;
    std::vector<float*> resPtr;
    std::vector<const float*> cPtr;

    // ---- output-side ratio scheduling (pitch automation) -----------------
    // alpha (the engine stretch) changes on the INPUT side: it applies to the
    // frames analysed from here on. postRatio() changes on the OUTPUT side.
    // Applying both the moment setPitchSemitones() is called puts the new
    // output ratio on audio that was stretched with the OLD alpha -- the two
    // are separated by the engine's latency -- and the mismatch accumulates
    // into a growing timing error. So the output ratio is queued and applied
    // where that audio actually reaches the output resampler.
    struct PostChange {
        long long atInput;  // resOut input position (frames fed to resOut)
        double ratio;
    };
    std::deque<PostChange> pendingPost;
    bool postActive = false;   // output resampler in the signal path
    double curPost = 1.0;      // ratio currently applied by resOut
    long long postBase = 0;    // engine output position where resOut started

    // ---- offline multi-resolution path ---------------------------------
    // Offline Music selects this whole-input path by default;
    // PBSHIFT_MULTIRES also opts other Offline modes into it. It is
    // buffered and time-stretched by MultiResStretch on finish(): a long
    // window for the low band (frequency resolution, kills chorusing) and a
    // short window for the high band (transient time resolution). Pitch is
    // still handled by the pre/post resamplers exactly as in the streaming
    // path, so this only replaces the "stretch" middle stage. Live/StudioRT
    // remain on the streaming engine; Offline Music enables this by default.
    bool useMultiRes = false;
    std::vector<std::vector<float>> mrIn;   // buffered (pre-resampled) input
    std::vector<std::vector<float>> mrOut;  // stretched (post-resampled) output
    long long mrReadPos = 0;
    bool mrDone = false;
    // Formant preservation is not implemented in the multires path; when it is
    // requested, fall back to the streaming engine (which has it). Evaluated
    // lazily because setFormantPreserve() is called after configure().
    bool mrActive() const { return useMultiRes && !formant; }

    // Pitch-synchronous voice engine. Streaming and causal, so unlike the
    // multi-resolution path it does not have to buffer the whole signal; it is
    // opt-in per mode and never touches any other mode's output.
    VoiceStretch voice;
    bool useVoice = false;
    bool vsActive() const { return useVoice && !formant && !nearBuffered; }
    SpeechProcessor autoVoiceEngine;
    std::unique_ptr<SpeechPeriodicity> autoVoicePitchProbe;
    double autoVoiceInitialPitch=0;
    double speechEvidence=0;long long speechEvidenceSource=-1;bool speechLatched=false;double speechPeakEnergy=0,speechQuietSeconds=0;
    bool autoVoiceConfigured=false,autoVoiceInitialized=false,autoVoiceFinished=false;
    bool autoVoiceAbandoned=false;long long autoVoiceFadeEnd=0;
    long long autoVoiceDiscard=0;
    std::vector<Ring<float>> autoVoiceRawHistory;
    Ring<float> autoVoiceNativeMono,autoVoiceAuxMono;
    int autoVoiceDelay=0;double autoVoicePeriod=0;bool autoVoiceAlignPending=false,autoVoiceReady=false;long long autoVoiceAlignRetryAt=0;int autoVoiceDamageRun=0;
    bool nativeVoiceDamaged(long long at) {
        const double sourceF0=autoVoiceEngine.inputFundamental(std::llround(at/autoVoiceMapRatio));
        if(sourceF0<80 || sourceF0>500)return false;
        autoVoicePeriod=cfg.sampleRate/sourceF0;
        const int period=std::clamp(int(std::lround(autoVoicePeriod)),8,600),span=3*period;
        if(at<span+int(std::ceil(1.2*period)))return false;
        double periodicity=-1;
        for(int lag=std::max(2,int(.8*period));lag<=int(std::ceil(1.2*period));lag+=2){double dot=0,a2=0,b2=0;for(int i=0;i<span;i+=2){const double a=autoVoiceNativeMono.at(at-i),b=autoVoiceNativeMono.at(at-i-lag);dot+=a*b;a2+=a*a;b2+=b*b;}periodicity=std::max(periodicity,dot/std::sqrt(a2*b2+1e-30));}
        if(periodicity>=2.0/3)return false;
        double auxiliaryPeriodicity=-1;
        for(int lag=std::max(2,int(.8*period));lag<=int(std::ceil(1.2*period));lag+=2){double dot=0,a2=0,b2=0;for(int i=0;i<span;i+=2){const double a=autoVoiceAuxMono.at(at-i),b=autoVoiceAuxMono.at(at-i-lag);dot+=a*b;a2+=a*a;b2+=b*b;}auxiliaryPeriodicity=std::max(auxiliaryPeriodicity,dot/std::sqrt(a2*b2+1e-30));}
        return auxiliaryPeriodicity>=.8;
    }
    void alignAutoVoice(long long at) {
        const int period=std::clamp(int(std::lround(autoVoicePeriod)),8,600),span=3*period;
        if(at<span+period)return;
        double referenceEnergy=0;for(int i=0;i<span;i+=2){const double a=autoVoiceNativeMono.at(at-i);referenceEnergy+=a*a;}
        int best=0;double similarity=-1;
        for(int delay=0;delay<=period;++delay){double dot=0,energy=0;for(int i=0;i<span;i+=2){const double a=autoVoiceNativeMono.at(at-i),b=autoVoiceAuxMono.at(at-i-delay);dot+=a*b;energy+=b*b;}const double value=dot/std::sqrt(referenceEnergy*energy+1e-30);if(value>similarity){similarity=value;best=delay;}}
        if(similarity>.6)autoVoiceDelay=best;
    }
    float autoVoiceMix=0;
    bool autoVoiceTarget=false,autoVoiceSupported=false,autoVoiceSupportPending=false,autoVoiceSupportObserved=false;
    int autoVoiceSupportRun=0;
    struct AutoVoiceFeature { long long source;std::array<double,8> bands;double fundamental; };
    PreparedQueue<AutoVoiceFeature> autoVoiceFeatures;
    std::vector<std::array<double,8>> autoVoiceBands;
    long long autoVoiceModulationUntil=-1;
    struct AutoVoiceSupportChange { long long at; bool supported;double period=0; };
    PreparedQueue<AutoVoiceSupportChange> autoVoiceSupportChanges;
    std::vector<float> autoVoiceMagnitude,autoVoiceResolutionTime,autoVoiceResolutionWindow,autoVoiceResolutionMagnitude;
    std::vector<std::complex<float>> autoVoiceResolutionSpectrum;
    std::unique_ptr<RealFFT> autoVoiceResolutionFFT;
    long double autoVoiceTimeline=0,autoVoiceAnchorOutput=0;
    long long autoVoiceInputPosition=0,autoVoiceAnchorInput=0;
    double autoVoiceMapRatio=1;
    long long autoVoiceOutputRead=0,autoVoiceExactEnd=0;
    struct AutoVoiceMixChange { long long at; bool wanted; };
    PreparedQueue<AutoVoiceMixChange> autoVoiceChanges;
    void scheduleAutoVoiceMix() {
        if(!autoVoiceInitialized || autoVoiceAbandoned)return;
        if(autoVoiceWanted() && alphaUser==autoVoiceMapRatio && pitchSemi==autoVoiceInitialPitch)return;
        autoVoiceAbandoned=true;
        const long long at=std::llround(finalized()*std::pow(2.0,-autoVoiceInitialPitch/12.0));
        autoVoiceFadeEnd=at+static_cast<long long>(std::ceil(.02*cfg.sampleRate));
        autoVoiceChanges.push_back({at,false});autoVoiceEngine.allowExternalExtrapolation();
    }
    std::vector<std::vector<float>> autoVoiceBuffer;
    std::vector<float*> autoVoicePointers;
    bool autoVoiceWanted() const {
        return !autoVoiceAbandoned && autoVoiceConfigured && cfg.mode==Config::Mode::Auto && formant && ((std::abs(pitchSemi)<1e-12 && alphaUser>=3.5 && alphaUser<=4) || (std::pow(2.0,pitchSemi/12.0)*autoVoiceEngine.pitchWindowSamples()>=2.0*N && alphaUser==1));
    }
    void discardAutoVoice() {
        while(autoVoiceDiscard>0 && autoVoiceEngine.available()>0) {
            const int n=int(std::min<long long>({autoVoiceDiscard,autoVoiceEngine.available(),8192}));
            const long long first=autoVoiceOutputRead-autoVoiceDiscard;
            const int actual=autoVoiceEngine.read(autoVoicePointers.data(),n);
            if(!autoVoiceAbandoned)for(int i=0;i<actual;++i){double mono=0;for(int ch=0;ch<cfg.channels;++ch){autoVoiceRawHistory[ch].at(first+i)=autoVoiceBuffer[ch][i];mono+=autoVoiceBuffer[ch][i]/cfg.channels;}autoVoiceAuxMono.at(first+i)=float(mono);}
            autoVoiceDiscard-=actual;
        }
    }


    // ---- content-adaptive near-unity offline path ----------------------
    // A single-resolution phase vocoder needlessly reconstructs every sample
    // when duration differs by only a few percent. Offline Auto/Voice/Music is
    // quality-first MultiRes; explicit Rhythm uses endpoint-anchored WSOLA for
    // broadband/transient material and falls back to MultiRes for tonal input.
    // Multichannel phase stays locked by one shared trajectory/reference.
    std::vector<std::vector<float>> nearIn;
    std::vector<std::vector<float>> nearOut;
    long long nearReadPos = 0;
    bool nearBuffered = false;
    bool nearDone = false;
    bool nearDisabledForStream = false;
    double nearRatio = 1.0;
    bool nearUnityEligible() const {
        return !nearDisabledForStream && cfg.tier == Config::Tier::Offline &&
               !formant && std::abs(pitchSemi) < 1e-12 &&
               alphaUser >= 0.97 && alphaUser <= 1.05;
    }

    double pitchFactor() const { return std::pow(2.0, pitchSemi / 12.0); }
    // Share (log domain) of a pitch-down done by the pre-engine resampler;
    // the rest is done by the post resampler. A -24 st split (sqrt(p) each)
    // was used to keep low harmonics resolvable, but with the N/4 analysis
    // hop cap and onset gating, all-pre measures better at -24 st on every
    // pbACC axis (fail reasons 15 cases: 28 -> 22, FR-QUAL A2 tone 1.46 ->
    // 1.71, speech NISQA noisiness 2.42 -> 2.89). PBSHIFT_DOWNPRE overrides.
    double downPreFrac() const {
        return 1.0;
    }
    double preRatio() const {  // input resampler ratio (>1 lengthens)
        if (pitchSemi >= 0.0) return 1.0;
        return std::pow(1.0 / pitchFactor(), downPreFrac());
    }
    double postRatio() const {  // output resampler ratio
        if (pitchSemi > 0.0) return 1.0 / pitchFactor();
        return std::pow(1.0 / pitchFactor(), 1.0 - downPreFrac());
    }
    void updateAlpha() {
        alpha = std::clamp(alphaUser * pitchFactor(), 0.02, 64.0);
        // deep compression: keep the ANALYSIS hop <= N/2 or magnitude
        // evolution gets temporally aliased (content skipped between frames).
        // Preserve the finer Music/Rhythm base hop selected by configure();
        // starting unconditionally at N/4 silently disabled Multi mode whenever
        // setTimeStretch() or setPitchSemitones() was called (the normal API
        // sequence), and also expanded its supposedly short synthesis window
        // back to the full analysis-window length.
        if (synthCount == 0) {
            int h = baseHs;
            while (alpha < 0.5 && h / alpha > N / 2 && h > N / 16) h /= 2;
            // Compression (incl. every pitch-down) also keeps the analysis hop
            // <= N/4: at N/2 an attack is seen by only two analysis frames
            // that land a synthesis hop apart = doubled onsets (pbACC
            // FR-ONSET fail cases 135 -> 130 over the 308-case matrix).
            static constexpr int aDiv = 4;
            if (aDiv > 0) while (h / alpha > N / aDiv && h > N / 16) h /= 2;
            hs = h;
        }
        if (!win.w.empty()) buildShortSyn();   // refresh short-window geometry
    }

    // Transient short synthesis window (4*hs Hann, centered on N/2) and its
    // analysis*synthesis product. The dual sig/norm WOLA needs norm += w_ana*w_syn
    // (NOT w_syn^2) to reconstruct exactly through a non-default window.
    void buildShortSyn() {
        shortSyn.assign(N, 0.0f);
        shortProd.assign(N, 0.0f);
        const int L = std::min(N, 4 * hs);   // 4x synthesis overlap
        const int off = (N - L) / 2;         // centered on N/2
        for (int i = 0; i < L; ++i) {
            const float w = 0.5f * (1.0f - std::cos(6.28318530717958648f * i / L));
            shortSyn[off + i] = w;
            shortProd[off + i] = win.w[off + i] * w;   // analysis x synthesis
        }
    }

    // input ring (per channel), absolute indexing
    std::vector<std::vector<float>> inRing;
    int inCap = 0;
    long long totalIn = 0;
    bool finished = false;
    long long padAppended = 0;

    // analysis
    WindowSet win;
    std::vector<std::unique_ptr<AnalysisFrontend>> fe;
    std::vector<AnalysisFrame> frames;
    std::unique_ptr<Rtpghi> pghi;
    std::vector<float> synthPhase;

    // synthesis
    std::vector<std::unique_ptr<WolaSynth>> wola;
    long long synthCount = 0;   // frames synthesized
    double inPos = 0.0;         // input center of next analysis frame
    long long readPos = 0;      // absolute output read position
    int refCh = 0;

    std::vector<float> frameBuf;
    std::vector<std::complex<float>> Y;

    // ---- staged frame pipeline (split computation) -------------------
    // One frame's synthesis is split into 3*C+1 stages: analyze per
    // channel, formant per channel, ONE atomic transient/phase/advance
    // decision, render (Y + iFFT + WOLA) per channel. Each stage has a
    // due point that is a pure function of totalIn (never call boundaries
    // or wall time), so worst-case per-feed() cost approaches the average
    // while the output stays bit-identical for any input chunking.
    bool fActive = false;      // a frame is mid-flight
    int fStage = 0;            // next stage to execute
    long long fCIn = 0;        // frame center (input samples)
    long long fDueBase = 0;    // fCIn + N/2 = input point the frame starts
    long long fSpread = 0;     // due points spread over this input span
    double fBestE = -1.0;      // reference-channel election accumulator
    int fBest = 0;
    bool fDoFormant = false;   // per-frame snapshot of the formant path
    double fP = 1.0;           // pitch factor snapshot for gains / audible band
    int fEffOrder = 0;         // TrueEnvelope order snapshot
    float fFmtStrength = 1.0f; // formant-correction strength snapshot
    double fAdvance = 0.0;     // input advance decided by the atomic stage
    long long fCOut = 0;       // output center for the render stages

    // transient engine (M4)
    std::vector<float> prevRefMag;   // previous analysis frame magnitudes (ref ch)
    std::vector<std::vector<float>> prevMagCh;  // per-channel prev magnitudes
    std::vector<uint8_t> resetMask;
    float prevFlux = 0.0f;
    double debt = 0.0;               // input samples consumed ahead of schedule
    double lastOnsetIn = -1e18;      // input position of last accepted onset
    int pendCount = 0;               // frames spent holding a coming attack
    int pinRemain = 0;               // remaining ratio-1 pinned frames
    float tauMean = 0.0f;            // mag^2-weighted rise-bin group delay
    float lastFlux = 0.0f;           // detector observables of current frame
    float lastRiseFrac = 0.0f;

    // noisiness estimator (M5a, Damskagg-style fuzzy classification)
    static constexpr int kHist = 5;
    std::vector<std::vector<float>> magHist;  // ring of last kHist ref-mags
    int histCount = 0;
    std::vector<float> jitterAmt;

    // Multi mode (Music/Rhythm / PBSHIFT_MULTI): transient-adaptive synthesis
    // window. Analysis stays full-N (tonal frequency resolution); output frames
    // are deposited through a narrow 4*hs Hann so attacks are time-localized
    // instead of being smeared over the full analysis window.
    std::vector<float> shortSyn, shortProd;  // short synth window + (ana*syn) product
    bool fShortWin = false;                  // per-frame gate (set in stageDecision)

    // ---- onset-gated grain deposition (transient relocation) ----------
    // Identity-locked grains keep each attack at its in-frame offset tau, so
    // with alpha != 1 every frame that sees an attack places a copy at
    // o_k + tau_k, which drifts from frame to frame by (alpha-1)*(c_k-t_a):
    // pre-echo when expanding (pitch up / stretch), extra onsets when
    // compressing (pitch down). One owner frame (centre nearest the attack)
    // deposits the attack; the others are windowed off on the wrong side of
    // it. The dual sig/norm WOLA keeps the result exactly normalized.
    // (Transient relocation, Nagel & Walther AES 2009; detector after Bello
    // et al. 2005.) Detection is a pure function of absolute input position.
    struct Onset {
        long long pos = 0;     // engine-input sample of the attack
        bool owned = false;
        long long outPos = 0;  // output position the owner placed it at
        long long ownerK = -1; // frame index of the owner
    };
    static constexpr int kOnsetCap = 64;
    Onset onsets[kOnsetCap];
    int onsetHead = 0, onsetCount = 0;   // ring
    int odBlk = 48;                      // detector block (1 ms)
    long long odScan = 0;                // next input sample to scan
    double odAcc = 0.0;                  // block energy accumulator
    std::vector<float> odPrev;           // previous sample per channel
    // Look-back blocks. Must exceed the longest pitch period the engine can
    // see (after a -24 st pre-resample a 110 Hz tone has a 36 ms period):
    // with 16 ms every glottal/saw pulse of a low tone re-triggered
    // (A2 at -12 st: 165 false onsets -> gated every period = 3 Hz AM).
    static constexpr int kOdHist = 60;
    double odHist[kOdHist] = {};
    long long odBlocks = 0;
    long long odLastOnset = -(1LL << 60);
    bool gateOn = true;
    std::vector<float> fGate, gateSyn, gateProd, wprodFull;
    std::vector<uint8_t> allReset;
    bool fGated = false;
    bool inMask = true;
    bool postSpan = true;
    static constexpr long long kNoOwner = std::numeric_limits<long long>::min();
    long long fOwnTau = kNoOwner;   // in-frame offset of the attack this frame owns
    bool multiMode = false;                  // latched in configure()

    // formant engine (M6)
    std::vector<std::unique_ptr<TrueEnvelope>> tenv;
    std::vector<float> envLog, fmtGain;
    std::vector<std::vector<float>> envSmooth;  // per-channel EMA state
    // Envelope temporal smoothing (EMA weight of the previous frame). It was
    // ON at 0.85 to damp per-frame envelope flutter, but a frame period is
    // ~28 ms, so 0.85 is a ~180 ms lag: the correction followed the average
    // envelope, not the current syllable, and speech formants stayed shifted
    // (pbACC FR-FORMANT -475 c at -5 st, -26 c without smoothing; FR-QUAL
    // and NISQA noisiness also better without). PBSHIFT_TE_SMOOTH re-enables.
    float envLambda = 0.0f;
    int teOrder = 256;       // base cepstral order (sweep-optimal at N=4096)

    void configure(const Config& c) {
        cfg = c;
        // Window length by tier. Auto/Voice round-trip latency is 1.5 N;
        // Music/Rhythm use the fine hop below and therefore report 1.25 N.
        //   Live      ~2048 @48k -> ~64 ms  (meets the <100 ms insert spec)
        //   StudioRT  ~4096 @48k -> ~128 ms (default, max real-time quality)
        //   Offline   ~8192 @48k -> ~256 ms (best bass resolution)
        double winSec = 0.085;  // -> 4096 @48k
        if (cfg.tier == Config::Tier::Live) winSec = 0.040;      // -> 2048
        else if (cfg.tier == Config::Tier::Offline) winSec = 0.17;  // -> 8192
        // Offline Music is the public "Multi" quality path: select the true
        // multi-resolution renderer by default.  The environment switch still
        // opts other modes into it for experiments/backward compatibility.
        useVoice = cfg.mode == Config::Mode::Voice;
        if (useVoice) {
            voice.configure(cfg.sampleRate, cfg.channels);
            voice.setStretch(alpha);
        }
        useMultiRes = cfg.tier == Config::Tier::Offline &&
                      (cfg.mode == Config::Mode::Music ||
                       std::getenv("PBSHIFT_MULTIRES") != nullptr);
        N = nextPow2(static_cast<int>(cfg.sampleRate * winSec));
        if (const char* f = std::getenv("PBSHIFT_FFT")) N = std::atoi(f);
        // Multi mode (Music/Rhythm / PBSHIFT_MULTI): general-purpose transient-
        // adaptive vocoder. It combines a long analysis window with a fine
        // synthesis hop and a dual short window. Music enables the short window
        // only for detected events; Rhythm keeps it active for every frame.
        // Spectrum-wide identity-locked coherent phase is used throughout
        // (Laroche & Dolson 1999). Latch the mode before hs so it selects the
        // finer grid; Auto/Voice retain the standard hop.
        multiMode = (cfg.mode == Config::Mode::Music ||
                     cfg.mode == Config::Mode::Rhythm) ||
                    std::getenv("PBSHIFT_MULTI") != nullptr;
        baseHs = multiMode ? N / 8 : N / 4; // multi: fine analysis/synthesis grid
        if (const char* h = std::getenv("PBSHIFT_HOPDIV")) {
            const int div = std::clamp(std::atoi(h), 1, N);
            baseHs = std::max(1, N / div);
        }
        hs = baseHs;
        autoVoiceConfigured=cfg.mode==Config::Mode::Auto && N/2>=2048 && (cfg.sampleRate==44100 || cfg.sampleRate==48000);
        autoVoicePitchProbe.reset();
        if(autoVoiceConfigured) {
            autoVoicePitchProbe.reset(new SpeechPeriodicity(N,cfg.sampleRate));
            autoVoiceEngine.configure(cfg.sampleRate,cfg.channels,N/2);
            autoVoiceFeatures.reserve(size_t(std::ceil(.1*cfg.sampleRate*20/baseHs))+4);autoVoiceSupportChanges.reserve(256);autoVoiceChanges.reserve(16);autoVoiceBuffer.assign(cfg.channels,std::vector<float>(8192));autoVoiceRawHistory.resize(cfg.channels);for(auto &history:autoVoiceRawHistory)history.init(4096);autoVoiceNativeMono.init(4096);autoVoiceAuxMono.init(4096);
            autoVoiceMagnitude.resize(N/2+1);
            autoVoiceResolutionFFT.reset(new RealFFT(N));autoVoiceResolutionTime.assign(N,0);autoVoiceResolutionWindow.assign(N,0);autoVoiceResolutionMagnitude.resize(N/2+1);autoVoiceResolutionSpectrum.resize(N/2+1);
            const int width=4*std::max(4,int(std::lround((N/2*5.0/4)*cfg.sampleRate/48000.0))/4);for(int i=0;i<N;++i){const double offset=i-N/2;if(std::abs(offset)<width/2.0)autoVoiceResolutionWindow[i]=float(.5+.5*std::cos(2*M_PI*offset/width));}
            autoVoiceBands.resize(N/2+1);
            for(int m=0;m<=N/2;++m) {
                const double octave=std::log2(std::max(1.0,double(m)*cfg.sampleRate/N)/60.0);
                for(int band=0;band<8;++band)autoVoiceBands[m][band]=std::max(0.0,1-std::abs(octave-band));
            }
            autoVoicePointers.resize(cfg.channels);
            for(int ch=0;ch<cfg.channels;++ch)autoVoicePointers[ch]=autoVoiceBuffer[ch].data();
        }
        win = WindowSet::hann(N);
        buildShortSyn();
        inCap = 1 << 19;
        while (inCap < 8 * N) inCap <<= 1;
        inRing.assign(cfg.channels, std::vector<float>(inCap, 0.0f));
        fe.clear();
        wola.clear();
        frames.resize(cfg.channels);
        for (int ch = 0; ch < cfg.channels; ++ch) {
            fe.emplace_back(new AnalysisFrontend(N, win));
            wola.emplace_back(new WolaSynth(N, win, inCap));
        }
        pghi.reset(new Rtpghi(N / 2 + 1));
        if (const char* t = std::getenv("PBSHIFT_TOL"))
            pghi->setTolerance(static_cast<float>(std::atof(t)));
        // per-channel phase state for partials the reference channel does not
        // carry and for stream ends (see chOwnOn, endTreatMode); multichannel
        // only
        chPghi.clear();
        if (cfg.channels > 1)
            for (int ch = 0; ch < cfg.channels; ++ch) {
                chPghi.emplace_back(new Rtpghi(N / 2 + 1));
                if (const char* t = std::getenv("PBSHIFT_TOL"))
                    chPghi.back()->setTolerance(static_cast<float>(std::atof(t)));
            }
        teOrder = 256 * N / 4096;
        if (const char* o = std::getenv("PBSHIFT_TE_ORDER")) teOrder = std::atoi(o);
        if (const char* l = std::getenv("PBSHIFT_TE_SMOOTH"))
            envLambda = static_cast<float>(std::atof(l));
        int teSub = 0;
        if (const char* s = std::getenv("PBSHIFT_TE_SUB")) teSub = std::atoi(s);
        tenv.clear();
        for (int ch = 0; ch < cfg.channels; ++ch)
            tenv.emplace_back(new TrueEnvelope(N, teOrder, teSub, cfg.sampleRate));
        envSmooth.assign(cfg.channels, {});
        Y.resize(N / 2 + 1);
        frameBuf.resize(N);
        resetState();
    }

    void resetState() {
        if(autoVoiceConfigured)autoVoiceEngine.reset();
        autoVoiceInitialized=autoVoiceFinished=false;autoVoiceDiscard=0;autoVoiceMix=0;autoVoiceAbandoned=false;autoVoiceFadeEnd=0;
        autoVoiceTarget=false;autoVoiceTimeline=autoVoiceAnchorOutput=0;autoVoiceInputPosition=autoVoiceAnchorInput=0;autoVoiceMapRatio=1;
        autoVoiceOutputRead=autoVoiceExactEnd=0;autoVoiceChanges.clear();speechEvidence=0;speechEvidenceSource=-1;speechLatched=false;speechPeakEnergy=0;speechQuietSeconds=0;if(autoVoicePitchProbe)autoVoicePitchProbe->reset();autoVoiceDelay=0;autoVoicePeriod=0;autoVoiceAlignPending=false;autoVoiceReady=false;autoVoiceAlignRetryAt=0;autoVoiceDamageRun=0;for(auto &history:autoVoiceRawHistory)history.clear();autoVoiceNativeMono.clear();autoVoiceAuxMono.clear();
        autoVoiceSupported=autoVoiceSupportPending=autoVoiceSupportObserved=false;autoVoiceSupportRun=0;autoVoiceSupportChanges.clear();autoVoiceFeatures.clear();autoVoiceModulationUntil=-1;
        if (useVoice) voice.reset();
        for (auto& r : inRing) std::fill(r.begin(), r.end(), 0.0f);
        for (auto& w : wola) w->clear();
        // per-stream formant-estimator state (the smoothed F0 period) and the
        // optional envelope smoothing: a reset stream starts like a fresh one
        for (auto& t : tenv) t->reset();
        for (auto& s : envSmooth) s.clear();
        pghi->reset();
        for (auto& p : chPghi) p->reset();
        {
            const int B = N / 2 + 1;
            const size_t C = chPghi.size();
            chPhase.assign(C, std::vector<float>(B, 0.0f));
            chGain.assign(C, std::vector<float>(B, 1.0f));
            chOwn.assign(C, std::vector<uint8_t>(B, 0));
            chOwnAny.assign(C, 0);
            chActual.assign(B, 0.0f);
            chPeakOwn.assign(B, 0);
        }
        totalIn = 0;
        finished = false;
        padAppended = 0;
        realEnd = 0;
        headReady = false;
        headExt.clear();
        tailExt.clear();
        tailUntrusted = false;
        endGlide = false;
        fEndTreat = false;
        synthCount = 0;
        inPos = 0.0;
        readPos = 0;
        refCh = 0;
        pendingPost.clear();
        postActive = postRatio() != 1.0;
        curPost = postRatio();
        postBase = 0;
        prevRefMag.clear();
        prevMagCh.assign(cfg.channels, {});
        resetMask.assign(N / 2 + 1, 0);
        prevFlux = 0.0f;
        debt = 0.0;
        lastOnsetIn = -1e18;
        pendCount = 0;
        pinRemain = 0;
        fShortWin = false;
        onsetHead = onsetCount = 0;
        odBlk = std::max(8, cfg.sampleRate / 1000);
        odScan = 0;
        odAcc = 0.0;
        odPrev.assign(cfg.channels, 0.0f);
        std::fill(odHist, odHist + kOdHist, 0.0);
        odBlocks = 0;
        odLastOnset = -(1LL << 60);
        fGate.assign(N, 1.0f);
        gateSyn.assign(N, 0.0f);
        gateProd.assign(N, 0.0f);
        wprodFull.resize(N);
        for (int i = 0; i < N; ++i) wprodFull[i] = win.w[i] * win.w[i];
        fGated = false;
        fOwnTau = kNoOwner;
        allReset.assign(N / 2 + 1, 1);
        magHist.assign(kHist, std::vector<float>(N / 2 + 1, 0.0f));
        histCount = 0;
        jitterAmt.assign(N / 2 + 1, 0.0f);
        glideMask.assign(N / 2 + 1, 0);   // sized here: no allocation per frame
        chirpGain.assign(N / 2 + 1, 1.0f);
        dcY.assign(static_cast<size_t>(cfg.channels) * kDcBins, {0.0f, 0.0f});
        dcPrevY.assign(kDcBins, {0.0f, 0.0f});
        dcEligible = dcPrevValid = dcPrevYValid = false;
        dcPsi = dcF0 = 0.0;
        fDcM = -1;
        fActive = false;
        fStage = 0;
        fCIn = fDueBase = fSpread = 0;
        fAligned = alignUsed = alPrevUsed = false;
        alRun = 0;
        alPrevC = 0;
        fAlignFrac = alPrevFrac = 0.0;
        fBestE = -1.0;
        fBest = 0;
        fDoFormant = false;
        fP = 1.0;
        fEffOrder = 0;
        fAdvance = 0.0;
        fCOut = 0;
        resIn.reset(new SincResampler(cfg.channels, 64, 12.0));
        resOut.reset(new SincResampler(cfg.channels, 64, 12.0));
        resBuf.assign(cfg.channels, std::vector<float>(8192));
        resPtr.resize(cfg.channels);
        cPtr.resize(cfg.channels);
        for (int c = 0; c < cfg.channels; ++c) resPtr[c] = resBuf[c].data();
        engineHanded = 0;
        mrIn.assign(cfg.channels, {});
        mrOut.assign(cfg.channels, {});
        mrReadPos = 0;
        mrDone = false;
        nearIn.assign(cfg.channels, {});
        nearOut.assign(cfg.channels, {});
        nearReadPos = 0;
        nearBuffered = false;
        nearDone = false;
        nearDisabledForStream = false;
        nearRatio = alphaUser;
        // Re-derive the hop after every configure/reset. This matters when a
        // ratio was changed during the previous stream: setters intentionally
        // do not retime an in-flight frame grid, but the next reset stream must
        // start with the correct mode base hop and deep-compression refinement.
        updateAlpha();
    }

    // ---- pitch-aware streaming wrappers -----------------------------
    void feedUser(const float* const* in, int nf) {
        if (nf > 0 && totalIn == 0) {
            // Keep the formant estimator and Voice path on their existing
            // source spectrum. Select the native interpolation profile once
            // per stream, before any unread input can be recoloured.
            const double passband = !formant && cfg.mode != Config::Mode::Voice
                                        ? 0.936 : 0.92;
            resIn->setInterpolationCutoff(passband);
            resOut->setInterpolationCutoff(passband);
        }

        if(nf>0 && totalIn==0 && autoVoiceWanted()) {
            autoVoiceInitialized=true;autoVoiceMix=1;autoVoiceTarget=true;autoVoiceMapRatio=alphaUser;
            autoVoiceEngine.setStretch(alphaUser);autoVoiceInitialPitch=pitchSemi;autoVoiceEngine.setPitch(pitchSemi);autoVoiceSupported=false;autoVoiceMix=0;
        }
        feedPrimary(in,nf);
        if(autoVoiceInitialized) {
            autoVoiceInputPosition+=nf;
            autoVoiceTimeline=autoVoiceAnchorOutput+(autoVoiceInputPosition-autoVoiceAnchorInput)*static_cast<long double>(autoVoiceMapRatio);
            autoVoiceEngine.feed(in,nf);
        }
    }
    void feedPrimary(const float* const* in,int nf) {
        if (preRatio() != 1.0) {
            // pre-resample (ratio > 1: more samples, lower pitch)
            resIn->feed(in, nf);
            const double ratio = preRatio();
            int made;
            do {
                made = resIn->process(resPtr.data(),
                                      static_cast<int>(resBuf[0].size()), ratio);
                if (made > 0) {
                    for (int c = 0; c < cfg.channels; ++c) cPtr[c] = resPtr[c];
                    feed(const_cast<const float* const*>(cPtr.data()), made);
                }
            } while (made == static_cast<int>(resBuf[0].size()));
        } else {
            feed(in, nf);
        }
    }

    void finishUser() {
        finishPrimary();
        if(autoVoiceInitialized && !autoVoiceFinished) {
            autoVoiceExactEnd=llround(static_cast<double>(autoVoiceTimeline));autoVoiceEngine.finish(autoVoiceExactEnd);autoVoiceFinished=true;
        }
    }
    void finishPrimary() {
        // A zero-length offline stream has no feed() call on which to latch
        // the path, so make the same decision here.
        if (!nearBuffered && totalIn == 0 && nearUnityEligible()) {
            nearBuffered = true;
            nearRatio = alphaUser;
        }
        if (preRatio() != 1.0) {
            resIn->finish();
            const double ratio = preRatio();
            int made;
            do {
                made = resIn->process(resPtr.data(),
                                      static_cast<int>(resBuf[0].size()), ratio);
                if (made > 0) {
                    for (int c = 0; c < cfg.channels; ++c) cPtr[c] = resPtr[c];
                    feed(const_cast<const float* const*>(cPtr.data()), made);
                }
            } while (made > 0);
        }
        if (nearBuffered) {
            if (preferNearMultiRes())
                runNearMultiRes();
            else
                runNearUnity();
            return;
        }
        if (vsActive()) {
            voice.finish();
            if (postActive) {
                drainVoiceToResOut();
                resOut->finish();
            }
            finished = true;
            return;
        }
        if (mrActive()) {
            runMultiRes();
            return;
        }
        realEnd = totalIn;
        finished = true;
        pump();
        if (postActive) {
            drainEngineToResOut();
            resOut->finish();
        }
    }

    // Queue the new output ratio at the output position where the audio that
    // the new alpha applies to will appear. The next frame synthesised with
    // the new alpha lands at synthCount * hs, so that is the switch point.
    // Called after every pitch change; a no-op while the stream stays at a
    // single pitch (the ratio never changes, nothing is queued).
    void schedulePostChange() {
        const double np = postRatio();
        // Settings made before the stream starts are not automation: apply
        // them straight away, exactly as before, so the initial latency and
        // the priming an insert host does stay unchanged.
        if (totalIn == 0 && synthCount == 0) {
            pendingPost.clear();
            postActive = np != 1.0;
            curPost = np;
            postBase = 0;
            engineHanded = 0;
            return;
        }
        if (!postActive) {
            if (np == 1.0) { curPost = 1.0; return; }  // resOut still not needed
            // Entering the resampled path mid-stream: everything up to readPos
            // was read straight from the WOLA accumulators, so resOut starts
            // from there, at unity, and switches to np at the queued point.
            postActive = true;
            postBase = readPos;
            engineHanded = readPos;
            curPost = 1.0;
            resOut->reset();
        }
        // The next frame synthesised under the new alpha lands at
        // synthCount * hs, so that is where its audio starts appearing.
        // (Measured: pushing the switch later -- by half a window, or by
        // carrying the not-yet-analysed input through the stretch -- moves the
        // click positions the wrong way in both ramp directions.)
        const long long at = std::max(0LL, synthCount * hs - postBase);
        if (!pendingPost.empty() && pendingPost.back().atInput >= at)
            pendingPost.back().ratio = np;  // same point: keep one entry
        else
            pendingPost.push_back({at, np});
    }

    // engine output finalized frames not yet handed to resOut
    long long engineHanded = 0;
    void drainVoiceToResOut() {
        for (;;) {
            const int room = static_cast<int>(resBuf[0].size());
            const int n = voice.read(resPtr.data(), room);
            if (n <= 0) return;
            for (int c = 0; c < cfg.channels; ++c) cPtr[c] = resPtr[c];
            resOut->feed(const_cast<const float* const*>(cPtr.data()), n);
        }
    }

    void drainEngineToResOut() {
        while (true) {
            long long fin = finalized();
            if (finished) fin = synthCount * hs + N / 2;
            const int n = static_cast<int>(
                std::min<long long>(fin - engineHanded,
                                    static_cast<long long>(resBuf[0].size())));
            if (n <= 0) return;
            for (int ch = 0; ch < cfg.channels; ++ch)
                wola[ch]->read(engineHanded, n, resPtr[ch]);
            engineHanded += n;
            for (int c = 0; c < cfg.channels; ++c) cPtr[c] = resPtr[c];
            resOut->feed(const_cast<const float* const*>(cPtr.data()), n);
        }
    }

    int availableUser() {
        const int n=availablePrimary();
        if(!autoVoiceInitialized)return n;
        discardAutoVoice();
        if(autoVoiceAbandoned) {
            const long long needed=std::min<long long>(n,std::max(0LL,autoVoiceFadeEnd-autoVoiceOutputRead));
            const long long ready=std::max(0LL,static_cast<long long>(autoVoiceEngine.available())-autoVoiceDiscard);
            return ready>=needed?n:int(std::min<long long>(n,ready));
        }
        if(autoVoiceFinished && autoVoiceOutputRead>=autoVoiceExactEnd) {
            autoVoiceMix=0;autoVoiceTarget=false;autoVoiceChanges.clear();
        }
        bool userMayWant=autoVoiceTarget,contentMayWant=autoVoiceSupported;
        for(const auto &change:autoVoiceChanges)if(change.wanted && change.at<autoVoiceOutputRead+n)userMayWant=true;
        for(const auto &change:autoVoiceSupportChanges)if(change.supported && change.at<autoVoiceOutputRead+n)contentMayWant=true;
        const bool needVoice=autoVoiceMix>0 || (userMayWant && contentMayWant);
        if(needVoice)
            return int(std::min<long long>(n,std::max(0LL,static_cast<long long>(autoVoiceEngine.available())-autoVoiceDiscard)));
        return n;
    }
    int readUser(float* const* out,int nf) {
        if(!autoVoiceInitialized)return readPrimary(out,nf);
        const int limit=std::min(nf,availableUser());
        if(limit<=0)return 0;
        const int n=readPrimary(out,limit);
        const float step=1.f/(.02f*cfg.sampleRate);
        int done=0;
        while(done<n) {
            const int count=std::min(8192,n-done);
            const int wanted=autoVoiceAbandoned?int(std::min<long long>(count,std::max(0LL,autoVoiceFadeEnd-autoVoiceOutputRead-done))):count;
            const int actual=autoVoiceEngine.read(autoVoicePointers.data(),wanted);
            autoVoiceDiscard+=wanted-actual;
            for(int i=0;i<count;++i) {
                const long long position=autoVoiceOutputRead+done+i;
                double nativeMono=0,auxMono=0;
                for(int ch=0;ch<cfg.channels;++ch){const float sample=i<actual?autoVoiceBuffer[ch][i]:0;autoVoiceRawHistory[ch].at(position)=sample;nativeMono+=out[ch][done+i]/cfg.channels;auxMono+=sample/cfg.channels;}
                autoVoiceNativeMono.at(position)=float(nativeMono);autoVoiceAuxMono.at(position)=float(auxMono);
                while(!autoVoiceChanges.empty() && autoVoiceChanges.front().at<=autoVoiceOutputRead+done+i) {
                    autoVoiceTarget=autoVoiceChanges.front().wanted;autoVoiceChanges.pop_front();
                }
                while(!autoVoiceSupportChanges.empty() && autoVoiceSupportChanges.front().at<=autoVoiceOutputRead+done+i) {
                    const auto change=autoVoiceSupportChanges.front();
                    if(change.supported && !autoVoiceSupported){autoVoiceAlignPending=true;autoVoiceReady=false;autoVoiceAlignRetryAt=0;autoVoiceDamageRun=0;autoVoicePeriod=change.period;}
                    autoVoiceSupported=change.supported;autoVoiceSupportChanges.pop_front();
                }
                if(autoVoiceOutputRead+done+i==0 && !autoVoiceSupported)autoVoiceMix=0;
                if(std::abs(autoVoiceInitialPitch)>1e-12){
                    if(autoVoiceAlignPending && autoVoiceMix<=step && autoVoiceSupported){alignAutoVoice(position);autoVoiceAlignPending=false;}
                    if(!autoVoiceSupported)autoVoiceAlignPending=false;
                    autoVoiceMix=(autoVoiceTarget && autoVoiceSupported)?std::min(1.f,autoVoiceMix+step):std::max(0.f,autoVoiceMix-step);
                }else{
                if(autoVoiceAlignPending && autoVoiceMix<=step && autoVoiceSupported && position>=autoVoiceAlignRetryAt){if(nativeVoiceDamaged(position))++autoVoiceDamageRun;else autoVoiceDamageRun=0;if(autoVoiceDamageRun>=2){alignAutoVoice(position);autoVoiceAlignPending=false;autoVoiceReady=true;}autoVoiceAlignRetryAt=position+std::max(1,int(std::ceil(3*autoVoicePeriod)));}
                if(!autoVoiceSupported){autoVoiceAlignPending=false;autoVoiceReady=false;autoVoiceDamageRun=0;}
                autoVoiceMix=(autoVoiceTarget && autoVoiceSupported && autoVoiceReady)?std::min(1.f,autoVoiceMix+step):std::max(0.f,autoVoiceMix-step);
                }
                if(autoVoiceAbandoned && autoVoiceOutputRead+done+i>=autoVoiceFadeEnd)autoVoiceMix=0;
                if(i>=actual)continue;
                for(int ch=0;ch<cfg.channels;++ch) {
                    if(autoVoiceMix>=1)out[ch][done+i]=autoVoiceRawHistory[ch].at(position-autoVoiceDelay);
                    else if(autoVoiceMix>0)out[ch][done+i]+=autoVoiceMix*(autoVoiceRawHistory[ch].at(position-autoVoiceDelay)-out[ch][done+i]);
                }
            }
            done+=count;
        }
        autoVoiceOutputRead+=n;
        if(std::abs(autoVoiceInitialPitch)<1e-12)autoVoiceEngine.releaseFundamentalBefore(std::llround(autoVoiceOutputRead/autoVoiceMapRatio));
        if(autoVoiceAbandoned && autoVoiceOutputRead>=autoVoiceFadeEnd) {
            autoVoiceInitialized=false;autoVoiceMix=0;autoVoiceTarget=false;autoVoiceDiscard=0;
            autoVoiceChanges.clear();autoVoiceSupportChanges.clear();
        }
        return n;
    }
    int availablePrimary() {
        if (nearBuffered) {
            if (!nearDone) return 0;
            return (int)std::min<long long>(
                1 << 30, (long long)nearOut[0].size() - nearReadPos);
        }
        if (vsActive() && !postActive) return voice.available();
        if (mrActive()) {
            if (!mrDone) return 0;
            return (int)std::min<long long>(
                1 << 30, (long long)mrOut[0].size() - mrReadPos);
        }
        if (postActive) {
            if (vsActive()) drainVoiceToResOut();
            else drainEngineToResOut();
            while (!pendingPost.empty() && resOut->outputsBefore(curPost, pendingPost.front().atInput) <= 0) {
                curPost = pendingPost.front().ratio;
                pendingPost.pop_front();
            }
            const int ready = resOut->readyOutputCount(curPost);
            return pendingPost.empty() ? ready : std::min(ready, resOut->outputsBefore(curPost, pendingPost.front().atInput));
        }
        long long f = finalized();
        if (finished) f = synthCount * hs + N / 2;
        return static_cast<int>(std::max(0LL, f - readPos));
    }

    int readPrimary(float* const* out, int nf) {
        if (nearBuffered) {
            if (!nearDone) return 0;
            const long long avail = (long long)nearOut[0].size() - nearReadPos;
            const int n = (int)std::min<long long>(nf, std::max(0LL, avail));
            for (int ch = 0; ch < cfg.channels; ++ch)
                std::memcpy(out[ch], nearOut[ch].data() + nearReadPos,
                            n * sizeof(float));
            nearReadPos += n;
            return n;
        }
        if (mrActive()) {
            if (!mrDone) return 0;
            const long long avail = (long long)mrOut[0].size() - mrReadPos;
            const int n = (int)std::min<long long>(nf, std::max(0LL, avail));
            for (int ch = 0; ch < cfg.channels; ++ch)
                std::memcpy(out[ch], mrOut[ch].data() + mrReadPos,
                            n * sizeof(float));
            mrReadPos += n;
            return n;
        }
        if (vsActive() && !postActive) return voice.read(out, nf);
        if (postActive) {
            if (vsActive()) drainVoiceToResOut();
            else drainEngineToResOut();
            // Produce in pieces so a queued ratio change lands exactly on the
            // audio it belongs to, instead of retroactively on what is already
            // in the resampler's queue.
            std::vector<float*> op(cfg.channels);
            int total = 0;
            while (total < nf) {
                int limit = nf - total;
                if (!pendingPost.empty()) {
                    const int room =
                        resOut->outputsBefore(curPost, pendingPost.front().atInput);
                    if (room <= 0) {
                        curPost = pendingPost.front().ratio;
                        pendingPost.pop_front();
                        continue;
                    }
                    limit = std::min(limit, room);
                }
                for (int ch = 0; ch < cfg.channels; ++ch) op[ch] = out[ch] + total;
                const int n = resOut->process(op.data(), limit, curPost);
                if (n <= 0) break;
                total += n;
            }
            return total;
        }
        const int n = std::min(nf, availablePrimary());
        for (int ch = 0; ch < cfg.channels; ++ch)
            wola[ch]->read(readPos, n, out[ch]);
        readPos += n;
        return n;
    }

    static float median5(float a, float b, float c, float d, float e) {
        float v[5] = {a, b, c, d, e};
        for (int i = 1; i < 5; ++i) {
            const float key = v[i];
            int j = i - 1;
            while (j >= 0 && v[j] > key) {
                v[j + 1] = v[j];
                --j;
            }
            v[j + 1] = key;
        }
        return v[2];
    }

    // Per-bin noisiness -> phase jitter amount (Damskagg & Valimaki 2017,
    // adapted): tonal bins have a high across-time median relative to the
    // across-frequency median; noisy bins the reverse. Jitter ramps up with
    // the effective stretch (alpha*, includes the pitch factor).
    // Harmonic-product-spectrum F0 estimate on the reference frame, restricted
    // to the speech/singing range. Returns the fundamental bin, or -1 if the
    // frame is not confidently voiced (harmonic-sum peak below a threshold).
    double estimateF0Bin(const AnalysisFrame& R) {
        const int B = N / 2 + 1;
        const double binHz = static_cast<double>(cfg.sampleRate) / N;
        const double f0lo = 70.0, f0hi = 500.0;
        const int lo = std::max(1, static_cast<int>(f0lo / binHz));
        const int hi = std::min(B / 4, static_cast<int>(f0hi / binHz) + 1);
        float maxMag = 0.0f;
        for (int m = 0; m < B; ++m) maxMag = std::max(maxMag, R.mag[m]);
        if (maxMag < 1e-8f) return -1.0;
        double bestScore = 0.0, meanScore = 0.0;
        int bestBin = -1;
        int cnt = 0;
        for (int f = lo; f <= hi; ++f) {
            // sum the first 6 harmonics (product-ish via log-sum for robustness)
            double s = 0.0;
            for (int h = 1; h <= 6; ++h) {
                const int mb = f * h;
                if (mb >= B) break;
                s += R.mag[mb];
            }
            meanScore += s;
            ++cnt;
            if (s > bestScore) { bestScore = s; bestBin = f; }
        }
        meanScore /= std::max(1, cnt);
        // voiced confidence: harmonic sum must dominate the mean substantially
        if (bestBin < 0 || bestScore < 2.2 * meanScore) return -1.0;
        // parabolic refine on the harmonic-sum curve neighbourhood: sub-bin F0
        // localization steadies the fundamental peak p1 in stepVoice (a bin of
        // F0 jitter jumps the whole harmonic stack -> warble).
        auto hsum = [&](int f) {
            double s = 0.0;
            for (int h = 1; h <= 6; ++h) {
                const int mb = f * h;
                if (mb >= B) break;
                s += R.mag[mb];
            }
            return s;
        };
        const double y0 = (bestBin > lo) ? hsum(bestBin - 1) : bestScore;
        const double y1 = bestScore;
        const double y2 = (bestBin < hi) ? hsum(bestBin + 1) : bestScore;
        const double denom = (y0 - 2.0 * y1 + y2);
        double frac = 0.0;
        if (std::abs(denom) > 1e-18) frac = 0.5 * (y0 - y2) / denom;
        frac = std::clamp(frac, -0.5, 0.5);
        return static_cast<double>(bestBin) + frac;
    }

    void pushMagHistory(const AnalysisFrame& R) {
        magHist[histCount % kHist].assign(R.mag.begin(), R.mag.end());
        ++histCount;
    }

    void computeJitter(const AnalysisFrame& R, double effAlpha) {
        const int B = N / 2 + 1;
        const float aTerm =
            0.5f * (std::tanh(4.0f * (static_cast<float>(effAlpha) - 1.5f)) + 1.0f);
        if (histCount < kHist || aTerm < 0.01f) {
            std::fill(jitterAmt.begin(), jitterAmt.end(), 0.0f);
            return;
        }
        const auto& h0 = magHist[(histCount - 1) % kHist];
        const auto& h1 = magHist[(histCount - 2) % kHist];
        const auto& h2 = magHist[(histCount - 3) % kHist];
        const auto& h3 = magHist[(histCount - 4) % kHist];
        const auto& h4 = magHist[(histCount - 5) % kHist];
        // frequency median over ~+-250 Hz (literature: 500 Hz window) —
        // wide enough that even dense harmonic combs stay below 50% fill
        const int half = std::max(8, static_cast<int>(250.0 * N / cfg.sampleRate));
        float maxMag = 0.0f;
        for (int m = 0; m < B; ++m)
            if (R.mag[m] > maxMag) maxMag = R.mag[m];
        const float lvlGate = maxMag * 0.005623f;  // -45 dB: leave the
        // inaudible deterministic floor (tone sidelobes) phase-locked
        std::vector<float> win;
        win.reserve(2 * half + 1);
        for (int m = 0; m < B; ++m) {
            if (R.mag[m] < lvlGate) {
                jitterAmt[m] = 0.0f;
                continue;
            }
            const int lo = std::max(0, m - half);
            const int hi = std::min(B - 1, m + half);
            win.assign(R.mag.begin() + lo, R.mag.begin() + hi + 1);
            const size_t mid = win.size() / 2;
            std::nth_element(win.begin(), win.begin() + mid, win.end());
            const float medF = win[mid];
            const float medT = median5(h0[m], h1[m], h2[m], h3[m], h4[m]);
            const float rn = medF / (medT + 1e-12f);
            // band-pass classification: rn ~ 1 = noise (jitter), rn << 1 =
            // tonal (locked), rn >> 1 = transient (locked, the transient
            // engine owns those bins)
            const float nTerm = 0.25f * (std::tanh(4.0f * (rn - 1.0f)) + 1.0f) *
                                (std::tanh(4.0f * (3.0f - rn)) + 1.0f);
            jitterAmt[m] = nTerm * aTerm;
        }
    }

    void feed(const float* const* in, int nf) {
        if (nearBuffered || (totalIn == 0 && !fActive && nearUnityEligible())) {
            if (!nearBuffered) {
                nearBuffered = true;
                nearRatio = alphaUser;  // ratio is constant within a stream
            }
            for (int ch = 0; ch < cfg.channels; ++ch)
                nearIn[ch].insert(nearIn[ch].end(), in[ch], in[ch] + nf);
            totalIn += nf;
            return;
        }
        if (vsActive()) {
            voice.setStretch(alpha);
            voice.feed(in, nf);
            totalIn += nf;
            if (postActive) drainVoiceToResOut();
            return;
        }
        if (mrActive()) {
            // buffer the (already pre-resampled) input; processed on finish
            for (int ch = 0; ch < cfg.channels; ++ch)
                mrIn[ch].insert(mrIn[ch].end(), in[ch], in[ch] + nf);
            totalIn += nf;
            return;
        }

        // Protect both fixed rings from a host that passes an entire file to
        // feed() before calling available()/read().  The input is admitted in
        // bounded pieces so pump() can consume it before wraparound, while the
        // WOLA accumulators are grown once, ahead of synthesis, to retain all
        // unread output.  External call boundaries still do not affect bits.
        const long long preserveFrom = postActive ? engineHanded : readPos;
        long long currentOutputEnd = 0;
        for (const auto& ws : wola)
            currentOutputEnd = std::max(currentOutputEnd, ws->highWater());
        // Base this on the CURRENT output high-water plus incremental work,
        // not absolute input * current ratio.  The latter under-reserved after
        // an in-flight high->low ratio change because the old 2x output epoch
        // had already advanced far beyond the recomputed 0.5x estimate.
        const long double futureIncrement =
            (static_cast<long double>(nf) + 8LL * N) *
                std::max(0.02, alpha) +
            4LL * N;
        const long double futureOutput =
            static_cast<long double>(currentOutputEnd) + futureIncrement;
        const long long requiredEnd = static_cast<long long>(
            std::min<long double>(
                futureOutput,
                static_cast<long double>(
                    std::numeric_limits<long long>::max())));
        for (auto& ws : wola)
            ws->reserveUnread(preserveFrom, requiredEnd);

        constexpr int kSafeFeed = 8192;
        int offset = 0;
        while (offset < nf) {
            const int k = std::min(kSafeFeed, nf - offset);
            for (int ch = 0; ch < cfg.channels; ++ch) {
                const float* s = in[ch] + offset;
                for (int i = 0; i < k; ++i)
                    inRing[ch][static_cast<size_t>((totalIn + i) % inCap)] = s[i];
            }
            totalIn += k;
            offset += k;
            scanOnsets();
            pump();
        }
    }

    // A parameter setter may legally arrive after the first feed().  If the
    // whole-signal near-unity path has already buffered audio, replay that
    // prefix through the regular engine with the OLD parameters before the
    // setter is applied.  The remainder of this stream then stays on the
    // regular path; reset() re-enables near-unity dispatch.  This preserves the
    // existing in-flight setter semantics instead of silently ignoring pitch,
    // stretch or formant changes.
    void fallbackNearToStreaming() {
        if (!nearBuffered || nearDone) return;
        std::vector<std::vector<float>> buffered = std::move(nearIn);
        const long long frames = buffered.empty()
                                     ? 0
                                     : (long long)buffered[0].size();
        resetState();
        nearDisabledForStream = true;
        std::vector<const float*> ptr(static_cast<size_t>(cfg.channels));
        long long fed = 0;
        while (fed < frames) {
            const int k = static_cast<int>(
                std::min<long long>(8192, frames - fed));
            for (int ch = 0; ch < cfg.channels; ++ch)
                ptr[static_cast<size_t>(ch)] = buffered[ch].data() + fed;
            feedUser(ptr.data(), k);
            fed += k;
        }
    }

    // Resample every channel of `in` by `ratio` through `rs` (streaming
    // feed/process/finish), returning the fully drained result.
    std::vector<std::vector<float>> resampleAll(
        SincResampler* rs, const std::vector<std::vector<float>>& in,
        double ratio) {
        std::vector<std::vector<float>> out(cfg.channels);
        const long long n = in.empty() ? 0 : (long long)in[0].size();
        const int CH = 8192;
        std::vector<std::vector<float>> buf(cfg.channels,
                                            std::vector<float>(CH * 2));
        std::vector<const float*> ip(cfg.channels);
        std::vector<float*> op(cfg.channels);
        auto drain = [&]() {
            int made;
            do {
                for (int c = 0; c < cfg.channels; ++c) op[c] = buf[c].data();
                made = rs->process(op.data(), CH * 2, ratio);
                if (made > 0)
                    for (int c = 0; c < cfg.channels; ++c)
                        out[c].insert(out[c].end(), buf[c].begin(),
                                      buf[c].begin() + made);
            } while (made == CH * 2);
        };
        long long fed = 0;
        while (fed < n) {
            const int k = (int)std::min<long long>(CH, n - fed);
            for (int c = 0; c < cfg.channels; ++c) ip[c] = in[c].data() + fed;
            rs->feed(ip.data(), k);
            fed += k;
            drain();
        }
        rs->finish();
        drain();
        return out;
    }

    // Shared-channel raw-waveform similarity against the already normalized
    // OLA overlap.  Each audible channel is normalized independently.
    double wsolaScore(const std::vector<std::vector<double>>& accum,
                      const std::vector<double>& norm, long long outPos,
                      long long inPos, int overlap, int stride) const {
        std::vector<double> scores(static_cast<size_t>(cfg.channels), 0.0);
        std::vector<double> levels(static_cast<size_t>(cfg.channels), 0.0);
        double maxLevel = 0.0;
        for (int ch = 0; ch < cfg.channels; ++ch) {
            double ab = 0.0, aa = 0.0, bb = 0.0;
            const auto& a = accum[ch];
            const auto& b = nearIn[ch];
            for (int i = 0; i < overlap; i += stride) {
                const size_t oi = static_cast<size_t>(outPos + i);
                const double av = norm[oi] > 1e-20 ? a[oi] / norm[oi] : 0.0;
                const double bv = b[static_cast<size_t>(inPos + i)];
                ab += av * bv;
                aa += av * av;
                bb += bv * bv;
            }
            const double rawDen = std::sqrt(aa * bb);
            const double raw = rawDen > 1e-20 ? ab / rawDen : 0.0;
            scores[static_cast<size_t>(ch)] = raw;
            levels[static_cast<size_t>(ch)] = std::max(aa, bb);
            maxLevel = std::max(maxLevel, levels[static_cast<size_t>(ch)]);
        }

        // Normalize each audible channel independently before averaging.  A
        // loud centre channel must not be allowed to hide a bad splice in a
        // quieter ambience/rear channel.  The relative -60 dB energy gate
        // rejects only channels that are effectively digital silence.
        const double gate = std::max(1e-20, maxLevel * 1e-6);
        double sum = 0.0;
        int active = 0;
        for (int ch = 0; ch < cfg.channels; ++ch) {
            if (levels[static_cast<size_t>(ch)] >= gate) {
                sum += scores[static_cast<size_t>(ch)];
                ++active;
            }
        }
        return active > 0 ? sum / active : 0.0;
    }

    bool wsolaTransient(long long start, int length) const {
        for (int ch = 0; ch < cfg.channels; ++ch) {
            double energy = 0.0;
            double peak = 0.0;
            const auto& x = nearIn[ch];
            for (int i = 0; i < length; ++i) {
                const double v = x[static_cast<size_t>(start + i)];
                energy += v * v;
                peak = std::max(peak, std::abs(v));
            }
            const double rms = std::sqrt(energy / std::max(1, length));
            if (peak > 1e-5 && peak > 6.0 * (rms + 1e-15)) return true;
        }
        return false;
    }

    bool preferNearMultiRes() const {
        if (nearIn.empty() ||
            llround(nearIn[0].size() * nearRatio) ==
                static_cast<long long>(nearIn[0].size()))
            return false;

        // A short clip has no safe WSOLA grid. Resampling it by endpoint
        // interpolation changes pitch by the stretch ratio, so every
        // non-trivial short render goes through the pitch-preserving spectral
        // renderer regardless of the requested content mode.
        if (nearIn[0].size() < static_cast<size_t>(cfg.sampleRate / 20))
            return true;

        // Offline Auto/Voice are quality-first whole-signal modes.  Keep the
        // absolute-grid WSOLA branch for explicit Rhythm/broadband work, where
        // transient timing is its measured advantage; this prevents an
        // unclassified low tonal bed from ever taking the risky path.
        if (cfg.mode != Config::Mode::Rhythm) return true;

        std::vector<double> energies(static_cast<size_t>(cfg.channels), 0.0);
        double bestEnergy = 0.0;
        for (int ch = 0; ch < cfg.channels; ++ch) {
            for (float s : nearIn[ch])
                energies[static_cast<size_t>(ch)] +=
                    static_cast<double>(s) * s;
            bestEnergy = std::max(bestEnergy,
                                  energies[static_cast<size_t>(ch)]);
        }
        // Absolute-grid WSOLA is strongest on broadband/transient mixtures.
        // Sustained tonal material exposes every splice as modulation; the
        // multi-resolution renderer is objectively cleaner there. Inspect
        // every audible channel independently so a loud broadband channel
        // cannot make a -40 dB tonal ambience/bass channel take the wrong path.
        for (int ch = 0; ch < cfg.channels; ++ch) {
            if (energies[static_cast<size_t>(ch)] < bestEnergy * 1e-6)
                continue;
            const double flat =
                MultiResStretch::spectralFlatness(nearIn[ch]);
            const double periodic =
                MultiResStretch::periodicity(nearIn[ch], cfg.sampleRate);
            if (flat < 0.02 && periodic > 0.40) return true;
        }
        return false;
    }

    int loudestChannel(
        const std::vector<std::vector<float>>& audio) const {
        int loudRef = 0;
        double loudEnergy = -1.0;
        for (int ch = 0; ch < static_cast<int>(audio.size()); ++ch) {
            double energy = 0.0;
            for (float s : audio[static_cast<size_t>(ch)])
                energy += static_cast<double>(s) * s;
            if (energy > loudEnergy) {
                loudEnergy = energy;
                loudRef = ch;
            }
        }
        return loudRef;
    }

    void rerenderQuietIndependentTonal(
        const std::vector<std::vector<float>>& input,
        std::vector<std::vector<float>>& output, int loudRef,
        double renderRatio, double layoutRatio) const {
        if (loudRef < 0 || loudRef >= static_cast<int>(input.size()) ||
            output.size() != input.size())
            return;

        std::vector<double> energies(input.size(), 0.0);
        for (size_t ch = 0; ch < input.size(); ++ch)
            for (float s : input[ch])
                energies[ch] += static_cast<double>(s) * s;
        const double loudEnergy = energies[static_cast<size_t>(loudRef)];
        if (loudEnergy <= 1e-20) return;

        // A tonal reference may have a legitimate low-level quadrature or
        // delayed partner whose zero-lag correlation is small. Keep that whole
        // stereo image locked; the independent fallback is only for a quiet
        // tonal bed underneath a broadband reference.
        if (MultiResStretch::spectralFlatness(
                input[static_cast<size_t>(loudRef)]) < 0.02 &&
            MultiResStretch::periodicity(
                input[static_cast<size_t>(loudRef)], cfg.sampleRate) > 0.40)
            return;

        // A single reference preserves a correlated stereo image, but it can
        // modulate an unrelated tonal bed that is tens of dB below a broadband
        // channel. Render only that narrow corner independently. Ordinary
        // stereo pairs (including opposite polarity) remain phase-locked.
        for (int ch = 0; ch < static_cast<int>(input.size()); ++ch) {
            if (ch == loudRef) continue;
            const double energy = energies[static_cast<size_t>(ch)];
            if (energy < loudEnergy * 1e-6 || energy >= loudEnergy * 0.01)
                continue;
            double dot = 0.0;
            const size_t count = std::min(input[static_cast<size_t>(ch)].size(),
                                          input[static_cast<size_t>(loudRef)].size());
            for (size_t i = 0; i < count; ++i)
                dot += static_cast<double>(input[static_cast<size_t>(ch)][i]) *
                       input[static_cast<size_t>(loudRef)][i];
            const double correlation = std::abs(dot) /
                std::sqrt(std::max(1e-30, energy * loudEnergy));
            if (correlation >= 0.5 ||
                MultiResStretch::spectralFlatness(
                    input[static_cast<size_t>(ch)]) >= 0.02 ||
                MultiResStretch::periodicity(
                    input[static_cast<size_t>(ch)], cfg.sampleRate) <= 0.40)
                continue;

            bool pin = true;
            std::vector<MultiResStretch::Scale> scales;
            if (cfg.mode == Config::Mode::Voice) {
                scales = MultiResStretch::voicedScales(layoutRatio);
                pin = false;
            } else {
                scales = MultiResStretch::autoScales(
                    input[static_cast<size_t>(ch)], cfg.sampleRate, &pin,
                    layoutRatio);
            }
            MultiResStretch renderer(cfg.sampleRate, scales, pin);
            std::vector<std::vector<float>> mono{
                input[static_cast<size_t>(ch)]};
            auto rendered = renderer.process(mono, renderRatio);
            if (rendered.size() != 1 ||
                rendered[0].size() != output[static_cast<size_t>(ch)].size())
                throw std::runtime_error(
                    "pbshift independent-channel length underflow");
            output[static_cast<size_t>(ch)] = std::move(rendered[0]);
        }
    }

    // Buffered Offline fallback for sustained tonal material.  The actual
    // multi-resolution renderer measures markedly cleaner than forcing WSOLA
    // period choices on low fundamentals.  Unlike the streaming-PV fallback,
    // it natively owns exact whole-signal length and cannot hide underflow by
    // returning a pre-zeroed tail.
    void runNearMultiRes() {
        const long long inputFrames = nearIn.empty()
                                          ? 0
                                          : (long long)nearIn[0].size();
        const long long target =
            std::max(0LL, llround(inputFrames * nearRatio));
        const int loudRef = loudestChannel(nearIn);
        bool pin = true;
        std::vector<MultiResStretch::Scale> scales;
        if (cfg.mode == Config::Mode::Voice) {
            scales = MultiResStretch::voicedScales(nearRatio);
            pin = false;
        } else {
            scales = MultiResStretch::autoScales(
                nearIn[loudRef], cfg.sampleRate, &pin, nearRatio);
        }
        MultiResStretch renderer(cfg.sampleRate, scales, pin);
        // MultiResStretch uses channel 0 for phase/onset decisions. Put the
        // loudest channel there temporarily: a silent left channel must not
        // make an active right channel inherit an undefined phase trajectory.
        if (loudRef != 0)
            std::swap(nearIn[0], nearIn[static_cast<size_t>(loudRef)]);
        try {
            nearOut = renderer.process(nearIn, nearRatio);
        } catch (...) {
            if (loudRef != 0)
                std::swap(nearIn[0], nearIn[static_cast<size_t>(loudRef)]);
            throw;
        }
        if (loudRef != 0) {
            std::swap(nearIn[0], nearIn[static_cast<size_t>(loudRef)]);
            if (nearOut.size() > static_cast<size_t>(loudRef))
                std::swap(nearOut[0],
                          nearOut[static_cast<size_t>(loudRef)]);
        }
        rerenderQuietIndependentTonal(nearIn, nearOut, loudRef, nearRatio,
                                      nearRatio);
        if (static_cast<int>(nearOut.size()) != cfg.channels)
            throw std::runtime_error("pbshift near-unity channel underflow");
        for (int ch = 0; ch < cfg.channels; ++ch) {
            auto& channel = nearOut[ch];
            if (static_cast<long long>(channel.size()) != target)
                throw std::runtime_error("pbshift near-unity length underflow");
            double inEnergy = 0.0, outEnergy = 0.0;
            for (float s : nearIn[ch]) inEnergy += static_cast<double>(s) * s;
            for (float s : channel) outEnergy += static_cast<double>(s) * s;
            if (inEnergy > 1e-20 && outEnergy > 1e-20) {
                const double gain = std::sqrt(
                    (inEnergy / inputFrames) / (outEnergy / target));
                for (float& s : channel) s = static_cast<float>(s * gain);
            }
        }
        nearReadPos = 0;
        nearDone = true;
        finished = true;
    }

    void runNearUnity() {
        const long long inputFrames = nearIn.empty()
                                          ? 0
                                          : (long long)nearIn[0].size();
        const long long target =
            std::max(0LL, llround(inputFrames * nearRatio));
        nearOut.assign(cfg.channels, {});
        for (auto& y : nearOut) y.reserve(static_cast<size_t>(target));

        if (inputFrames == 0 || target == 0) {
            nearDone = true;
            finished = true;
            return;
        }

        // This is the continuity anchor: unity (and ratios whose rounded
        // target is identical) is a literal copy, not a reconstruction.
        if (target == inputFrames) {
            nearOut = nearIn;
            nearDone = true;
            finished = true;
            return;
        }

        const int nominalWindow = std::clamp(
            static_cast<int>(llround(cfg.sampleRate * 0.040)), 32, 32768);
        if (inputFrames <= nominalWindow || target <= nominalWindow ||
            target < 2) {
            // Defensive fallback if this function is called directly: never
            // turn a time stretch into pitch-changing interpolation.
            runNearMultiRes();
            return;
        }

        // Endpoint-anchored absolute-grid WSOLA.  Synthesis starts are spread
        // uniformly rather than appending a tiny remainder frame, which gives
        // both ends a protected ~10 ms single-frame region.  Analysis nominal
        // positions use the same endpoint affine map; similarity search is
        // always relative to that absolute grid and cannot accumulate drift.
        const int window = nominalWindow;
        const int nominalHop = std::clamp(
            static_cast<int>(llround(cfg.sampleRate * 0.010)), 8, 8192);
        const long long outSpan = target - window;
        const long long inSpan = inputFrames - window;
        const long long intervals = std::max(
            1LL, (outSpan + nominalHop - 1) / nominalHop);
        const int search = std::clamp(
            static_cast<int>(llround(cfg.sampleRate * 0.010)), 16, 8192);
        const int coarseStep = std::clamp(cfg.sampleRate / 12000, 1, 32);
        const int fineStride = 1;

        std::vector<double> win(static_cast<size_t>(window));
        for (int i = 0; i < window; ++i)
            win[static_cast<size_t>(i)] =
                0.5 - 0.5 * std::cos(2.0 * M_PI * (i + 0.5) / window);
        std::vector<std::vector<double>> accum(
            cfg.channels, std::vector<double>(static_cast<size_t>(target), 0.0));
        std::vector<double> norm(static_cast<size_t>(target), 0.0);

        long long previousIn = 0;
        long long previousOut = 0;
        long long covered = 0;
        for (long long k = 0; k <= intervals; ++k) {
            const long long outStart = llround(
                static_cast<long double>(k) * outSpan / intervals);
            const long long nominal = llround(
                static_cast<long double>(k) * inSpan / intervals);
            long long chosen = nominal;

            if (k != 0 && k != intervals) {
                long long lo = std::max(0LL, nominal - search);
                long long hi = std::min(inSpan, nominal + search);
                // Preserve source order.  With near-unity ratios the absolute
                // corridor always has ample room; this guard also makes odd
                // tiny/sample-rate configurations deterministic.
                const long long minAdvance = std::max(
                    1LL, (outStart - previousOut) / 4);
                if (previousIn + minAdvance <= hi)
                    lo = std::max(lo, previousIn + minAdvance);
                hi = std::min(hi, previousIn + window - 1);

                const int overlap = static_cast<int>(
                    std::max(0LL, std::min(covered, outStart + window) -
                                      outStart));
                if (wsolaTransient(nominal, window)) {
                    // Coded impulses/percussion stay on the uniform map so a
                    // similarity offset cannot duplicate or skip the event.
                    chosen = std::clamp(nominal, lo, hi);
                } else {
                    long long best = std::clamp(nominal, lo, hi);
                    double bestScore = -1e30;
                    auto consider = [&](long long p, int stride) {
                        const double corr = wsolaScore(
                            accum, norm, outStart, p, overlap, stride);
                        // A modest absolute-grid prior breaks near-equal
                        // periodic matches in favour of correct time mapping,
                        // while still allowing a true waveform period to win.
                        const double score = corr -
                            0.03 * std::abs(p - nominal) /
                                std::max(1, search);
                        if (score > bestScore) {
                            bestScore = score;
                            best = p;
                        }
                    };
                    for (long long p = lo; p <= hi; p += coarseStep)
                        consider(p, coarseStep);
                    consider(hi, coarseStep);
                    consider(std::clamp(nominal, lo, hi), coarseStep);
                    const long long refineLo =
                        std::max(lo, best - 2LL * coarseStep);
                    const long long refineHi =
                        std::min(hi, best + 2LL * coarseStep);
                    bestScore = -1e30;
                    for (long long p = refineLo; p <= refineHi; ++p)
                        consider(p, fineStride);
                    chosen = best;
                }
            }

            for (int i = 0; i < window; ++i) {
                const size_t oi = static_cast<size_t>(outStart + i);
                const double w = win[static_cast<size_t>(i)];
                norm[oi] += w;
                for (int ch = 0; ch < cfg.channels; ++ch)
                    accum[ch][oi] +=
                        nearIn[ch][static_cast<size_t>(chosen + i)] * w;
            }
            covered = std::max(covered, outStart + window);
            previousIn = chosen;
            previousOut = outStart;
        }

        nearOut.assign(cfg.channels,
                       std::vector<float>(static_cast<size_t>(target), 0.0f));
        for (int ch = 0; ch < cfg.channels; ++ch)
            for (long long i = 0; i < target; ++i)
                nearOut[ch][static_cast<size_t>(i)] =
                    norm[static_cast<size_t>(i)] > 1e-20
                        ? static_cast<float>(
                              accum[ch][static_cast<size_t>(i)] /
                              norm[static_cast<size_t>(i)])
                        : 0.0f;
        // Make the protected single-frame head/tail literal even in the
        // presence of final float rounding.  Uniform frame spacing guarantees
        // this region is about one synthesis hop, never a one-sample remainder.
        const long long protectedFrames = std::min<long long>(
            {inputFrames, target,
             std::max(1LL, llround(
                 static_cast<long double>(outSpan) / intervals))});
        for (int ch = 0; ch < cfg.channels; ++ch) {
            for (long long i = 0; i < protectedFrames; ++i) {
                nearOut[ch][static_cast<size_t>(i)] =
                    nearIn[ch][static_cast<size_t>(i)];
                nearOut[ch][static_cast<size_t>(target - protectedFrames + i)] =
                    nearIn[ch][static_cast<size_t>(inputFrames -
                                                    protectedFrames + i)];
            }
        }

        nearDone = true;
        finished = true;
    }

    // Run the whole-signal multi-resolution stretch (Offline Music default,
    // or explicit opt-in for the other modes).
    void runMultiRes() {
        std::vector<MultiResStretch::Scale> scales;
        bool pin = true;
        const int loudRef = loudestChannel(mrIn);
        if (cfg.mode == Config::Mode::Voice) {
            scales = MultiResStretch::voicedScales(alphaUser);
            pin = false;
        } else {
            // content-adaptive: percussion -> single window, tone -> long
            // window, tonal mix -> multi-resolution (detected on the loudest
            // channel, which is also used as the phase/onset reference below)
            scales = MultiResStretch::autoScales(
                mrIn.empty() ? std::vector<float>()
                             : mrIn[static_cast<size_t>(loudRef)],
                cfg.sampleRate, &pin, alphaUser);
        }
        MultiResStretch mr(cfg.sampleRate, scales, pin);
        // engine's internal stretch = alphaUser * pitchFactor (the pre/post
        // resamplers convert the pitchFactor part into the actual pitch shift)
        const double engineStretch =
            std::clamp(alphaUser * pitchFactor(), 0.02, 64.0);
        if (loudRef != 0)
            std::swap(mrIn[0], mrIn[static_cast<size_t>(loudRef)]);
        std::vector<std::vector<float>> y;
        try {
            const bool isolated=cfg.tier==Config::Tier::Offline && cfg.mode==Config::Mode::Music && std::abs(pitchSemi)<1e-12 &&
                MultiResStretch::isolatedTransientStretch(mrIn,engineStretch,cfg.sampleRate,y);
            if(!isolated)y=mr.process(mrIn,engineStretch);
        } catch (...) {
            if (loudRef != 0)
                std::swap(mrIn[0], mrIn[static_cast<size_t>(loudRef)]);
            throw;
        }
        if (loudRef != 0) {
            std::swap(mrIn[0], mrIn[static_cast<size_t>(loudRef)]);
            if (y.size() > static_cast<size_t>(loudRef))
                std::swap(y[0], y[static_cast<size_t>(loudRef)]);
        }
        if (postRatio() != 1.0)
            mrOut = resampleAll(resOut.get(), y, postRatio());
        else
            mrOut = std::move(y);
        mrDone = true;
        finished = true;
    }

    // Time-domain onset detector on the engine input (after the pitch
    // pre-resampler). Per 1 ms block: first-difference energy summed over
    // channels (high-frequency emphasis); an onset fires when a block exceeds
    // every block of the previous 60 ms by riseDb and an absolute floor,
    // with a 20 ms refractory period. Runs on complete blocks only, so the
    // result depends on absolute position, never on feed() boundaries.
    void scanOnsets() {
        static constexpr double riseDb = 10.0;
        const double riseK = std::pow(10.0, riseDb / 10.0);
        const int C = cfg.channels;
        while (odScan < totalIn) {
            const long long p = odScan++;
            const size_t idx = static_cast<size_t>(p % inCap);
            for (int ch = 0; ch < C; ++ch) {
                const float x = inRing[ch][idx];
                const double d = static_cast<double>(x) - odPrev[ch];
                odAcc += d * d;
                odPrev[ch] = x;
            }
            if ((p + 1) % odBlk != 0) continue;
            const double e = odAcc;
            odAcc = 0.0;
            double prevMax = 0.0;
            for (double h : odHist) prevMax = std::max(prevMax, h);
            const long long blkStart = p + 1 - odBlk;
            const double floorE = 1e-7 * odBlk * C;   // ~-70 dB diff level
            if (odBlocks >= kOdHist && e > riseK * prevMax && e > floorE &&
                blkStart - odLastOnset >= cfg.sampleRate / 50) {
                odLastOnset = blkStart;
                if (onsetCount == kOnsetCap) {   // drop the oldest
                    onsetHead = (onsetHead + 1) % kOnsetCap;
                    --onsetCount;
                }
                onsets[(onsetHead + onsetCount) % kOnsetCap] = {blkStart, false, 0};
                if (debugOnsets) std::fprintf(stderr, "td-onset %.4f\n", blkStart / (double)cfg.sampleRate);
                ++onsetCount;
            }
            odHist[odBlocks % kOdHist] = e;
            ++odBlocks;
        }
    }

    // Build the deposition gate of the current frame (centre fCIn, output
    // centre fCOut, nominal input advance adv). Returns true if any gating.
    bool buildGate(double adv) {
        fOwnTau = kNoOwner;
        fAnaShift = 0;
        fHasPost = false;
        if (!gateOn || std::abs(alpha - 1.0) < 1e-9 || onsetCount == 0)
            return false;
        static constexpr double attackMs = 10.0;
        static constexpr double fadeMs = 2.0;
        const int half = N / 2;
        const long long c = fCIn, o = fCOut;
        const int fade = std::max(1, static_cast<int>(fadeMs * cfg.sampleRate / 1000.0));
        const int attack = alpha > 1.0
            ? static_cast<int>(attackMs * cfg.sampleRate / 1000.0) : 0;
        // drop onsets that no future frame can see
        // (an owned attack stays until no output span can reach it either)
        while (onsetCount > 0 &&
               onsets[onsetHead].pos < c - half - static_cast<long long>(4 * adv) &&
               (!onsets[onsetHead].owned || onsets[onsetHead].outPos <= o - half)) {
            onsetHead = (onsetHead + 1) % kOnsetCap;
            --onsetCount;
        }
        bool any = false;
        std::fill(fGate.begin(), fGate.end(), 1.0f);
        // keep > 0: keep offsets >= ed (after); keep < 0: keep offsets < ed
        auto applyEdge = [&](int keep, long long ed) {
            any = true;
            for (int i = 0; i < N; ++i) {
                const long long off = i - half;
                float g;
                if (keep > 0) {
                    if (off < ed) g = 0.0f;
                    else if (off < ed + fade)
                        g = 0.5f - 0.5f * std::cos(3.14159265f * (off - ed) / fade);
                    else g = 1.0f;
                } else {
                    if (off >= ed) g = 0.0f;
                    else if (off >= ed - fade)
                        g = 0.5f - 0.5f * std::cos(3.14159265f * (ed - off) / fade);
                    else g = 1.0f;
                }
                fGate[i] *= g;
            }
        };
        // Owner election and placement. An identity-locked grain carries the
        // attack at its in-frame offset tau, i.e. to o + tau, while the time
        // map puts it at o + alpha*tau: a placement error of (alpha-1)*tau,
        // up to (adv-hs)/2 (measured 7.5 ms early at -24 st, where it reads
        // as pre-echo and as a doubled hit). The owner is therefore analysed
        // at the shifted centre c + (1-alpha)*tau, which carries the attack
        // to exactly o + alpha*tau (the same point the coherent kernel of
        // the Music/Rhythm modes relocates every other frame's copy to). The
        // owner is chosen so that this shift only ever reaches back in time
        // (compressing: first frame at or after the attack; expanding: last
        // frame at or before it): no look-ahead beyond the frame's own input
        // span, so latency and determinism are unchanged.
        long long ce = c;
        if (place) {
            for (int j = 0; j < onsetCount; ++j) {
                Onset& on = onsets[(onsetHead + j) % kOnsetCap];
                const long long t = on.pos;
                if (on.owned || t <= c - half || t + odBlk > c + half) continue;
                const bool own = alpha < 1.0 ? c >= t
                                             : static_cast<double>(t - c) < adv;
                if (!own) continue;
                on.owned = true;
                on.ownerK = synthCount;
                if (fOwnTau == kNoOwner) {
                    fAnaShift = std::min<long long>(0, llround((1.0 - alpha) * (t - c)));
                    ce = c + fAnaShift;
                    fOwnTau = t - ce;
                }
                on.outPos = o + (t - ce);
                if (debugOnsets) std::fprintf(stderr, "  owner c=%.4f o=%lld tau=%lld shift=%lld\n", c / (double)cfg.sampleRate, o, t - c, fAnaShift);
            }
        }
        for (int j = 0; j < onsetCount; ++j) {
            Onset& on = onsets[(onsetHead + j) % kOnsetCap];
            const long long t = on.pos;
            if (place && on.owned && on.ownerK == synthCount)
                continue;   // owned by this frame
            // Compressing, the output grid runs slower than the input one: a
            // frame whose input span starts after the attack still reaches
            // back over the owner's placement in the output. Its content is
            // all later decay, which (averaged in by the normalized WOLA)
            // fills the gap before the attack and dilutes the attack itself.
            if (t <= ce - half) {
                if (postSpan && on.owned && on.outPos - o > -half) {
                    applyEdge(1, on.outPos - o);
                    fHasPost = true;
                }
                continue;
            }
            // visible only once its whole detection block lies inside the
            // frame's input span (deterministic regardless of chunking)
            if (t + odBlk > ce + half) continue;
            const long long tau = t - ce;
            if (debugOnsets) std::fprintf(stderr, "  frame c=%lld t=%lld owned=%d tot=%lld\n", c, t, (int)on.owned, totalIn);
            if (!place && !on.owned && static_cast<double>(c) >= t - adv / 2.0) {
                on.owned = true;            // this frame deposits the attack
                on.outPos = o + tau;
                if (fOwnTau == kNoOwner) fOwnTau = tau;
                if (debugOnsets) std::fprintf(stderr, "  owner c=%.4f o=%lld tau=%lld adv=%.0f hs=%d\n", c /(double)cfg.sampleRate, o, tau, adv, hs);
                continue;
            }
            if (on.owned) {
                // after the owner: keep only what follows the placed attack
                // (and, when expanding, skip this frame's own copy of it)
                applyEdge(1, std::max<long long>(tau + attack, on.outPos - o));
                fHasPost = true;
            } else {
                // before the owner: predict where the owner will place the
                // attack and keep only what precedes both copies
                long long Po;
                if (place) {
                    Po = llround(alpha * static_cast<double>(t - c));  // time map
                } else {
                    const double m = std::max(1.0, std::ceil((t - adv / 2.0 - c) / adv));
                    Po = static_cast<long long>(m * hs) + llround(t - (c + m * adv));
                }
                applyEdge(-1, std::min<long long>(tau, Po));
            }
        }
        return any;
    }
    // ---- end of a stream no predictor can continue (expansion) ----------
    // The tail padding lets the last analysis frames see a steady signal
    // where the edge can be continued (periodic, or a trusted linear
    // prediction). A glide or noise cannot: there the padding is a faded
    // burst of the prediction (foreign frequencies). Under expansion an
    // identity-locked frame places its input offset tau at o + tau, before
    // the time-mapped end alpha * tau_e for tau < alpha * tau_e, so the
    // padding and the cut land inside the output: they split a fast chirp's
    // spectrum into ripple lobes that the identity lock rotated apart (log
    // sweeps at 2x/4x stretch and +3 st: +2..+12 dB overshoots over the last
    // 50 ms, crossing 0 dBFS). When the stream ends on such a glide (the last
    // fully real frame rendered gliding-partial regions), its end keeps the
    // plain zero pad, and every frame that reaches past it is rendered as one
    // rigid rotation per channel (the propagated phase advance of the
    // channel's strongest peak, see Rtpghi::stepRigid): a phase-rotated copy
    // of the windowed input, whose envelope -- and that of any normalised
    // overlap-add of such frames -- cannot exceed the input's. (Other ends a
    // predictor cannot continue -- a decaying chord, noise -- keep the
    // existing handling: one rotation for many partials breaks each one's
    // continuity, measured as a -66 dB low-frequency "hum" at the end of a
    // stereo pad at +4 st.) Under compression the
    // padding lands only after the mapped end; the existing handling (the
    // spectral hold the cut triggers keeps the last full frame's levels, the
    // chirp path keeps the glide consistent) measured within a dB of the
    // ideal there, while a rigid end lost the glide path and faded ~3 dB.
    // Not on the formant path either: its per-frame envelope correction
    // reshapes the magnitudes, a rigid rotation is then no copy of the input
    // and the bound does not hold (a formant-corrected linear sweep at +5 st
    // reached 0 dBFS that way). Frames that reach past the end run only after
    // finish(), so none of this depends on feed() boundaries. Identity-lock
    // (Auto) path only.
    bool tailZero = true;
    bool endTreatOn = true;
    bool tailUntrusted = false;   // some channel's end could not be continued
    bool endGlide = false;        // ... and the stream ends on a glide
    bool fEndTreat = false;       // this frame reaches past such an end
    bool endTreatMode() const {
        return endTreatOn && alpha > 1.0 && !multiMode && !usePghi &&
               !forceCoherent && cfg.mode != Config::Mode::Voice && !forceVoice &&
               !(formant && pitchSemi != 0.0);
    }
    long long fAnaShift = 0;    // owner analysis-centre shift (<= 0)
    bool fHasPost = false;      // frame carries a keep-after (post-attack) edge
    int cohPreMask = 1;
    bool place = true;

    // Tail padding after finish(). Instead of zeros (which makes the last
    // frames see a half-empty window and fade a steady signal out), continue
    // the signal by Burg linear-prediction extrapolation of its last N samples
    // (a steady tone stays a steady tone through the final analysis frames;
    // noise-like material decays smoothly). Training span and order are fixed
    // functions of the data, so output stays chunk-independent. Beyond
    // edgeExt() samples the pad is zero.
    void appendZeros(int nf) {
        if (padAppended == 0 && edgeReflect) buildTailExt();
        for (int ch = 0; ch < cfg.channels; ++ch)
            for (int i = 0; i < nf; ++i) {
                const long long q = padAppended + i;  // 0-based pad index
                const float v = (edgeReflect && q < (long long)tailExt[ch].size())
                                    ? tailExt[ch][static_cast<size_t>(q)]
                                    : 0.0f;
                inRing[ch][static_cast<size_t>((totalIn + i) % inCap)] = v;
            }
        totalIn += nf;
        padAppended += nf;
    }
    bool edgeReflect = true;
    long long realEnd = 0;  // input length when finish() was called
    long long edgeExt() const { return 3LL * N; }
    static constexpr int kEdgeOrder = 48;
    std::vector<std::vector<float>> tailExt, headExt;
    bool headReady = false;

    void buildTailExt() {
        tailExt.assign(cfg.channels, {});
        // the last fully real frame (this runs on the first padded one)
        // rendered a glide: the edge is a gliding partial's (see endTreatMode)
        endGlide = pghi->chirpFound();
        const long long Lt = std::min<long long>(realEnd, N);
        if (Lt < 32) return;
        std::vector<float> tr(static_cast<size_t>(Lt));
        for (int ch = 0; ch < cfg.channels; ++ch) {
            for (long long i = 0; i < Lt; ++i)
                tr[static_cast<size_t>(i)] = inRing[ch][static_cast<size_t>(
                    (realEnd - Lt + i) % inCap)];
            tailExt[ch].assign(static_cast<size_t>(edgeExt()), 0.0f);
            // a gliding edge no predictor can continue keeps the zero pad and
            // marks the stream's end (see endTreatMode)
            if (!burgExtrapolate(tr.data(), static_cast<int>(Lt), kEdgeOrder,
                                 static_cast<int>(edgeExt()), tailExt[ch].data(),
                                 /*periodic=*/true,
                                 /*zeroUntrusted=*/tailZero && endTreatMode() && endGlide))
                tailUntrusted = true;
        }
    }

    // Head: backward extrapolation of the first N/2 samples (the span that is
    // guaranteed present when the first frame runs), giving x[-1], x[-2], ...
    void buildHeadExt() {
        headReady = true;
        headExt.assign(cfg.channels, {});
        const long long lim = finished ? realEnd : totalIn;
        const long long Lp = std::min<long long>(N / 2, lim);
        if (Lp < 32) return;
        std::vector<float> rev(static_cast<size_t>(Lp));
        for (int ch = 0; ch < cfg.channels; ++ch) {
            for (long long i = 0; i < Lp; ++i)
                rev[static_cast<size_t>(i)] =
                    inRing[ch][static_cast<size_t>(Lp - 1 - i)];
            headExt[ch].assign(static_cast<size_t>(N / 2 + 1), 0.0f);
            // runs inside a streaming feed(): no periodic lag search here
            burgExtrapolate(rev.data(), static_cast<int>(Lp), kEdgeOrder,
                            N / 2 + 1, headExt[ch].data(), /*periodic=*/false);
        }
    }

    // extract N input samples centered at integer position c (zeros outside)
    void extract(int ch, long long c, float* dst) {
        const long long start = c - N / 2;
        if (start < 0 && edgeReflect && !headReady) buildHeadExt();
        for (int i = 0; i < N; ++i) {
            const long long p = start + i;
            if (p >= 0 && p < totalIn) {
                dst[i] = inRing[ch][static_cast<size_t>(p % inCap)];
            } else if (p < 0 && edgeReflect &&
                       static_cast<size_t>(-p - 1) < headExt[ch].size()) {
                dst[i] = headExt[ch][static_cast<size_t>(-p - 1)];
            } else {
                dst[i] = 0.0f;
            }
        }
    }

    // Frame scheduler: activates a frame once its input span is available
    // and runs its stages as their due points are reached. Stage due
    // points are anchored to the frame's own input-availability point and
    // spread over one nominal frame period, so the whole-frame cost never
    // lands in a single feed() call in steady state. Everything here is a
    // pure function of totalIn and the audio content, so scheduling is
    // deterministic and chunk-size independent. Once finished, stages run
    // eagerly and the tail is zero-padded as before.
    void pump() {
        while (true) {
            if (!fActive) {
                const long long cIn = llround(inPos);
                if (cIn + N / 2 > totalIn) {
                    if (!finished) return;
                    // finished: pad zeros so the tail can be rendered
                    if (padAppended > static_cast<long long>((N / 2 + 2 * hs) / std::min(alpha, 1.0)) + N)
                        return;  // tail fully covered
                    appendZeros(hs);
                    continue;
                }
                beginFrame(cIn);
            }
            while (fActive) {
                if (!finished && totalIn < stageDue(fStage)) return;
                runStage();
            }
        }
    }

    int numStages() const { return 3 * cfg.channels + 1; }

    // due point of stage s: fDueBase + (s+1)/S of the spread window
    long long stageDue(int s) const {
        return fDueBase + static_cast<long long>(s + 1) * fSpread / numStages();
    }

    // ---- waveform-synchronous frame alignment (PVSOLA-style) -----------
    // A phase vocoder can only move a partial it resolves. After a deep
    // pitch-down pre-resample the harmonics of a low tone sit closer than
    // the analysis main lobe (A2 at -24 st: 27.5 Hz = 2.3 bins at N=4096),
    // several partials share one identity-locked peak region and get one
    // phase advance; across the 16 overlapping compressed frames the
    // non-peak harmonics cancel (measured: 2nd/4th harmonic -19/-23 dB
    // low, harmonic-to-residual 9 dB, waveform crest +4 dB to 0 dBFS).
    // For a locally periodic tone the exact answer needs no resolution at
    // all: move the analysis position back by up to one period (backward
    // only, so no extra look-ahead) to where the waveform continues the
    // previous frame's segment one synthesis hop later, and take that
    // frame's phases verbatim -- each frame is then a time-aligned copy of
    // the input (WSOLA, Verhelst & Roelands 1993; phase-vocoder/SOLA hybrid
    // after Moinet & Dutoit, "PVSOLA", DAFx 2011), with sub-sample
    // alignment by a linear phase term. Resolved harmonics (a period down
    // to N/8, partials up to 8 bins apart) gain from it too: the identity
    // lock keeps each one's level but propagates each one's phase on its
    // own, so the waveform shape (the relative phases of the harmonics) is
    // whatever the first frames gave it -- frames whose window reaches into
    // the stream head's padding -- and stays that way (A2 sawtooth at -12 st:
    // harmonic phases off by up to 0.35 rad for the whole file; aligned, the
    // output carries the input's shape exactly). Engaged only for sustained
    // (>= 3 frames), strongly periodic (autocorrelation >= 0.97) input whose
    // period is >= N/8, no onset nearby, compression only, and -- on entry --
    // an output that is still waveform-coherent with the input (polyphonic
    // material whose phase-locked output is not stays on the vocoder: a
    // splice there measured worse). Pure function of the input ring and the
    // WOLA accumulators -> chunking-independent.
    bool alignOn = true;
    bool fAligned = false;
    std::unique_ptr<RealFFT> alFft;
    float* alT = nullptr;
    std::vector<std::complex<float>> alTS, alSS, alP;
    std::vector<double> alCum;
    std::vector<double> alR;
    long long alPrevC = 0;     // analysis centre of the previous frame
    double fAlignFrac = 0.0;   // sub-sample part of the chosen offset
    double alPrevFrac = 0.0;
    int alRun = 0;             // consecutive frames qualifying for alignment
    bool alPrevUsed = false;   // previous frame was rendered aligned
    bool alignUsed = false;    // current frame took the aligned path
    // decimated pre-gate scratch
    std::unique_ptr<RealFFT> alFftD;
    float* alTD = nullptr;
    std::vector<std::complex<float>> alSD, alTDS, alPD;
    std::vector<float> alDec;
    std::vector<double> alCumD, alRD;
    ~Impl() {
        if (alT) pffft_aligned_free(alT);
        if (alTD) pffft_aligned_free(alTD);
    }

    // Normalized cross-correlation of the input segment starting at tStart
    // (length L) against the candidate segments x[cIn - d + jlo ...],
    // d = 0..Dmax, summed over channels (alSS = candidate spectra per
    // channel, alCum = running candidate energy).
    void alignCorr(long long tStart, int L, int Dmax, int M,
                   std::vector<double>& out, bool fromOutput = false) {
        std::fill(alP.begin(), alP.end(), std::complex<float>(0.0f, 0.0f));
        const int B2 = M / 2 + 1;
        double Et = 0.0;
        for (int ch = 0; ch < cfg.channels; ++ch) {
            if (fromOutput) wola[ch]->read(tStart, L, alT);
            for (int i = 0; i < M; ++i) {
                const float v = i >= L ? 0.0f
                    : fromOutput ? alT[i]
                    : inRing[ch][static_cast<size_t>((tStart + i) % inCap)];
                alT[i] = v;
                Et += static_cast<double>(v) * v;
            }
            alFft->forward(alT, alTS.data());
            const auto* S = alSS.data() + static_cast<size_t>(ch) * B2;
            for (int k = 0; k < B2; ++k) alP[k] += std::conj(alTS[k]) * S[k];
        }
        alFft->inverse(alP.data(), alT);   // r[l] = sum_i t[i] s[i + l]
        out.assign(Dmax + 1, 0.0);
        if (Et < 1e-10 * L) return;
        for (int d = 0; d <= Dmax; ++d) {
            const int l = Dmax - d;
            const double Es = alCum[l + L] - alCum[l];
            out[d] = alT[l] / std::sqrt(Et * Es + 1e-30);
        }
    }

    static double parabolic(const std::vector<double>& y, int i) {
        if (i <= 0 || i + 1 >= static_cast<int>(y.size())) return 0.0;
        const double den = y[i - 1] - 2.0 * y[i] + y[i + 1];
        if (den > -1e-12) return 0.0;
        return std::clamp(0.5 * (y[i - 1] - y[i + 1]) / den, -0.5, 0.5);
    }

    long long chooseAlign(long long cIn) {
        fAligned = false;
        const int prevRun = alRun;
        alRun = 0;
        static constexpr double thr = 0.97;
        static constexpr double aMax = 0.999;
        fAlignFrac = 0.0;
        if (!alignOn || alpha > aMax) return 0;
        // A low band that the previous frame rendered through the near-DC
        // partial model is one clean sinusoid: nothing is unresolved there,
        // and the model follows it exactly. Waveform alignment would only
        // move the analysis back by up to a period -- on a gliding tone that
        // reads earlier (lower) frequencies and drifts the phase (measured:
        // -1.5 rad over 0.25 s of the -12 st sweep).
        if (dcOn && dcPrevValid) return 0;
        // Search geometry is sized for the period range of the default
        // (StudioRT, ~85 ms) window even when the analysis window is
        // shorter (Live): there the unresolved range starts at shorter
        // periods and must still find periods up to ~43 ms. The segment end
        // never passes cIn + 3N/8, so no extra look-ahead is needed.
        const int Na = std::max(N, nextPow2(static_cast<int>(cfg.sampleRate * 0.085)));
        const int Dmax = Na / 2;
        const int jhi = 3 * N / 8, jlo = jhi - 3 * Na / 4;
        const int L = jhi - jlo;
        // target = natural continuation of the previous frame's analysis
        // segment one synthesis hop later (WSOLA criterion)
        const long long tc = alPrevC + hs;
        if (synthCount < 2 || cIn - Dmax + jlo < 0 || tc + jlo < 0 ||
            tc + jhi > totalIn)
            return 0;
        // no onset anywhere near: attacks belong to the gated/owner path.
        // Only onsets whose detection block lies inside the frame's input
        // span count: one further ahead may or may not be detected yet when
        // the frame starts, depending on how the host chunked the input (an
        // attack just past the span is seen by the next frame instead).
        const double adv = hs / alpha;
        for (int j = 0; j < onsetCount; ++j) {
            const long long t = onsets[(onsetHead + j) % kOnsetCap].pos;
            if (t > cIn - N / 2 - Dmax - static_cast<long long>(4 * adv) &&
                t + odBlk <= cIn + N / 2)
                return 0;
        }
        static constexpr int minP = 8;
        // 0) cheap pre-gate on a 4x box-decimated copy (runs every
        //    compressed frame, so it must cost little): strongly periodic
        //    with a long period, or nothing else is computed
        {
            const int Rd = 4, Ld = L / Rd, Dd = Dmax / Rd;
            int Md = 32;
            while (Md < Ld + Dd) Md <<= 1;
            const int Bd = Md / 2 + 1;
            if (!alFftD || alFftD->size() != Md) {
                if (alTD) pffft_aligned_free(alTD);
                alFftD.reset(new RealFFT(Md));
                alTD = RealFFT::alloc(Md);
                alSD.resize(Bd);
                alTDS.resize(Bd);
                alPD.resize(Bd);
            }
            const int SLd = Ld + Dd;
            alCumD.assign(SLd + 1, 0.0);
            alDec.assign(static_cast<size_t>(SLd) * cfg.channels, 0.0f);
            const long long s0 = cIn - Dmax + jlo;
            for (int ch = 0; ch < cfg.channels; ++ch) {
                float* dst = alDec.data() + static_cast<size_t>(ch) * SLd;
                for (int k = 0; k < SLd; ++k) {
                    float a = 0.0f;
                    for (int q = 0; q < Rd; ++q)
                        a += inRing[ch][static_cast<size_t>((s0 + Rd * k + q) % inCap)];
                    dst[k] = 0.25f * a;
                    alCumD[k + 1] += static_cast<double>(dst[k]) * dst[k];
                }
            }
            for (int k = 1; k <= SLd; ++k) alCumD[k] += alCumD[k - 1];
            std::fill(alPD.begin(), alPD.end(), std::complex<float>(0.0f, 0.0f));
            double Et = 0.0;
            for (int ch = 0; ch < cfg.channels; ++ch) {
                const float* src = alDec.data() + static_cast<size_t>(ch) * SLd;
                for (int k = 0; k < Md; ++k) alTD[k] = k < SLd ? src[k] : 0.0f;
                alFftD->forward(alTD, alSD.data());
                for (int k = 0; k < Md; ++k) {
                    const float v = k < Ld ? src[Dd + k] : 0.0f;   // d = 0 segment
                    alTD[k] = v;
                    Et += static_cast<double>(v) * v;
                }
                alFftD->forward(alTD, alTDS.data());
                for (int k = 0; k < Bd; ++k) alPD[k] += std::conj(alTDS[k]) * alSD[k];
            }
            if (Et < 1e-10 * Ld) return 0;
            alFftD->inverse(alPD.data(), alTD);
            alRD.assign(Dd + 1, 0.0);
            for (int d = 0; d <= Dd; ++d) {
                const int l = Dd - d;
                const double Es = alCumD[l + Ld] - alCumD[l];
                alRD[d] = alTD[l] / std::sqrt(Et * Es + 1e-30);
            }
            double bestD = -2.0;
            for (int d = 4; d <= Dd; ++d) bestD = std::max(bestD, alRD[d]);
            int PD = -1;
            for (int d = 4; d < Dd; ++d)
                if (alRD[d] >= alRD[d - 1] && alRD[d] >= alRD[d + 1] &&
                    alRD[d] >= std::max(thr, bestD - 0.02)) { PD = d; break; }
            if (bestD < thr || PD < 0 ||
                (minP > 0 && Rd * PD + Rd < N / minP))
                return 0;
        }
        int M = 32;
        while (M < L + Dmax) M <<= 1;
        const int B2 = M / 2 + 1;
        if (!alFft || alFft->size() != M) {
            if (alT) pffft_aligned_free(alT);
            alFft.reset(new RealFFT(M));
            alT = RealFFT::alloc(M);
            alTS.resize(B2);
            alP.resize(B2);
        }
        alSS.resize(static_cast<size_t>(B2) * cfg.channels);
        const int SL = L + Dmax;
        alCum.assign(SL + 1, 0.0);
        const long long s0 = cIn - Dmax + jlo;
        for (int ch = 0; ch < cfg.channels; ++ch) {
            for (int k = 0; k < M; ++k) {
                const float v = k < SL
                    ? inRing[ch][static_cast<size_t>((s0 + k) % inCap)] : 0.0f;
                alT[k] = v;
                if (k < SL) alCum[k + 1] += static_cast<double>(v) * v;
            }
            alFft->forward(alT, alSS.data() + static_cast<size_t>(ch) * B2);
        }
        for (int k = 1; k <= SL; ++k) alCum[k] += alCum[k - 1];

        // 1) local periodicity of the current segment: autocorrelation over
        //    lags 0..Dmax; the fundamental period P is the first strong
        //    maximum. Only a long period (partials closer than minP bins:
        //    unresolved by the Hann main lobe, or resolved but with a
        //    waveform shape the vocoder would not keep) is handled here;
        //    shorter periods stay on the phase-locked vocoder.
        alignCorr(cIn + jlo, L, Dmax, M, alR);
        double best = -2.0;
        for (int d = 16; d <= Dmax; ++d) best = std::max(best, alR[d]);
        if (best < thr) {
            if (debugOnsets) std::fprintf(stderr, "align c=%lld per=%.4f\n", cIn, best);
            alRun = 0;
            return 0;
        }
        int P = -1;
        for (int d = 16; d < Dmax; ++d)
            if (alR[d] >= alR[d - 1] && alR[d] >= alR[d + 1] &&
                alR[d] >= std::max(thr, best - 0.02)) { P = d; break; }
        // Resolved harmonics (P < N/4) are aligned only when the input is
        // periodic to the strict standard and keeps its own spectrum: there
        // the vocoder already holds every partial's level and the gain is
        // the waveform shape, which a quasi-periodic source (a voice: jitter,
        // vibrato, formant motion) does not keep from vowel to vowel anyway,
        // and which the formant correction re-shapes in every frame -- aligned
        // speech switched paths at every vowel (at -24 st: +18..+24 extra
        // onsets, NISQA noisiness -0.12).
        static const double strictPerShort = 0.995;
        if (P < 0 || (minP > 0 && P < N / minP) ||
            (P < N / 4 && (best < strictPerShort || (formant && pitchSemi != 0.0)))) {
            if (debugOnsets)
                std::fprintf(stderr, "align c=%lld per=%.4f P=%d short\n", cIn, best, P);
            alRun = 0;
            return 0;
        }
        const double Pf = P + parabolic(alR, P);
        // hysteresis: only sustained periodicity switches the path (an
        // isolated aligned frame between phase-vocoder frames is a splice)
        static constexpr int kRun = 3;
        alRun = prevRun + 1;
        if (alRun < kRun) return 0;
        if (!alPrevUsed) {
            // entry from the phase-vocoder path: match the output already
            // synthesised around this frame's output centre, one period wide
            const long long o = synthCount * hs;
            if (o + jlo < 0) return 0;
            alignCorr(o + jlo, L, Dmax, M, alR, /*fromOutput=*/true);
            const int hiE = std::min(Dmax - 1, static_cast<int>(std::ceil(Pf)));
            int bd = 1;
            for (int d = 1; d <= hiE; ++d)
                if (alR[d] > alR[bd]) bd = d;
            if (debugOnsets)
                std::fprintf(stderr, "align c=%lld per=%.4f P=%.2f entry d=%d out=%.4f\n",
                             cIn, best, Pf, bd, alR[bd]);
            static constexpr double entryThr = 0.5;
            // A strictly periodic input enters even when the vocoder output
            // has already lost its waveform (that is exactly the failure
            // being repaired); quasi-periodic mixtures need a coherent join.
            // A blind join is only worth it for a long steady tone: real
            // mixtures reach this periodicity for a few frames at a time
            // (bass under a mix), and a short aligned run is two splices.
            static constexpr double strictPer = 0.995;
            static constexpr int strictRun = 12;
            if (alR[bd] < entryThr && (best < strictPer || alRun < strictRun))
                return 0;
            double tot = -parabolic(alR, bd);
            long long d = bd;
            const double r = std::floor(tot + 0.5);
            tot -= r;
            d -= static_cast<long long>(r);
            if (d < 0) return 0;
            fAligned = true;
            fAlignFrac = tot;
            return d;
        }
        // 2) predicted position: continue the previous frame's waveform by
        //    one synthesis hop, moved back by whole periods (backward only:
        //    no look-ahead beyond the nominal frame)
        const double want = static_cast<double>(alPrevC) + alPrevFrac + hs;
        double dPred = std::fmod(static_cast<double>(cIn) - want, Pf);
        if (dPred < 0.0) dPred += Pf;
        // 3) refine against the natural continuation itself (tracks slow
        //    pitch drift) within a few samples of the prediction
        alignCorr(tc + jlo, L, Dmax, M, alR);
        const int W = 4 + static_cast<int>(0.01 * Pf);
        const int lo = std::max(1, static_cast<int>(std::floor(dPred)) - W);
        const int hi = std::min(Dmax - 1, static_cast<int>(std::ceil(dPred)) + W);
        if (lo > hi) return 0;
        int bd = lo;
        for (int d = lo; d <= hi; ++d)
            if (alR[d] > alR[bd]) bd = d;
        if (alR[bd] < thr) {
            if (debugOnsets)
                std::fprintf(stderr, "align c=%lld per=%.4f P=%d nat=%.4f low\n",
                             cIn, best, P, alR[bd]);
            return 0;
        }
        // centre cIn - bd + frac matches x[tc]; x[tc] stands for content the
        // previous frame placed at alPrevC + alPrevFrac -> carry that over and
        // fold the total into the integer offset (|frac| <= 0.5)
        double tot = -parabolic(alR, bd) + alPrevFrac;
        long long d = bd;
        const double r = std::floor(tot + 0.5);
        tot -= r;
        d -= static_cast<long long>(r);
        if (d < 0) return 0;
        fAligned = true;
        fAlignFrac = tot;
        if (debugOnsets)
            std::fprintf(stderr,
                         "align c=%lld per=%.4f P=%.2f dPred=%.2f d=%lld frac=%.3f nat=%.4f\n",
                         cIn, best, Pf, dPred, d, fAlignFrac, alR[bd]);
        return d;
    }

    void beginFrame(long long cIn) {
        fActive = true;
        fStage = 0;
        alPrevUsed = alignUsed;
        alignUsed = false;
        if (!alPrevUsed) alPrevFrac = 0.0;
        fCIn = cIn - chooseAlign(cIn);
        fDcFitDone = false;
        alPrevC = fCIn;
        alPrevFrac = fAligned ? fAlignFrac : 0.0;
        fDueBase = cIn + N / 2;
        // spread the stages over one nominal frame period: the last stage
        // is due exactly when the next frame's input span completes (one
        // hop of extra output availability latency, see outputLatency())
        fSpread = std::max<long long>(0, llround(hs / alpha));
        fBestE = -1.0;
        fBest = refCh;
        // onset gate of this frame: decided before analysis so the analysis
        // input itself can be masked (see stageAnalyze)
        fCOut = synthCount * hs;
        // reaches past an end no predictor could continue (only possible once
        // finished: the frame's input span is not complete before that)
        fEndTreat = tailUntrusted && endGlide && finished && endTreatMode() &&
                    fCIn + N / 2 > realEnd;
        fGated = buildGate(hs / alpha);
        // per-frame snapshots so all channels see identical parameters
        fDoFormant = formant && pitchSemi != 0.0;
        fP = (fDoFormant || postActive) ? std::pow(2.0, pitchSemi / 12.0) : 1.0;
        // resampled input has detail compressed by p: scale the order
        fEffOrder = static_cast<int>(teOrder * std::max(1.0, 1.0 / fP) + 0.5);
        // Formant-correction strength taper (measured via SQUIM MOS): full
        // correction wins for upshift and mild downshift but over-corrects at
        // deep downshift (-24 st scored best with NO correction). Ramp from
        // full at >= -12 st to zero at -24 st. Re-checked after the sub-F0 /
        // F0-tracker fixes (speech -24 st, NISQA noisiness en/ja): strength
        // 0 -> 2.95/2.75, 0.3 -> 2.13/2.03, 0.6 -> 2.05/1.95 (fail < 2.68):
        // at -24 st the output's upper formant band is made of the input's
        // 6-24 kHz breath noise, which any correction would have to amplify.
        fFmtStrength = 1.0f;
        if (pitchSemi < -12.0)
            fFmtStrength = std::clamp(
                static_cast<float>((pitchSemi + 24.0) / 12.0), 0.0f, 1.0f);
        if (const char* s = std::getenv("PBSHIFT_FMT_STRENGTH"))
            fFmtStrength = static_cast<float>(std::atof(s));
    }

    void runStage() {
        const int C = cfg.channels;
        const int s = fStage++;
        if (s < C) {
            stageAnalyze(s);
        } else if (s < 2 * C) {
            stageFormant(s - C);
        } else if (s == 2 * C) {
            stageDecision();
        } else {
            stageRender(s - 2 * C - 1);
            if (s == 3 * C) {  // frame complete
                ++synthCount;
                inPos += fAdvance;
                fActive = false;
            }
        }
    }

    // Transient detector on consecutive analysis frames of the reference
    // channel: SuperFlux-style max-filtered log flux OR-gated with a
    // broadband rise counter; fills resetMask for rising bins.
    bool detectTransient(const AnalysisFrame& R, long long cIn) {
        const int B = N / 2 + 1;
        std::fill(resetMask.begin(), resetMask.end(), 0);
        if (prevRefMag.empty()) {
            prevRefMag.assign(R.mag.begin(), R.mag.end());
            return false;
        }
        float fluxSum = 0.0f;
        int riseCnt = 0, actCnt = 0;
        float maxMag = 0.0f;
        for (int m = 0; m < B; ++m)
            if (R.mag[m] > maxMag) maxMag = R.mag[m];
        const float floor_ = std::max(1e-7f, 1e-4f * maxMag);
        for (int m = 0; m < B; ++m) {
            // 3-bin max filter on previous frame (vibrato tolerance)
            float pm = prevRefMag[m];
            if (m > 0 && prevRefMag[m - 1] > pm) pm = prevRefMag[m - 1];
            if (m + 1 < B && prevRefMag[m + 1] > pm) pm = prevRefMag[m + 1];
            const float cm = R.mag[m];
            if (cm > floor_ || pm > floor_) {
                ++actCnt;
                const float d = std::log(cm + floor_) - std::log(pm + floor_);
                if (d > 0.0f) fluxSum += d;
                if (cm > 1.41f * pm) {  // +3 dB rise
                    ++riseCnt;
                    resetMask[m] = 1;
                }
            }
        }
        lastFlux = 0.0f;
        lastRiseFrac = 0.0f;
        // mag^2-weighted group delay of the rising bins = where the incoming
        // event sits relative to the window center (+ = still ahead)
        double tw = 0.0, twSum = 0.0;
        for (int m = 0; m < B; ++m) {
            if (!resetMask[m]) continue;
            const double w2 = static_cast<double>(R.mag[m]) * R.mag[m];
            tw += w2 * R.tau[m];
            twSum += w2;
        }
        tauMean = twSum > 0.0 ? static_cast<float>(tw / twSum) : 0.0f;

        const float flux = actCnt ? fluxSum / actCnt : 0.0f;
        const float riseFrac = actCnt ? static_cast<float>(riseCnt) / actCnt : 0.0f;
        const bool refractory =
            (cIn - lastOnsetIn) < 0.05 * cfg.sampleRate;  // 50 ms
        bool onset = !refractory &&
                     ((flux > 0.22f && flux > 1.1f * prevFlux) ||
                      riseFrac > 0.35f) &&
                     riseFrac > 0.12f;
        // A fading uncontinuable edge redistributes energy into the spectral
        // sidelobes of the truncated window. Rising sidelobes are not an
        // attack when the frame as a whole is losing energy. Keep attacks
        // that add energy, and leave all fully supported frames unchanged.
        if (onset && fEndTreat) {
            // On pitch-up the post resampler discards the upper band. A
            // falling discarded partial can still introduce rising audible
            // sidelobes, which require the established transient treatment.
            const int audibleBins = fP > 1.0
                ? std::min(B, static_cast<int>(B / fP) + 1) : B;
            double now = 0.0, before = 0.0, audibleNow = 0.0, audibleBefore = 0.0;
            for (int m = 0; m < B; ++m) {
                const double eNow = static_cast<double>(R.mag[m]) * R.mag[m];
                const double eBefore = static_cast<double>(prevRefMag[m]) * prevRefMag[m];
                now += eNow;
                before += eBefore;
                if (m < audibleBins) {
                    audibleNow += eNow;
                    audibleBefore += eBefore;
                }
            }
            if (now <= before && audibleNow <= audibleBefore) onset = false;
        }
        // An attack ADDS energy; a gliding partial only moves its energy:
        // one flank of the moving main lobe rises while the other falls by as
        // much, so the per-bin rise count fires every few frames on a log
        // sweep (riseFrac 0.35-0.55 at 0.00 dB frame energy change). Such a
        // partial is rendered by the chirp path of Rtpghi::stepLocked, and a
        // hold would freeze the flank it glides into while a fire resets its
        // phase: both break the overlap of consecutive grains (-7 dB on the
        // upper half of the -12 st sweep; under expansion, where the detector
        // fired every 70 ms on a log sweep, holds and pins left dips and
        // re-timed stretches). So when most of the energy that arrived lies
        // inside regions the phase engine classifies as gliding partials
        // (isChirp, or a clean near-DC partial), the frame is not an onset.
        // Other material (speech glides between resolved harmonics, attacks,
        // phoneme changes) keeps its detection.
        if (onset && glideK > 0.0f && !multiMode && chirpOn &&
            (alpha < 1.0 || (glideX && alpha > 1.0))) {
            int nc = pghi->chirpRegions(R, glideMask);
            // a clean near-DC partial (see nearDcModel) glides the same way
            if (dcOn && dcFitFrame()) {
                for (int m = 0; m <= fDcFitM; ++m) glideMask[m] = 1;
                ++nc;
            }
            if (nc > 0) {
                double eu = 0.0, euC = 0.0;
                for (int m = 0; m < B; ++m) {
                    const double d = static_cast<double>(R.mag[m]) * R.mag[m] -
                                     static_cast<double>(prevRefMag[m]) * prevRefMag[m];
                    if (d <= 0.0) continue;
                    eu += d;
                    if (glideMask[m]) euC += d;
                }
                if (euC >= glideK * eu) onset = false;
            }
        }
        lastFlux = flux;
        lastRiseFrac = riseFrac;
        prevFlux = flux;
        prevRefMag.assign(R.mag.begin(), R.mag.end());
        if (!onset && pendCount == 0)
            std::fill(resetMask.begin(), resetMask.end(), 0);
        if (onset && debugOnsets)
            std::fprintf(stderr, "onset@in=%.3fs flux=%.3f rise=%.3f tau=%.0f\n",
                         cIn / (double)cfg.sampleRate, flux, riseFrac, tauMean);
        return onset;
    }
    bool debugOnsets = std::getenv("PBSHIFT_DEBUG_ONSETS") != nullptr;
    float glideK = 0.5f;
    std::vector<uint8_t> glideMask;   // scratch: bins of gliding-partial regions
    std::vector<float> chirpGain;     // scratch: glide-bin magnitude factors
    bool chirpOn = true;
    bool chirpX = true;
    bool glideX = true;
    // ---- partials the reference channel does not share -----------------
    // Every channel is rotated region by region with the reference channel's
    // phase decision (inter-channel phase differences copied verbatim), which
    // keeps a correlated stereo image intact. A partial of another channel
    // that is not the reference's (uncorrelated stereo: a tone on one side,
    // a sweep on the other) then sits inside a reference region built around
    // some other peak, or around the reference's sidelobes, and takes that
    // peak's rotation: its phase advances at the wrong frequency every frame
    // (measured: a -20 dB 330 Hz tone opposite a log sweep lost 3-21 dB,
    // median, with dips to -60 dB). A region of a channel whose peak the
    // reference channel does not carry at all -- at least 30 dB weaker there,
    // i.e. leakage, not the same partial at another gain -- is propagated
    // from that channel's own previous output (its own identity lock), the
    // way a mono input would be. Partials both channels carry keep the shared
    // reference, also when their level differs (panned sources, detuned
    // layers whose beating swings the L/R ratio by +-14 dB: a looser test, or
    // a coherence test, flipped those regions back and forth and the phase
    // jumps read as extra onsets on stereo pads).
    bool chOwnOn = true;
    std::vector<std::unique_ptr<Rtpghi>> chPghi;     // per channel (C > 1 only)
    std::vector<std::vector<float>> chPhase, chGain; // own-region phases / chirp gains
    std::vector<std::vector<uint8_t>> chOwn;         // own-region bins per channel
    std::vector<uint8_t> chOwnAny;
    std::vector<float> chActual;                     // scratch: rendered phases
    std::vector<uint8_t> chPeakOwn;                  // scratch: bins the ref lacks
    bool noTransient = std::getenv("PBSHIFT_NO_TRANSIENT") != nullptr;
    bool noPin = std::getenv("PBSHIFT_NO_PIN") != nullptr;
    bool noTauGate = false;
    bool forceVoice = std::getenv("PBSHIFT_VOICE") != nullptr;
    // Coherence-locked identity kernel (rebuild each peak region's phase from
    // the reassignment group delay tau, scaled by alpha, instead of copying the
    // analysis phase verbatim). Cures the hop-synchronous comb/"fine DelayEcho"
    // on voice. Opt-in; also used as the non-voiced fallback inside Voice mode.
    bool forceCoherent = std::getenv("PBSHIFT_COHERENT") != nullptr;
    // measured on the current metric set: phase jitter hurt attack/LTAS and
    // bought nothing on warble — opt-in only until real Noise Morphing lands
    bool noJitter = std::getenv("PBSHIFT_JITTER") == nullptr;
    bool usePghi = std::getenv("PBSHIFT_PHASE") &&
                   !std::strcmp(std::getenv("PBSHIFT_PHASE"), "pghi");

    // ---- frame stages (formerly one monolithic synthesizeFrame) ------

    // analyze one channel; accumulate reference election (sticky energy
    // argmax); the election closes on the last channel
    void stageAnalyze(int ch) {
        extract(ch, fCIn + fAnaShift, frameBuf.data());
        // A gated frame only deposits the part of its grain on its own side
        // of the attack. Analyse only that part: any phase modification
        // (propagation, region locking) spreads a frame's content over its
        // whole grain, so an attack still present in the analysis input
        // leaks as a smeared pre-echo (frames before the owner) or as a
        // smeared repeat (frames after it) even through the output gate
        // (measured -24 st drums: a ~50 ms pre-attack ramp up to -20 dB).
        // Identity-locked frames only: the gate offsets are in their
        // (verbatim) time coordinates. The coherent kernel of Music/Rhythm
        // rebuilds phase from reassigned group delays, which a truncated
        // analysis window corrupts (measured: more extra onsets there) --
        // except on frames that only precede an attack: those must carry
        // nothing of it, and without the mask their coherent rebuild smeared
        // it up to 20 ms ahead (clicks Rhythm +12 st: -46..-35 dB of
        // pre-echo over digital silence, 0 with the mask).
        const bool cohFrame = multiMode && fOwnTau == kNoOwner;
        if (fGated && inMask && (!cohFrame || (cohPreMask && !fHasPost))) {
            float* b = frameBuf.data();
            for (int i = 0; i < N; ++i) b[i] *= fGate[i];
        }
        fe[ch]->analyze(frameBuf.data(), frames[ch]);
        if (fAligned && fAlignFrac != 0.0) {
            // sub-sample part of the alignment: centre-referenced linear phase
            auto& X = frames[ch].X;
            const double w = 2.0 * M_PI / N * fAlignFrac;
            for (size_t m = 0; m < X.size(); ++m)
                X[m] *= std::polar(1.0f, static_cast<float>(w * m));
        }
        double e = 0.0;
        for (float m : frames[ch].mag) e += static_cast<double>(m) * m;
        // Compare hysteresis-weighted scores, rather than multiplying the
        // running maximum after the comparison. The old ordering only favoured
        // reference channel 0; when another channel was the current reference,
        // an only-slightly louder earlier channel could steal the phase anchor
        // every frame and destabilize the stereo image.
        const double score = (ch == refCh) ? 1.5 * e : e;
        if (score > fBestE) {
            fBestE = score;
            fBest = ch;
        }
        if (ch == cfg.channels - 1) refCh = fBest;
    }

    // ---- formant preservation (M6), one channel per stage ------------
    // The front resampler moved formants by p; restore the original
    // envelope: desired(f) = env_in(f*p).
    void stageFormant(int ch) {
        if (!fDoFormant) return;
        AnalysisFrame& F = frames[ch];
        // voice F0 range 60-1100 Hz, moved by the pre-resampler on pitch-down
        const double pre = std::min(1.0, fP);
        tenv[ch]->setF0Range(60.0 * pre, 1100.0 * pre);
        tenv[ch]->compute(F.mag.data(), envLog, fEffOrder);
        // optional temporal smoothing (off by default, see envLambda)
        if (envLambda > 0.0f) {
            auto& s = envSmooth[ch];
            if (s.size() != envLog.size()) {
                s = envLog;
            } else {
                for (size_t m = 0; m < s.size(); ++m)
                    s[m] = envLambda * s[m] + (1.0f - envLambda) * envLog[m];
                envLog = s;
            }
        }
        formantGains(envLog, fP, fmtGain, fFmtStrength);
        const int B = N / 2 + 1;
        // Below the fundamental there is no envelope to restore: the region
        // holds only the floor (breath, rumble, window leakage). There the
        // correction gain is env(f*p)/env(f) = (fundamental's own envelope) /
        // (floor), i.e. up to the +15 dB cap, and the energy renormalization
        // below lifted it further by the level the correction had removed
        // elsewhere. Sung vowel +12 st: the floor under f0 rose ~14 dB to
        // 8.6 dB under the shifted fundamental, an f0_in "phantom" that made
        // pbACC B-F0 track an octave low in 10% of frames; +5 st: D-HUM;
        // speech +24 st: 0-100 Hz carried most of the output energy. So the
        // sub-F0 bins are left exactly as the plain shift renders them (gain 1,
        // excluded from the renormalization), faded into the corrected gain
        // over 0.6..0.85 f0. f0 comes from the envelope estimator's
        // autocorrelation pitch (robust; the old cepstral pick and the
        // "lowest -40 dB peak" fallback both locked onto the floor or onto
        // 2T0/3T0 in most frames of the sung vowel).
        const double f0Bin = tenv[ch]->lastF0Bin();
        const int f0lo = f0Bin > 1.0
                             ? std::min(B - 1, static_cast<int>(0.85 * f0Bin))
                             : 0;
        // The correction re-shapes the envelope; it must not change the
        // frame's level. The asymmetric clamp (deep cuts, capped boosts)
        // otherwise drains level: measured -10 dB at +12 st speech and -15 dB
        // on a sung vowel at +5 st (pbACC FR-LEVEL). Renormalize per frame to
        // equal energy (L2) -- but never beyond equal spectral L1 norm. sum|X|
        // bounds the worst-case waveform peak of the frame; a correction that
        // flattens a spectrum dominated by a few harmonics raises the crest
        // factor, so an L2-only match pushes peaks past the input's (measured:
        // sung vowel +5 st peak 0.71 -> 1.39, crest 1.7 -> 4, NR-LEVEL fail).
        {
            // Pitch-up: the post resampler (ratio 1/p) keeps only the bins
            // below Nyquist/p; match the level of what is actually heard (the
            // discarded band must not set the gain).
            const int Bk = fP > 1.0 ? std::min(B, static_cast<int>(B / fP) + 1) : B;
            double e0 = 0.0, e1 = 0.0, l0 = 0.0, l1 = 0.0;
            for (int m = f0lo; m < Bk; ++m) {
                const double a = F.mag[m];
                e0 += a * a;
                e1 += a * a * fmtGain[m] * fmtGain[m];
                l0 += a;
                l1 += a * fmtGain[m];
            }
            if (e1 > 1e-30 && l1 > 1e-30) {
                const float k = static_cast<float>(
                    std::min(std::sqrt(e0 / e1), l0 / l1));
                for (int m = f0lo; m < B; ++m) fmtGain[m] *= k;
            }
        }
        if (f0lo > 0) {
            const int f0x = static_cast<int>(0.6 * f0Bin);
            const float lg = std::log(fmtGain[f0lo]);
            for (int m = 0; m < f0lo; ++m) {
                const float t = m <= f0x ? 0.0f
                                         : static_cast<float>(m - f0x) / (f0lo - f0x);
                fmtGain[m] = std::exp(t * lg);
            }
        }
        for (int m = 0; m < B; ++m) F.mag[m] *= fmtGain[m];
    }

    // ---- atomic decision stage ---------------------------------------
    // Transient detect + hold/fire/pin + advance/debt decision + phase
    // engine, in the exact per-frame sequential order the state machines
    // (pendCount/pinRemain/debt/lastOnsetIn/prevFlux, magHist, envSmooth,
    // pghi) were designed around. Decides fAdvance and fCOut.
    void stageDecision() {
        const long long cIn = fCIn;
        // ---- transient handling (Roebel hold -> fire -> pin) ---------
        // hold: rising bins of a coming attack keep previous magnitudes so
        //       pre-fire frames don't pre-echo the attack
        // fire: when the attack reaches the window center, synchronized
        //       phase reset places it once
        // pin:  local ratio 1 across all frames overlapping the attack so
        //       every frame agrees on its output position (no doubling)
        // The formant path skips the spectral hold/fire scheduling (the
        // time-domain onset gate stays active). Its hold freezes the rising
        // bins of every frame centred before the event at their pre-event
        // level; with an 85 ms window those frames still cover 40 ms AFTER
        // the event, so a soft vowel onset came out as a dip: speech ja +5 st
        // --formant 5.765-5.785 s at -33 dB against -16 dB in the input
        // (pbACC FR-DROP), and up to 15 ms late when the per-frame formant
        // correction moved the detection by a frame. Without it the onset
        // follows the input within 1-2 dB; formant-mode pitch cases: FR-QUAL
        // mean 3.23 -> 3.27, extra onsets 243 -> 223.
        const bool rise = (noTransient || fDoFormant) ? false
                                                      : detectTransient(frames[refCh], cIn);
        const float Ce = 0.044f * N;  // Roebel center-of-gravity threshold
        bool fire = false;
        // While a time-domain onset is inside this frame, its owner frame
        // places the attack, so the spectral hold/fire/pin scheduling is not
        // used: pinning changes the frame advance around the attack and
        // breaks the owner prediction of the frames before it. Elsewhere
        // (sweeps, soft onsets) the spectral path stays as it was.
        static constexpr bool legacy = false;
        const bool gating = !legacy && (fGated || fOwnTau != kNoOwner);
        if (gating) {
            pendCount = 0;   // fire stays false: no spectral partial resets
        } else if (pendCount > 0) {
            // already holding: fire when event reaches center or timeout
            if (tauMean <= Ce || pendCount >= 3) {
                fire = true;
                pendCount = 0;
            } else {
                ++pendCount;
            }
        } else if (rise) {
            if (tauMean > Ce && std::abs(debt) < 1.5 * hs) {
                pendCount = 1;  // event still ahead: hold this frame
            } else {
                fire = true;    // event at/past center: place it now
            }
        }
        if (pendCount > 0) {
            // hold rising bins at their pre-event magnitudes (per channel)
            const int B = N / 2 + 1;
            for (int ch = 0; ch < cfg.channels; ++ch) {
                auto& pm = prevMagCh[ch];
                if (pm.size() == static_cast<size_t>(B))
                    for (int m = 0; m < B; ++m)
                        if (resetMask[m]) frames[ch].mag[m] = pm[m];
            }
        }
        const double nominal = hs / alpha;
        double advance = nominal;
        float localAlpha = static_cast<float>(alpha);
        const bool onset = fire;
        if (fire) {
            lastOnsetIn = static_cast<double>(cIn);
            // Ratio pinning is for percussive material: on voice it warps
            // local timing around every consonant (measured +2..4 dB
            // envelope LSD at stretch<1). Pin only when (a) the material
            // context allows it and (b) the event itself is a strong
            // broadband hit (speech consonants measure flux <~2, drum and
            // click hits ~5-7).
            const bool voiceish =
                cfg.mode == Config::Mode::Voice ||
                (cfg.mode == Config::Mode::Auto && formant) || noPin;
            const bool strongHit = lastFlux > 2.5f && lastRiseFrac > 0.5f;
            // NOTE: gating the pin to expansion-only was measured net-negative
            // (clicks 0.5x LTAS 0.97 -> 1.71: the phase-reset-only fallback
            // leaks MORE than the pinned path). The pin helps at every ratio
            // it fires; the residual LTAS cost at large expansion is the
            // inherent price of attack-sharpness preservation.
            // Multi mode never pins: rate pinning replays the attack at ratio 1.0
            // across overlapping long analysis windows, so they disagree on where
            // the transient is -> offset OLA copies = flam/doubling. Sharp single
            // attacks come instead from coherent phase + the short synthesis
            // window + partial suppression (no scheduling trick needed).
            if (!gating && !voiceish && !multiMode && strongHit && std::abs(debt) < 4.0 * hs)
                pinRemain = N / hs;  // this frame + the overlapping ones
        }
        if (pinRemain > 0) {
            advance = hs;
            debt += advance - nominal;
            localAlpha = 1.0f;
            --pinRemain;
        } else if (std::abs(debt) > 1e-6 && pendCount == 0) {
            // repay timing debt over ~100 ms of non-transient audio
            const double framesPerRepay =
                std::max(1.0, 0.1 * cfg.sampleRate / hs);
            double repay = debt / framesPerRepay;
            const double cap = 0.5 * nominal;  // keep ratio swing bounded
            repay = std::clamp(repay, -cap, cap);
            advance = nominal - repay;
            debt -= repay;
        }
        // store post-hold magnitudes for the next frame's hold reference;
        // while holding, the detector's reference must also stay pre-event
        // so the fire frame still sees the full rising set
        for (int ch = 0; ch < cfg.channels; ++ch)
            prevMagCh[ch].assign(frames[ch].mag.begin(), frames[ch].mag.end());
        if (pendCount > 0)
            prevRefMag.assign(frames[refCh].mag.begin(),
                              frames[refCh].mag.end());

        const bool transientActive = onset || pendCount > 0 || pinRemain > 0 ||
                                     (gating && (rise || fOwnTau != kNoOwner));
        // Music mode keeps the full synthesis window on steady tonal frames
        // (best harmonic purity and stereo stability), then switches the whole
        // channel set to the short dual window on detected event frames. Rhythm
        // mode intentionally stays short at all times: it trades a little tonal
        // efficiency for the sharpest possible dense percussion.
        const bool rhythmShort = cfg.mode == Config::Mode::Rhythm;
        fShortWin = multiMode && (rhythmShort || transientActive);
        // Coherent (Multi/Rhythm) phase relocates each region's group delay tau
        // onto the stretched grid: an event sitting tau from the window centre
        // lands alpha*tau from the output frame centre. When alpha*|tau|
        // exceeds the synthesis window it either falls outside the window
        // (harmless) or, past N/2, WRAPS around the circular iFFT and re-appears
        // on the other side as a ghost copy of the click (measured: a replica
        // ~N samples after every click at 4x stretch, -6 dB below the click).
        // Bins whose relocated delay lies outside the window belong to an event
        // that another frame (the one centred on it) renders in place, so they
        // are dropped here.
        // Only short-window frames are gated: there the synthesis window is
        // 4*hs wide, so anything relocated beyond 2*hs is discarded anyway, and
        // only the wrapped copy survives. Full-window (tonal) frames keep their
        // behaviour (gating decaying partials there measured as dropouts).
        // (The owner of an attack takes the analysis phase verbatim, see
        // below: nothing is relocated, so nothing can wrap. Gating it would
        // cut the attack's own bins whenever it sits off-centre.)
        static constexpr bool ownerTauGate = false;
        const bool ownerVerbatim = !ownerTauGate && fOwnTau != kNoOwner && std::abs(fOwnTau) < N / 4;
        if (fShortWin && !noTauGate && localAlpha > 1.0f && !ownerVerbatim) {
            const float T = static_cast<float>(2 * hs);
            const float lim = T / localAlpha;
            for (int ch = 0; ch < cfg.channels; ++ch) {
                AnalysisFrame& F = frames[ch];
                for (size_t m = 0; m < F.mag.size(); ++m)
                    if (std::fabs(F.tau[m]) > lim) F.mag[m] = 0.0f;
            }
        }
        if(autoVoiceInitialized && autoVoiceWanted()) {
            const bool protectedAttack=fOwnTau!=kNoOwner && !speechLatched;
            if(protectedAttack) {
                if(autoVoiceSupportPending)autoVoiceSupportChanges.push_back({std::max(0LL,llround((fCOut+fOwnTau-hs)*postRatio())),false});
                autoVoiceSupportPending=false;autoVoiceSupportObserved=false;autoVoiceSupportRun=0;
            }
            double monoEnergy=0,referenceEnergy=0;
            for(size_t m=0;m<autoVoiceMagnitude.size();++m) {
                std::complex<float> mono(0,0);
                for(int ch=0;ch<cfg.channels;++ch)mono+=frames[ch].X[m]/float(cfg.channels);
                autoVoiceMagnitude[m]=std::abs(mono);monoEnergy+=std::norm(mono);
                referenceEnergy+=std::norm(frames[refCh].X[m]);
            }
            const bool cancelled=cfg.channels>1 && monoEnergy<.25*referenceEnergy;
            std::array<double,8> bands{};double bandSum=0,peakEnergy=0;size_t strongest=0;
            for(size_t m=0;m<autoVoiceMagnitude.size();++m) {
                const double energy=double(autoVoiceMagnitude[m])*autoVoiceMagnitude[m];
                if(energy>peakEnergy){peakEnergy=energy;strongest=m;}
                for(int b=0;b<8;++b)bands[b]+=energy*autoVoiceBands[m][b];
            }
            double concentration=0;
            for(size_t m=strongest>3?strongest-3:0;m<std::min(autoVoiceMagnitude.size(),strongest+4);++m)
                concentration+=double(autoVoiceMagnitude[m])*autoVoiceMagnitude[m];
            const bool sparse=concentration>.9*monoEnergy;
            for(double value:bands)bandSum+=value;
            autoVoicePitchProbe->setF0Range(30,std::min(4000,cfg.sampleRate/8));
            const double fundamental=autoVoicePitchProbe->fundamentalBin(autoVoiceMagnitude.data())*cfg.sampleRate/N;
            const long long source=fDueBase-N/2,lag=cfg.sampleRate/10;
            bool prosodic=false;const bool reliable=fundamental>0 && autoVoicePitchProbe->lastPeriodicity()>=.8;
            if(bandSum>1e-10) {
                for(double &value:bands)value=std::sqrt(value/bandSum);
                while(autoVoiceFeatures.size()>1 && autoVoiceFeatures[1].source<=source-lag)autoVoiceFeatures.pop_front();
                if(!autoVoiceFeatures.empty() && autoVoiceFeatures.front().source<=source-lag) {
                    double difference=0;
                    for(int b=0;b<8;++b){const double d=bands[b]-autoVoiceFeatures.front().bands[b];difference+=d*d;}
                    if(difference>.15*.15)autoVoiceModulationUntil=source+cfg.sampleRate/2;
                }
                int count=0;double sum=0,squared=0;long long first=source,last=source;
                for(const auto &feature:autoVoiceFeatures)if(feature.source>=source-lag && feature.fundamental>=80 && feature.fundamental<=500){const double value=std::log(feature.fundamental);sum+=value;squared+=value*value;++count;first=std::min(first,feature.source);last=std::max(last,feature.source);}
                if(reliable && fundamental>=80 && fundamental<=500){const double value=std::log(fundamental);sum+=value;squared+=value*value;++count;last=source;}
                if(count>=3 && last-first>=.04*cfg.sampleRate)prosodic=squared/count-std::pow(sum/count,2)>=.025*.025;
                autoVoiceFeatures.push_back({source,bands,reliable?fundamental:0});
            }
            bool resolved=false;
            if(reliable && fundamental>=80 && fundamental<=500) {
                for(int i=0;i<N;++i){const long long position=source+i-N/2;double mono=0;if(position>=0 && position<totalIn)for(int ch=0;ch<cfg.channels;++ch)mono+=inRing[ch][size_t(position%inCap)]/cfg.channels;autoVoiceResolutionTime[i]=float(mono)*autoVoiceResolutionWindow[i];}
                autoVoiceResolutionFFT->forward(autoVoiceResolutionTime.data(),autoVoiceResolutionSpectrum.data());
                for(size_t i=0;i<autoVoiceResolutionMagnitude.size();++i)autoVoiceResolutionMagnitude[i]=std::abs(autoVoiceResolutionSpectrum[i]);
                std::array<int,32> peaks{};int peakCount=0;
                float maximum=0;for(float value:autoVoiceResolutionMagnitude)maximum=std::max(maximum,value);
                for(int harmonic=1;harmonic<=std::min(32,int(3000/fundamental));++harmonic){const int low=std::max(1,int((harmonic-.18)*fundamental*N/cfg.sampleRate));const int high=std::min(N/2-1,int(std::ceil((harmonic+.18)*fundamental*N/cfg.sampleRate)));if(high<=low)continue;int peak=low;for(int m=low+1;m<=high;++m)if(autoVoiceResolutionMagnitude[m]>autoVoiceResolutionMagnitude[peak])peak=m;peaks[peakCount++]=peak;}
                int observed=0,separated=0;
                for(int k=1;k<peakCount;++k){const int a=peaks[k-1],b=peaks[k];const double weaker=std::min(autoVoiceResolutionMagnitude[a],autoVoiceResolutionMagnitude[b]);if(weaker<maximum*.005623)continue;const double middle=.5*(a+b),radius=std::max(1.0,.15*(b-a));const int low=std::max(a+1,int(std::ceil(middle-radius))),high=std::min(b-1,int(std::floor(middle+radius)));if(high<low)continue;double energy=0;for(int m=low;m<=high;++m)energy+=double(autoVoiceResolutionMagnitude[m])*autoVoiceResolutionMagnitude[m];++observed;if(std::sqrt(energy/(high-low+1))<.177828*weaker)++separated;}
                resolved=observed>=3 && separated==observed;
            }
            const bool modulated=source<=autoVoiceModulationUntil;
            const double elapsed=speechEvidenceSource<0?0:std::clamp(double(source-speechEvidenceSource)/cfg.sampleRate,0.0,.05);
            speechEvidenceSource=source;const double retain=std::exp(-elapsed/.3);
            speechEvidence=retain*speechEvidence+(1-retain)*(prosodic && modulated && resolved?1.0:0.0);
            // Retain speech identity across phonemes; quiet resets the segment.
            if(speechEvidence>=.2)speechLatched=true;
            if(std::abs(autoVoiceInitialPitch)<1e-12){speechPeakEnergy=std::max(speechPeakEnergy,monoEnergy);speechQuietSeconds=monoEnergy<speechPeakEnergy*1e-6?speechQuietSeconds+elapsed:0;if(speechQuietSeconds>=.2){speechLatched=false;speechEvidence=0;}}else{speechPeakEnergy=std::max(speechPeakEnergy,monoEnergy);speechQuietSeconds=monoEnergy<speechPeakEnergy*0.0001?speechQuietSeconds+elapsed:0;if(speechQuietSeconds>=.2){speechLatched=false;speechEvidence=0;}}


            if(!speechLatched || !modulated || sparse || cancelled || (fundamental>0 && autoVoicePitchProbe->lastPeriodicity()>=.8)) {
                const bool supported=speechLatched && !sparse && !cancelled && fundamental>=80 && fundamental<=500;
                if(supported==autoVoiceSupportObserved)++autoVoiceSupportRun;
                else {autoVoiceSupportObserved=supported;autoVoiceSupportRun=1;}
                if(!protectedAttack && (fCOut==0 || autoVoiceSupportRun>=3) && supported!=autoVoiceSupportPending) {
                    autoVoiceSupportChanges.push_back({llround(fCOut*postRatio()),supported,supported?cfg.sampleRate/(fundamental*std::pow(2.0,pitchSemi/12.0)):0});autoVoiceSupportPending=supported;
                }
            }
        }
        pushMagHistory(frames[refCh]);
        if (!noJitter && !transientActive)
            computeJitter(frames[refCh], alpha);
        else
            std::fill(jitterAmt.begin(), jitterAmt.end(), 0.0f);
        // Voice mode: shape-invariant harmonic locking is beneficial only in
        // its measured range. Extending it to 2x time stretch produced a strong
        // analysis-hop comb on real speech, so outside [0.9, 1.6] Voice stays on
        // the coherence-locked kernel instead of silently falling all the way
        // back to plain identity locking. Pitch-shift cases (alphaUser == 1)
        // retain the SHIP-style low-harmonic phase relationship.
        const bool voiceRequested =
            cfg.mode == Config::Mode::Voice || forceVoice;
        const bool voiceShip = voiceRequested && !onset &&
                               alphaUser >= 0.9 && alphaUser <= 1.6;
        double f0Bin = -1.0;
        if (voiceShip) f0Bin = estimateF0Bin(frames[refCh]);
        // The owner of an attack takes the analysis phase verbatim in every
        // region: region-wise phase rotation would shift each partial's share
        // of the impulse differently and smear it across the grain.
        static const bool ownerReset = true;
        std::fill(chOwnAny.begin(), chOwnAny.end(), 0);
        if (ownerReset && fOwnTau != kNoOwner && std::abs(fOwnTau) < N / 4) {
            pghi->stepLocked(frames[refCh], hs, allReset.data(), synthPhase);
        } else if (fAligned && !transientActive && !fGated) {
            // waveform-aligned frame: verbatim analysis phase (see chooseAlign)
            pghi->stepLocked(frames[refCh], hs, allReset.data(), synthPhase);
            alignUsed = true;
        } else if (fEndTreat) {
            // past an end no predictor could continue: one rigid rotation
            // per channel, from that channel's own phase state (see
            // endTreatMode)
            pghi->stepRigid(frames[refCh], hs, synthPhase);
            for (int ch = 0; ch < static_cast<int>(chPghi.size()); ++ch) {
                if (ch == refCh) continue;
                chPghi[ch]->stepRigid(frames[ch], hs, chPhase[ch]);
                std::fill(chOwn[ch].begin(), chOwn[ch].end(), 1);
                chOwnAny[ch] = 1;
            }
        } else if (voiceShip && f0Bin > 0.0) {
            pghi->stepVoice(frames[refCh], hs, f0Bin, synthPhase);
        } else if (usePghi) {
            pghi->step(frames[refCh], hs, localAlpha,
                       static_cast<uint64_t>(synthCount),
                       onset ? resetMask.data() : nullptr, synthPhase);
        } else if (forceCoherent || voiceRequested || multiMode) {
            // coherence-locked identity kernel: explicit opt-in (PBSHIFT_COHERENT
            // on any mode), Multi mode, or the non-voiced fallback inside Voice.
            // Multi/coherent tonal frames stay comb-free (stepLockedCoherent);
            // transient frames take verbatim stepLocked (below) + short window.
            // Transient frames take the verbatim identity lock instead: the
            // coherent kernel rebuilds phase from the reassignment group delay
            // tau, which is ill-defined on broadband non-tonal hits (snare) and
            // smears the attack/decay into an audible "delay"; stepLocked keeps
            // the real per-bin phase -> sharp percussive transients.
            if (transientActive && !multiMode) {
                // forceCoherent / Voice fallback: verbatim sharp attack.
                pghi->stepLocked(frames[refCh], hs,
                                 onset ? resetMask.data() : nullptr, synthPhase,
                                 noJitter ? nullptr : jitterAmt.data(),
                                 static_cast<uint64_t>(synthCount) | (1ull << 40));
            } else {
                // Multi mode: spectrum-wide coherent phase on EVERY frame (tonal
                // AND transient) -- relocates each region's group delay onto the
                // stretched grid so overlapping grains reinforce into one impulse
                // (no per-region verbatim comb = no snare chorus; identity phase
                // locking, Laroche & Dolson 1999).
                // Multi mode passes NO resetMask: a synchronized onset reset makes
                // the onset frame verbatim while its overlapping neighbours are
                // not -> the same attack sums at mismatched sub-sample phase =
                // doubling. Unbroken time-propagated phase continuity instead makes
                // all overlapping grains place the peak at one instant = 1 impulse.
                const uint8_t* rm =
                    (onset && !multiMode) ? resetMask.data() : nullptr;
                // Region rebuild uses the reassignment group delay (tau); energy-
                // rising partial suppression collapses a broadband hit onto few
                // coherent phase centres = one sharp attack.
                pghi->stepLockedCoherent(frames[refCh], hs, localAlpha, rm,
                                         synthPhase,
                                         /*rawPhase=*/false,
                                         /*suppress=*/multiMode);
            }
        } else {
            // gliding partials: alpha-scaled group delay, under compression
            // and (fast glides, with per-bin gains) under expansion (see
            // Rtpghi::stepLocked / isChirp / chirpExpand)
            const float chirpA =
                (chirpOn && (localAlpha < 1.0f || (chirpX && localAlpha > 1.0f)))
                    ? localAlpha : 0.0f;
            pghi->stepLocked(frames[refCh], hs,
                             onset ? resetMask.data() : nullptr, synthPhase,
                             noJitter ? nullptr : jitterAmt.data(),
                             static_cast<uint64_t>(synthCount) | (1ull << 40),
                             chirpA, chirpA > 0.0f ? chirpGain.data() : nullptr);
            const bool refChirp = chirpA > 0.0f && pghi->chirpFound();
            const int B = N / 2 + 1;
            // partials of another channel that the reference does not carry
            // take that channel's own identity lock (see chOwnOn)
            if (chOwnOn && !chPghi.empty()) {
                static const float kOwn = std::pow(10.0f, -30.0f / 20.0f);
                const AnalysisFrame& R = frames[refCh];
                for (int ch = 0; ch < cfg.channels; ++ch) {
                    if (ch == refCh) continue;
                    const AnalysisFrame& F = frames[ch];
                    // audible partials only (within 40 dB of the channel's
                    // strongest bin): the sidelobe floor of two channels
                    // differs at random, and switching its regions to their
                    // own phase changed nothing but low-level detail
                    float fMax = 0.0f;
                    for (int m = 0; m < B; ++m) fMax = std::max(fMax, F.mag[m]);
                    const float fFloor = 0.01f * fMax;
                    for (int m = 0; m < B; ++m)
                        chPeakOwn[m] = F.mag[m] >= fFloor && R.mag[m] < kOwn * F.mag[m];
                    chOwnAny[ch] = chPghi[ch]->ownRegions(
                                       F, chPeakOwn.data(), chOwn[ch]) > 0;
                    if (!chOwnAny[ch]) continue;
                    chPghi[ch]->stepLocked(frames[ch], hs,
                                           onset ? resetMask.data() : nullptr,
                                           chPhase[ch],
                                           noJitter ? nullptr : jitterAmt.data(),
                                           static_cast<uint64_t>(synthCount) | (1ull << 40),
                                           chirpA,
                                           chirpA > 0.0f ? chGain[ch].data() : nullptr,
                                           chPeakOwn.data());
                }
            }
            for (int ch = 0; ch < cfg.channels; ++ch) {
                const bool own = ch != refCh && !chOwnAny.empty() && chOwnAny[ch];
                const bool ownChirp = own && chirpA > 0.0f && chPghi[ch]->chirpFound();
                if (!refChirp && !ownChirp) continue;
                for (int m = 0; m < B; ++m) {
                    if (own && chOwn[ch][m]) {
                        if (ownChirp) frames[ch].mag[m] *= chGain[ch][m];
                    } else if (refChirp) {
                        frames[ch].mag[m] *= chirpGain[m];
                    }
                }
            }
            dcEligible = !transientActive && !fGated && fOwnTau == kNoOwner &&
                         !fDoFormant && localAlpha < 1.0f;
        }
        // near-DC partial model (identity-locked frames only, see nearDcModel)
        fDcM = -1;
        if (dcOn && dcEligible) nearDcModel();
        else dcPrevValid = false;
        dcEligible = false;
        fAdvance = advance;
        if (fGated || fOwnTau != kNoOwner) {
            // an owner whose attack lies outside the short window keeps the
            // full window, or the attack it owns would never be deposited
            const bool shortW = fShortWin &&
                !(fOwnTau != kNoOwner && std::abs(fOwnTau) > 3 * hs / 2);
            if (fShortWin && !shortW && !fGated) {
                fShortWin = false;
            } else if (fGated) {
                const float* sw = shortW ? shortSyn.data() : win.w.data();
                const float* sp = shortW ? shortProd.data() : wprodFull.data();
                for (int i = 0; i < N; ++i) {
                    gateSyn[i] = sw[i] * fGate[i];
                    gateProd[i] = sp[i] * fGate[i];
                }
            }
        }
    }

    // ---- near-DC partial model ----------------------------------------
    // A partial within ~3 bins of DC overlaps its own negative-frequency
    // image inside the Hann main lobe (after a -24 st pre-resample the
    // whole 20-47 Hz octave of the input lands there). The region rotation
    // of the identity lock turns that image the wrong way round, and when
    // the low maximum sits on the DC bin itself (no peak at m >= 1) the
    // region even takes the rotation of the next peak up: the partial
    // cancels in the overlap-add and leaks into 15-40 Hz (measured on the
    // -24 st log sweep: 1-15 Hz band 18-27 dB low over the first 0.3 s,
    // output level 6-15 dB low there). Such a low region is instead fitted
    // as one real sinusoid,
    //     X(m) = a W(m - f) + conj(a) W(m + f),  W = centred Hann kernel,
    // (frequency f by a coarse-to-fine scan, complex amplitude a by least
    // squares, shared frequency and per-channel amplitude) and, when every
    // channel fits it closely, re-synthesised from the same model with the
    // partial's time-propagated phase: each image turns its own way
    // (sinusoidal-model resynthesis, McAulay & Quatieri 1986, applied only
    // to the band the frame cannot resolve). Compression only (the all-pre
    // pitch-down is what pushes whole octaves there). Frames where the fit
    // fails keep the identity lock; the model state is a pure function of
    // the frames, so output stays chunk-independent.
    bool dcOn = true;
    static constexpr int kDcBins = 10;
    bool dcEligible = false;     // this frame took the plain identity lock
    bool dcPrevValid = false;    // previous frame rendered its low band by the model
    double dcPsi = 0.0;          // that frame's partial phase (positive image)
    double dcF0 = 0.0;           // that frame's partial frequency (bins)
    int fDcM = -1;               // this frame: bins 0..fDcM come from the model
    std::vector<std::complex<float>> dcY;   // [ch * kDcBins + m]
    std::vector<std::complex<float>> dcPrevY;   // last rendered low band (ref channel)
    bool dcPrevYValid = false;

    // centre-referenced periodic-Hann kernel at an offset of g bins
    std::complex<double> hannKernel(double g) const {
        auto S = [&](double x) -> std::complex<double> {
            const double sd = std::sin(M_PI * x / N);
            if (std::abs(sd) < 1e-12) return {static_cast<double>(N), 0.0};
            return std::polar(std::sin(M_PI * x) / sd, M_PI * x / N);
        };
        return 0.5 * S(g) + 0.25 * S(g - 1.0) + 0.25 * S(g + 1.0);
    }

    // least-squares a for X(m) = a u(m) + conj(a) v(m), m = 0..M, with
    // u(m) = W(m - f), v(m) = W(m + f); returns the residual energy
    double dcFit(const std::complex<float>* X, int M, double f,
                 std::complex<double>& a, std::complex<double>* u,
                 std::complex<double>* v, double* kappa = nullptr) const {
        const std::complex<double> I(0.0, 1.0);
        double sss = 0.0, stt = 0.0, sst = 0.0, sx = 0.0, tx = 0.0;
        for (int m = 0; m <= M; ++m) {
            u[m] = hannKernel(m - f);
            v[m] = hannKernel(m + f);
            const std::complex<double> s = u[m] + v[m], t = I * (u[m] - v[m]);
            const std::complex<double> x(X[m].real(), X[m].imag());
            sss += std::norm(s);
            stt += std::norm(t);
            sst += (std::conj(s) * t).real();
            sx += (std::conj(s) * x).real();
            tx += (std::conj(t) * x).real();
        }
        const double det = sss * stt - sst * sst;
        if (kappa) {
            // conditioning of the 2x2 normal matrix (eigenvalue ratio)
            const double tr = sss + stt;
            const double disc = std::sqrt(std::max(0.0, 0.25 * tr * tr - det));
            const double lmax = 0.5 * tr + disc, lmin = 0.5 * tr - disc;
            *kappa = lmax > 0.0 ? std::max(0.0, lmin) / lmax : 0.0;
        }
        if (!(std::abs(det) > 1e-30 * (sss * stt + 1e-300))) {
            a = 0.0;
            return 1e300;
        }
        a = {(sx * stt - tx * sst) / det, (tx * sss - sx * sst) / det};
        double res = 0.0;
        for (int m = 0; m <= M; ++m) {
            const std::complex<double> x(X[m].real(), X[m].imag());
            res += std::norm(x - (a * u[m] + std::conj(a) * v[m]));
        }
        return res;
    }

    // Fit of the current frame's low band (cached per frame: the transient
    // gate and the renderer both ask). Valid when the band holds one real
    // sinusoid, the same frequency in every channel, closely enough that its
    // re-synthesis at any phase is reliable (see the residual test below).
    bool fDcFitDone = false, fDcFitValid = false;
    int fDcFitM = -1;
    double fDcFitF = 0.0;
    bool dcFitFrame() {
        if (fDcFitDone) return fDcFitValid;
        fDcFitDone = true;
        fDcFitValid = false;
        const int B = N / 2 + 1;
        const AnalysisFrame& R = frames[refCh];
        float maxMag = 0.0f;
        for (int m = 0; m < B; ++m) maxMag = std::max(maxMag, R.mag[m]);
        int p0 = 0;
        for (int m = 1; m <= 5 && m < B; ++m)
            if (R.mag[m] > R.mag[p0]) p0 = m;
        int M = p0;
        while (M + 1 < std::min(B, kDcBins) && R.mag[M + 1] < R.mag[M]) ++M;
        // a dominant (>= -20 dB re the frame peak) low maximum within 4 bins
        // of DC, with at least 3 bins down to its upper valley
        if (p0 > 4 || R.mag[p0] < 0.1f * maxMag || M < 2) return false;
        double E = 0.0;
        for (int m = 0; m <= M; ++m) E += std::norm(std::complex<double>(R.X[m]));
        std::complex<double> u[kDcBins], v[kDcBins], a;
        double bestF = -1.0, bestR = 1e300;
        auto scan = [&](double lo, double hi, double step) {
            for (int i = 0;; ++i) {
                const double f = lo + i * step;
                if (f > hi + 1e-9) break;
                const double r = dcFit(R.X.data(), M, f, a, u, v);
                if (r < bestR) { bestR = r; bestF = f; }
            }
        };
        // f >= 0.25 bin: below that the window holds under a quarter period,
        // the sine part of the partial is unobservable (u - v -> 0, so its
        // least-squares amplitude is unbounded) and "rotating" it is
        // meaningless -- a DC offset or a slow drift, not a partial.
        scan(0.25, p0 + 1.5, 0.05);
        if (bestF < 0.0) return false;
        scan(std::max(0.25, bestF - 0.05), bestF + 0.05, 0.005);
        // The residual must be small against the weakest observable
        // direction of the model: near DC the sine part of the partial
        // projects onto the window with only kappa (~ f^2) of the cosine
        // part's energy, so a residual that is small overall can still hide
        // an amplitude error that a phase rotation would expose. Requiring
        // res <= 2% * kappa * E keeps the re-synthesised amplitude within
        // ~15% in every direction (a slow drift or DC offset fails here).
        double kap = 0.0;
        dcFit(R.X.data(), M, bestF, a, u, v, &kap);
        kap = std::min(0.5, kap);
        if (bestR > 0.02 * kap * E) return false;
        // every channel must follow the same one-partial model
        for (int ch = 0; ch < cfg.channels; ++ch) {
            if (ch == refCh) continue;
            const AnalysisFrame& F = frames[ch];
            double Ec = 0.0;
            for (int m = 0; m <= M; ++m) Ec += std::norm(std::complex<double>(F.X[m]));
            const double rc = dcFit(F.X.data(), M, bestF, a, u, v);
            if (Ec > 1e-6 * E && rc > 0.02 * kap * Ec) return false;
        }
        fDcFitValid = true;
        fDcFitM = M;
        fDcFitF = bestF;
        return true;
    }

    void nearDcModel() {
        if (!dcFitFrame()) { dcPrevValid = false; return; }
        const int M = fDcFitM;
        const double bestF = fDcFitF;
        const AnalysisFrame& R = frames[refCh];
        const int C = cfg.channels;
        std::complex<double> u[kDcBins], v[kDcBins];
        std::complex<double> aRef;
        dcFit(R.X.data(), M, bestF, aRef, u, v);
        // partial phase: time-propagated (trapezoidal) from the previous
        // frame's model. On entry, from the partial as the previous frame
        // actually rendered it (its low band fitted with the same model), so
        // the join is continuous; if that band was not one clean partial,
        // continue the rotation the identity lock gave the partial's bin.
        double psi = 0.0;
        if (dcPrevValid) {
            psi = dcPsi + hs * M_PI * (dcF0 + bestF) / N;
        } else {
            const int pm = std::clamp(static_cast<int>(std::lround(bestF)), 1, M);
            psi = std::arg(aRef) + (synthPhase[pm] - std::arg(R.X[pm]));
            if (synthCount * hs < N) {
                // stream start (the partial has been there from the first
                // frame, whose low band the head padding kept from fitting):
                // the phase the vocoder's own convention gives it -- frame 0
                // verbatim, then propagated over k synthesis hops -- i.e. this
                // frame's analysis phase taken back over its input position
                // and forward over its output position.
                psi = std::arg(aRef) - (2.0 * M_PI * bestF / N) *
                                           static_cast<double>(fCIn - fCOut);
            } else if (dcPrevYValid) {
                std::complex<double> up[kDcBins], vp[kDcBins], ap;
                double Ep = 0.0;
                for (int m = 0; m <= M; ++m) Ep += std::norm(std::complex<double>(dcPrevY[m]));
                double bf = bestF, br = 1e300;
                for (int i = -20; i <= 20; ++i) {
                    const double f = bestF + 0.005 * i;
                    if (f < 0.25) continue;
                    const double r = dcFit(dcPrevY.data(), M, f, ap, up, vp);
                    if (r < br) { br = r; bf = f; }
                }
                double kp = 0.0;
                dcFit(dcPrevY.data(), M, bf, ap, up, vp, &kp);
                if (Ep > 0.0 && br <= 0.02 * std::min(0.5, kp) * Ep)
                    psi = std::arg(ap) + hs * M_PI * (bf + bestF) / N;
            }
        }
        psi -= 2.0 * M_PI * std::floor(psi / (2.0 * M_PI) + 0.5);
        for (int ch = 0; ch < C; ++ch) {
            std::complex<double> ac = aRef;
            if (ch != refCh) dcFit(frames[ch].X.data(), M, bestF, ac, u, v);
            else dcFit(R.X.data(), M, bestF, ac, u, v);
            const std::complex<double> as =
                std::polar(std::abs(ac), psi + std::arg(ac) - std::arg(aRef));
            for (int m = 0; m <= M; ++m)
                dcY[static_cast<size_t>(ch * kDcBins + m)] =
                    std::complex<float>(as * u[m] + std::conj(as) * v[m]);
        }
        for (int m = 0; m <= M; ++m) {
            const float ph = std::arg(dcY[static_cast<size_t>(refCh * kDcBins + m)]);
            synthPhase[m] = ph;
            pghi->setPhase(m, ph);
        }
        fDcM = M;
        dcPrevValid = true;
        dcPsi = psi;
        dcF0 = bestF;
    }

    // render one channel: spectrum assembly + iFFT + WOLA accumulate
    void stageRender(int ch) {
        const int B = N / 2 + 1;
        const AnalysisFrame& R = frames[refCh];
        const AnalysisFrame& F = frames[ch];
        if (ch == refCh) {
            for (int m = 0; m < B; ++m)
                Y[m] = std::polar(F.mag[m], synthPhase[m]);
        } else {
            // copy input inter-channel phase delta verbatim
            const bool own = !chOwnAny.empty() && chOwnAny[ch];
            for (int m = 0; m < B; ++m) {
                if (own && chOwn[ch][m]) {
                    // a partial of this channel alone: its own identity lock
                    Y[m] = std::polar(F.mag[m], chPhase[ch][m]);
                    continue;
                }
                std::complex<float> rel = F.X[m] * std::conj(R.X[m]);
                const float rm = std::abs(rel);
                if (rm > 1e-20f)
                    rel /= rm;
                else
                    rel = {1.0f, 0.0f};
                Y[m] = F.mag[m] * std::polar(1.0f, synthPhase[m]) * rel;
            }
        }
        // near-DC partial rendered by its sinusoid model (see nearDcModel)
        for (int m = 0; m <= fDcM; ++m)
            Y[m] = dcY[static_cast<size_t>(ch * kDcBins + m)];
        // real-FFT edge bins must stay real
        Y[0] = {std::abs(Y[0]) * (std::cos(std::arg(Y[0])) < 0 ? -1.0f : 1.0f), 0.0f};
        Y[B - 1] = {std::abs(Y[B - 1]) * (std::cos(std::arg(Y[B - 1])) < 0 ? -1.0f : 1.0f), 0.0f};
        if (!chPghi.empty()) {
            // this channel's own phase state continues from what it rendered
            // (phases taken lazily at the bins a later step reads; the
            // fallback covers bins rendered at exactly zero)
            const bool own = ch != refCh && !chOwnAny.empty() && chOwnAny[ch];
            if (own) {
                for (int m = 0; m < B; ++m)
                    chActual[m] = chOwn[ch][m] ? chPhase[ch][m] : synthPhase[m];
                chPghi[ch]->commit(F, Y, chActual);
            } else {
                chPghi[ch]->commit(F, Y, synthPhase);
            }
        }
        if (ch == refCh) {
            // the rendered low band, for the near-DC model's next entry
            for (int m = 0; m < kDcBins && m < B; ++m) dcPrevY[m] = Y[m];
            dcPrevYValid = !fGated && !fShortWin;
        }
        if (fGated)
            wola[ch]->addFrame(Y.data(), fCOut, gateSyn.data(), gateProd.data());
        else if (fShortWin)
            wola[ch]->addFrame(Y.data(), fCOut, shortSyn.data(), shortProd.data());
        else
            wola[ch]->addFrame(Y.data(), fCOut);
    }

    long long finalized() const {
        // a sample is final when no future frame can touch it
        const long long nextCenter = synthCount * hs;
        return std::max(0LL, nextCenter - N / 2);
    }
};

Stretcher::Stretcher() : impl_(new Impl) {}
Stretcher::~Stretcher() = default;

void Stretcher::configure(const Config& cfg) { impl_->configure(cfg); }
void Stretcher::reset() { impl_->resetState(); }

void Stretcher::setTimeStretch(double ratio) {
    const double value = std::clamp(ratio, 0.05, 20.0);
    if (value != impl_->alphaUser) impl_->fallbackNearToStreaming();
    impl_->alphaUser = value;
    impl_->scheduleAutoVoiceMix();
    impl_->updateAlpha();
}
void Stretcher::setPitchSemitones(double s) {
    const double value = std::clamp(s, -24.0, 24.0);
    if (value == impl_->pitchSemi) return;
    impl_->fallbackNearToStreaming();
    impl_->pitchSemi = value;
    impl_->scheduleAutoVoiceMix();
    impl_->updateAlpha();
    // The stretch change above takes effect on the input side; queue the
    // matching output-side ratio so both land on the same audio.
    impl_->schedulePostChange();
}
void Stretcher::setFormantPreserve(bool e) {
    if (e != impl_->formant) impl_->fallbackNearToStreaming();
    impl_->formant = e;
    impl_->scheduleAutoVoiceMix();
}

void Stretcher::feed(const float* const* in, int frames) {
    impl_->feedUser(in, frames);
}

void Stretcher::finish() { impl_->finishUser(); }

int Stretcher::available() const {
    return const_cast<Impl*>(impl_.get())->availableUser();
}

int Stretcher::read(float* const* out, int frames) {
    return impl_->readUser(out, frames);
}

int Stretcher::inputLatency() const {
    if (impl_->vsActive()) return impl_->voice.inputLatency();
    return impl_->N / 2;
}
// + one extra hop: the staged frame pipeline finishes a frame up to one
// nominal frame period after its input span completes (output content is
// unchanged; only availability shifts by at most hs output samples).
int Stretcher::outputLatency() const {
    if (impl_->vsActive()) return impl_->voice.outputLatency();
    const bool attackPinning=impl_->cfg.mode==Config::Mode::Auto && !impl_->formant;
    return impl_->N / 2 + (attackPinning?2*impl_->hs:impl_->hs+128);
}

}  // namespace pbshift
