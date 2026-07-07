// wav_io.h - minimal single-header WAV reader/writer
//
// Read : 16/24/32-bit PCM and 32-bit IEEE float, any channel count, any rate.
//        (WAVE_FORMAT_EXTENSIBLE is unwrapped to its sub-format.)
// Write: 32-bit IEEE float.
//
// Samples are stored deinterleaved as float in [-1, 1], one vector per channel.
// No external dependencies. C++17.
//
// namespace pbwav

#ifndef PBWAV_WAV_IO_H
#define PBWAV_WAV_IO_H

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace pbwav {

struct AudioFile {
	int sampleRate = 0;
	int channels = 0;
	// samples[channel][frame], float in [-1, 1]
	std::vector<std::vector<float>> samples;

	size_t frames() const {
		return samples.empty() ? 0 : samples[0].size();
	}
	void resize(int nChannels, size_t nFrames) {
		channels = nChannels;
		samples.assign(nChannels, std::vector<float>(nFrames, 0.0f));
	}
};

namespace detail {
	inline uint16_t readU16(const uint8_t *p) {
		return uint16_t(p[0] | (p[1] << 8));
	}
	inline uint32_t readU32(const uint8_t *p) {
		return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
	}
	inline void writeU16(std::ostream &out, uint16_t v) {
		uint8_t b[2] = {uint8_t(v & 0xFF), uint8_t(v >> 8)};
		out.write(reinterpret_cast<const char *>(b), 2);
	}
	inline void writeU32(std::ostream &out, uint32_t v) {
		uint8_t b[4] = {uint8_t(v & 0xFF), uint8_t((v >> 8) & 0xFF), uint8_t((v >> 16) & 0xFF), uint8_t((v >> 24) & 0xFF)};
		out.write(reinterpret_cast<const char *>(b), 4);
	}
} // namespace detail

// Reads a WAV file. Returns true on success; on failure `error` (if non-null)
// receives a description.
inline bool readWav(const std::string &path, AudioFile &audio, std::string *error = nullptr) {
	using namespace detail;
	auto fail = [&](const std::string &msg) {
		if (error) *error = msg;
		return false;
	};

	std::ifstream in(path, std::ios::binary);
	if (!in) return fail("cannot open file: " + path);

	uint8_t riff[12];
	if (!in.read(reinterpret_cast<char *>(riff), 12)) return fail("file too small for RIFF header");
	if (std::memcmp(riff, "RIFF", 4) != 0 || std::memcmp(riff + 8, "WAVE", 4) != 0)
		return fail("not a RIFF/WAVE file");

	uint16_t format = 0, numChannels = 0, bitsPerSample = 0;
	uint32_t sampleRate = 0;
	bool haveFmt = false;
	std::vector<uint8_t> data;

	// Walk chunks
	while (true) {
		uint8_t hdr[8];
		if (!in.read(reinterpret_cast<char *>(hdr), 8)) break; // normal EOF
		uint32_t chunkSize = readU32(hdr + 4);

		if (std::memcmp(hdr, "fmt ", 4) == 0) {
			if (chunkSize < 16) return fail("fmt chunk too small");
			std::vector<uint8_t> fmt(chunkSize);
			if (!in.read(reinterpret_cast<char *>(fmt.data()), chunkSize)) return fail("truncated fmt chunk");
			format        = readU16(fmt.data() + 0);
			numChannels   = readU16(fmt.data() + 2);
			sampleRate    = readU32(fmt.data() + 4);
			bitsPerSample = readU16(fmt.data() + 14);
			if (format == 0xFFFE) { // WAVE_FORMAT_EXTENSIBLE: sub-format GUID at offset 24
				if (chunkSize < 40) return fail("extensible fmt chunk too small");
				format = readU16(fmt.data() + 24); // first two bytes of the GUID are the format code
			}
			haveFmt = true;
		} else if (std::memcmp(hdr, "data", 4) == 0) {
			data.resize(chunkSize);
			if (!in.read(reinterpret_cast<char *>(data.data()), chunkSize)) {
				// tolerate a truncated final data chunk
				data.resize(size_t(in.gcount()));
			}
		} else {
			in.seekg(chunkSize, std::ios::cur);
		}
		if (chunkSize & 1) in.seekg(1, std::ios::cur); // chunks are word-aligned
		if (!in) break;
	}

	if (!haveFmt) return fail("missing fmt chunk");
	if (data.empty()) return fail("missing or empty data chunk");
	if (numChannels < 1) return fail("invalid channel count");
	if (sampleRate < 1) return fail("invalid sample rate");

	const size_t bytesPerSample = bitsPerSample / 8;
	if (format == 1) { // PCM
		if (bitsPerSample != 16 && bitsPerSample != 24 && bitsPerSample != 32)
			return fail("unsupported PCM bit depth: " + std::to_string(bitsPerSample));
	} else if (format == 3) { // IEEE float
		if (bitsPerSample != 32)
			return fail("unsupported float bit depth: " + std::to_string(bitsPerSample));
	} else {
		return fail("unsupported WAV format code: " + std::to_string(format));
	}

	const size_t frameBytes = bytesPerSample * numChannels;
	const size_t nFrames = data.size() / frameBytes;

	audio.sampleRate = int(sampleRate);
	audio.resize(numChannels, nFrames);

	const uint8_t *p = data.data();
	for (size_t i = 0; i < nFrames; ++i) {
		for (int c = 0; c < numChannels; ++c, p += bytesPerSample) {
			float v = 0.0f;
			if (format == 3) { // float32
				uint32_t u = readU32(p);
				std::memcpy(&v, &u, 4);
			} else if (bitsPerSample == 16) {
				int16_t s = int16_t(readU16(p));
				v = float(s) / 32768.0f;
			} else if (bitsPerSample == 24) {
				int32_t s = int32_t(uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16));
				if (s & 0x800000) s |= ~0xFFFFFF; // sign-extend
				v = float(s) / 8388608.0f;
			} else { // 32-bit PCM
				int32_t s = int32_t(readU32(p));
				v = float(double(s) / 2147483648.0);
			}
			audio.samples[c][i] = v;
		}
	}
	return true;
}

// Writes a 32-bit IEEE float WAV file. Returns true on success.
inline bool writeWavFloat32(const std::string &path, const AudioFile &audio, std::string *error = nullptr) {
	using namespace detail;
	auto fail = [&](const std::string &msg) {
		if (error) *error = msg;
		return false;
	};

	if (audio.channels < 1 || audio.samples.size() != size_t(audio.channels))
		return fail("invalid channel setup");
	const size_t nFrames = audio.frames();
	for (const auto &ch : audio.samples) {
		if (ch.size() != nFrames) return fail("channel length mismatch");
	}

	std::ofstream out(path, std::ios::binary);
	if (!out) return fail("cannot open file for writing: " + path);

	const uint32_t dataBytes = uint32_t(nFrames * audio.channels * 4);
	const uint32_t byteRate = uint32_t(audio.sampleRate) * audio.channels * 4;
	const uint16_t blockAlign = uint16_t(audio.channels * 4);
	// RIFF size = 4 ("WAVE") + (8+16) fmt + (8+4) fact + 8 data header + data
	const uint32_t riffSize = 4 + (8 + 16) + (8 + 4) + 8 + dataBytes;

	out.write("RIFF", 4);
	writeU32(out, riffSize);
	out.write("WAVE", 4);

	out.write("fmt ", 4);
	writeU32(out, 16);
	writeU16(out, 3); // IEEE float
	writeU16(out, uint16_t(audio.channels));
	writeU32(out, uint32_t(audio.sampleRate));
	writeU32(out, byteRate);
	writeU16(out, blockAlign);
	writeU16(out, 32);

	// fact chunk (recommended for non-PCM formats)
	out.write("fact", 4);
	writeU32(out, 4);
	writeU32(out, uint32_t(nFrames));

	out.write("data", 4);
	writeU32(out, dataBytes);

	std::vector<uint8_t> row(size_t(audio.channels) * 4);
	for (size_t i = 0; i < nFrames; ++i) {
		uint8_t *p = row.data();
		for (int c = 0; c < audio.channels; ++c, p += 4) {
			float v = audio.samples[c][i];
			uint32_t u;
			std::memcpy(&u, &v, 4);
			p[0] = uint8_t(u & 0xFF);
			p[1] = uint8_t((u >> 8) & 0xFF);
			p[2] = uint8_t((u >> 16) & 0xFF);
			p[3] = uint8_t((u >> 24) & 0xFF);
		}
		out.write(reinterpret_cast<const char *>(row.data()), std::streamsize(row.size()));
	}

	if (!out) return fail("write error: " + path);
	return true;
}

} // namespace pbwav

#endif // PBWAV_WAV_IO_H
