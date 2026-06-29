# sounds

## 1) What

Experimental Rust repo for local audio capture and speech-to-text experiments.

The current CLI can:

- play WAV files through the default output device,
- capture microphone input with CPAL,
- downmix/resample captured audio to 16 kHz mono,
- transcribe live chunks with whisper.cpp through `whisper-rs`.

This is not a polished app or library. The code is being iterated on to explore audio capture, preprocessing, and local Whisper inference behavior.

Local model files and audio samples are intentionally ignored by git.

## Models

Live transcription uses whisper.cpp models through `whisper-rs`. Model files are not downloaded by the app and must exist locally.

The default model path is:

```
ggml-medium.bin
```

Supported local model files are the whisper.cpp GGML models, for example:

- `ggml-base.bin`
- `ggml-small.bin`
- `ggml-medium.bin`

Download them from the whisper.cpp model repository:

```
curl -L -o ggml-small.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin
curl -L -o ggml-medium.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin
```

Use a different model with:

```
cargo run --release -- detect --model ggml-small.bin
```

On macOS with Metal, enable the optional Metal backend:

```
cargo run --release --features metal -- detect
```

## How to run it

```
cargo run --release -- detect

[2026-06-29T06:02:11Z INFO  sounds::detect] Is James May knighted?
[2026-06-29T06:02:14Z INFO  sounds::detect] why would I be
[2026-06-29T06:02:17Z INFO  sounds::detect] Is James May neurodivergent?
[2026-06-29T06:02:20Z INFO  sounds::detect] now this is a word
[2026-06-29T06:02:23Z INFO  sounds::detect] Neurodivergent is a polite way of saying a bit odd.
[2026-06-29T06:02:26Z INFO  sounds::detect] a bit on the spectrum.
[2026-06-29T06:02:29Z INFO  sounds::detect] s a fashionable word at the moment isn't it?
[2026-06-29T06:02:38Z INFO  sounds::detect] A lot of people are neurodivergent. So I'm going to say yes, I am neurodivergent. Is James May married? No, not technically
```
