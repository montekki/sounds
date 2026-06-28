# sounds

Experimental Rust repo for local audio capture and speech-to-text experiments.

The current CLI can:

- play WAV files through the default output device,
- capture microphone input with CPAL,
- downmix/resample captured audio to 16 kHz mono,
- transcribe live chunks with Candle Whisper.

This is not a polished app or library. The code is being iterated on to explore audio capture, preprocessing, and local Whisper inference behavior.

Local model files and audio samples are intentionally ignored by git.
