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

## How to run it

```
cargo run --release -- detect

[2026-06-28T19:30:01Z INFO  sounds::detect] [BLANK_AUDIO]
[2026-06-28T19:30:11Z INFO  sounds::detect] Is it pressed? No. Is it? Is it way
[2026-06-28T19:30:21Z INFO  sounds::detect] I feel like it has a there's a wave crest wake Kelvin wake that
[2026-06-28T19:30:31Z INFO  sounds::detect] and
[2026-06-28T19:30:41Z INFO  sounds::detect] They are debating with ourselves between wake crest and break That tsunami is not no, no, no, that's different
[2026-06-28T19:30:51Z INFO  sounds::detect] It can probably be used interchangeably, right? Like crests and no, because a wave is what you get behind a wave
[2026-06-28T19:31:01Z INFO  sounds::detect] I mean, is what you get behind a boat? I think it's crest. Wave crests.
[2026-06-28T19:31:11Z INFO  sounds::detect] No, the Christmas is the highest point of a wave. So arguably we should be in trouble because we're the
[2026-06-28T19:31:21Z INFO  sounds::detect] bottom part of a wave because we haven't implemented it yet. Sure, sure, why not? Great, let's do that. We call
[2026-06-28T19:31:31Z INFO  sounds::detect] this tool trough because it's at the bottom of a wave because we haven't started implementing wave yet. Perfect. Okay.
[2026-06-28T19:31:41Z INFO  sounds::detect] So what we're gonna do, there's an interesting question here also about whether we should make this print a standard out. I'm a little tempted
[2026-06-28T19:31:51Z INFO  sounds::detect] the deduceso. So I'll stood out
[2026-06-28T19:32:01Z INFO  sounds::detect] Oops, yeah. And then I'm gonna do out is out the lock. And then we're going to do
[2026-06-28T19:32:11Z INFO  sounds::detect] right this is to be mute right to out it said first you need to write riff
[2026-06-28T19:32:21Z INFO  sounds::detect] Right? First four bytes of the chunk data are an additional forces heat tag that specify the form type in a
[2026-06-28T19:32:31Z INFO  sounds::detect] all by sequence of subjects. Yeah, but do we need
[2026-06-28T19:32:41Z INFO  sounds::detect] Do we need to give the length of the chunk as well?
[2026-06-28T19:32:51Z INFO  sounds::detect] said and the size number of bytes of the chunk. So we need to know
[2026-06-28T19:33:01Z INFO  sounds::detect] the size of th
```
