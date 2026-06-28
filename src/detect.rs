use std::path::Path;

use crate::{
    audio::{audio_stats, resample_mono, write_wav_i16},
    capture::AudioCapture,
    whisper_backend::WhisperTranscriber,
};

use anyhow::{Context, Result};
use log::{debug, info, warn};

const WHISPER_SAMPLE_RATE: u32 = 16_000;
const TRANSCRIBE_CHUNK_SECONDS: usize = 10;
const MAX_LIVE_GAIN: f32 = 3.0;
const LIVE_TARGET_PEAK: f32 = 0.5;
const MIN_TRANSCRIBE_RMS: f32 = 0.006;

pub(crate) fn run_detect() -> Result<()> {
    DetectionSession::new()?.run()
}

struct DetectionSession {
    capture: AudioCapture,
    transcriber: WhisperTranscriber,
    pending: Vec<f32>,
    sample_rate: u32,
    chunk_samples: usize,
    wrote_debug_capture: bool,
}

impl DetectionSession {
    fn new() -> Result<Self> {
        let capture = AudioCapture::new()?;
        let sample_rate = capture.sample_rate();
        let channels = capture.channels();
        let sample_format = capture.sample_format();
        debug!(
            "using input config: format={sample_format}, sample_rate={sample_rate}, channels={channels}"
        );

        debug!("input sample rate {sample_rate} Hz");
        let transcriber = WhisperTranscriber::load().context("failed to load whisper.cpp model")?;
        let chunk_samples = sample_rate as usize * TRANSCRIBE_CHUNK_SECONDS;

        Ok(Self {
            capture,
            transcriber,
            pending: Vec::with_capacity(chunk_samples * 2),
            sample_rate,
            chunk_samples,
            wrote_debug_capture: false,
        })
    }

    fn run(&mut self) -> Result<()> {
        self.capture.start()?;

        while let Ok(samples) = self.capture.recv() {
            self.pending.extend(samples);
            self.transcribe_pending_chunks();
        }

        Ok(())
    }

    fn transcribe_pending_chunks(&mut self) {
        while self.pending.len() >= self.chunk_samples {
            let chunk = self.pending.drain(..self.chunk_samples).collect::<Vec<_>>();
            let queued_seconds = self.pending.len() as f64 / self.sample_rate as f64;
            if queued_seconds >= 1.0 {
                debug!("audio backlog: {queued_seconds:.1}s queued");
            }
            let debug_wav_path = self.next_debug_wav_path();
            self.transcribe_utterance(&chunk, debug_wav_path);
        }
    }

    fn next_debug_wav_path(&mut self) -> Option<&'static Path> {
        if self.wrote_debug_capture {
            None
        } else {
            self.wrote_debug_capture = true;
            Some(Path::new("/tmp/debug_capture.wav"))
        }
    }

    fn transcribe_utterance(&mut self, utterance: &[f32], debug_wav_path: Option<&Path>) {
        if utterance.is_empty() {
            return;
        }

        let (rms, peak) = audio_stats(utterance);
        debug!(
            "chunk: {} samples @ {} Hz, rms={rms:.6}, peak={peak:.6}",
            utterance.len(),
            self.sample_rate
        );

        let mut audio = resample_mono(utterance, self.sample_rate, WHISPER_SAMPLE_RATE);
        normalize_live_chunk(&mut audio);
        let (normalized_rms, normalized_peak) = audio_stats(&audio);
        debug!(
            "resampled: {} samples @ {} Hz, rms={normalized_rms:.6}, peak={normalized_peak:.6}",
            audio.len(),
            WHISPER_SAMPLE_RATE
        );

        if normalized_rms < MIN_TRANSCRIBE_RMS {
            debug!("skipping low-energy chunk, rms={normalized_rms:.6}");
            return;
        }

        if let Some(path) = debug_wav_path {
            match write_wav_i16(path, &audio, WHISPER_SAMPLE_RATE) {
                Ok(()) => debug!("wrote debug capture to {}", path.display()),
                Err(error) => warn!("failed to write debug capture: {error}"),
            }
        }

        match self.transcriber.transcribe(&audio) {
            Ok(text) => {
                let text = text.trim();
                if !text.is_empty() {
                    info!("{text}");
                } else {
                    debug!("<empty transcription>");
                }
            }
            Err(error) => warn!("transcription error: {error}"),
        }
    }
}

fn normalize_live_chunk(samples: &mut [f32]) {
    let peak = samples
        .iter()
        .map(|sample| sample.abs())
        .fold(0.0, f32::max);

    if peak == 0.0 {
        return;
    }

    let gain = (LIVE_TARGET_PEAK / peak).min(MAX_LIVE_GAIN);
    for sample in samples {
        *sample = (*sample * gain).clamp(-1.0, 1.0);
    }
}
