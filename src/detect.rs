use std::path::{Path, PathBuf};

use crate::{
    audio::{audio_stats, resample_mono, write_wav_i16},
    capture::AudioCapture,
    whisper_backend::WhisperTranscriber,
};

use anyhow::{Context, Result};
use log::{debug, info, warn};

const WHISPER_SAMPLE_RATE: u32 = 16_000;
const TRANSCRIBE_WINDOW_SECONDS: usize = 10;
const TRANSCRIBE_STEP_SECONDS: usize = 3;
const MAX_LIVE_GAIN: f32 = 3.0;
const LIVE_TARGET_PEAK: f32 = 0.5;
const MIN_RAW_TRANSCRIBE_RMS: f32 = 0.003;
const SILENT_WINDOWS_BEFORE_RESET: usize = 2;

pub(crate) fn run_detect(model_path: PathBuf) -> Result<()> {
    DetectionSession::new(model_path)?.run()
}

struct DetectionSession {
    capture: AudioCapture,
    transcriber: WhisperTranscriber,
    emitted_text: String,
    silent_windows: usize,
    audio_window: Vec<f32>,
    samples_since_transcription: usize,
    sample_rate: u32,
    window_samples: usize,
    step_samples: usize,
    wrote_debug_capture: bool,
}

impl DetectionSession {
    fn new(model_path: PathBuf) -> Result<Self> {
        let capture = AudioCapture::new()?;
        let sample_rate = capture.sample_rate();
        let channels = capture.channels();
        let sample_format = capture.sample_format();
        debug!(
            "using input config: format={sample_format}, sample_rate={sample_rate}, channels={channels}"
        );

        debug!("input sample rate {sample_rate} Hz");
        let transcriber =
            WhisperTranscriber::load(&model_path).context("failed to load whisper.cpp model")?;
        let window_samples = sample_rate as usize * TRANSCRIBE_WINDOW_SECONDS;
        let step_samples = sample_rate as usize * TRANSCRIBE_STEP_SECONDS;

        Ok(Self {
            capture,
            transcriber,
            emitted_text: String::new(),
            silent_windows: 0,
            audio_window: Vec::with_capacity(window_samples),
            samples_since_transcription: 0,
            sample_rate,
            window_samples,
            step_samples,
            wrote_debug_capture: false,
        })
    }

    fn run(&mut self) -> Result<()> {
        self.capture.start()?;

        while let Ok(samples) = self.capture.recv() {
            self.push_samples(samples);
        }

        Ok(())
    }

    fn push_samples(&mut self, samples: Vec<f32>) {
        self.samples_since_transcription += samples.len();
        self.audio_window.extend(samples);

        if self.audio_window.len() > self.window_samples {
            let excess_samples = self.audio_window.len() - self.window_samples;
            self.audio_window.drain(..excess_samples);
        }

        if self.samples_since_transcription >= self.step_samples
            && self.audio_window.len() >= self.window_samples
        {
            self.samples_since_transcription %= self.step_samples;
            let window = self.audio_window.clone();
            let debug_wav_path = self.next_debug_wav_path();
            self.transcribe_utterance(&window, debug_wav_path);
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

        if rms < MIN_RAW_TRANSCRIBE_RMS {
            self.note_silent_window(rms);
            return;
        }
        self.silent_windows = 0;

        let mut audio = resample_mono(utterance, self.sample_rate, WHISPER_SAMPLE_RATE);
        normalize_live_chunk(&mut audio);
        let (normalized_rms, normalized_peak) = audio_stats(&audio);
        debug!(
            "resampled: {} samples @ {} Hz, rms={normalized_rms:.6}, peak={normalized_peak:.6}",
            audio.len(),
            WHISPER_SAMPLE_RATE
        );

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
                    self.emit_new_text(text);
                } else {
                    debug!("<empty transcription>");
                }
            }
            Err(error) => warn!("transcription error: {error}"),
        }
    }

    fn note_silent_window(&mut self, rms: f32) {
        self.silent_windows += 1;
        debug!("skipping silent window, raw rms={rms:.6}");

        if self.silent_windows >= SILENT_WINDOWS_BEFORE_RESET && !self.emitted_text.is_empty() {
            debug!(
                "resetting emitted text after {} silent windows",
                self.silent_windows
            );
            self.emitted_text.clear();
        }
    }

    fn emit_new_text(&mut self, text: &str) {
        let new_text = new_text_after_overlap(&self.emitted_text, text);
        let new_text = trim_leading_non_alphanumeric(new_text).trim();

        if new_text.is_empty() {
            debug!("transcription fully overlaps previous output");
            return;
        }

        if !new_text.chars().any(char::is_alphanumeric) {
            debug!("suppressing punctuation-only transcription: {new_text:?}");
            return;
        }

        if !self.emitted_text.is_empty() {
            self.emitted_text.push(' ');
        }
        self.emitted_text.push_str(new_text);
        info!("{new_text}");
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

fn trim_leading_non_alphanumeric(text: &str) -> &str {
    text.trim_start_matches(|character: char| !character.is_alphanumeric())
}

fn new_text_after_overlap<'a>(emitted_text: &str, text: &'a str) -> &'a str {
    const MIN_OVERLAP_WORDS: usize = 4;
    const MIN_OVERLAP_SIMILARITY: f64 = 0.78;

    let emitted_words = normalized_words(emitted_text);
    let text_words = indexed_normalized_words(text);
    if emitted_words.is_empty() || text_words.len() < MIN_OVERLAP_WORDS {
        return text;
    }

    for overlap_len in (MIN_OVERLAP_WORDS..=text_words.len()).rev() {
        for start in 0..=text_words.len() - overlap_len {
            let overlap = &text_words[start..start + overlap_len];
            if best_word_sequence_similarity(&emitted_words, overlap) >= MIN_OVERLAP_SIMILARITY {
                let suffix_start = overlap
                    .last()
                    .map(|word| word.end)
                    .unwrap_or_default()
                    .min(text.len());
                return trim_leading_non_alphanumeric(&text[suffix_start..]);
            }
        }
    }

    text
}

fn best_word_sequence_similarity(haystack: &[String], needle: &[IndexedWord]) -> f64 {
    if needle.len() > haystack.len() {
        return 0.0;
    }

    haystack
        .windows(needle.len())
        .map(|window| word_sequence_similarity(window, needle))
        .fold(0.0, f64::max)
}

fn word_sequence_similarity(left: &[String], right: &[IndexedWord]) -> f64 {
    let left = left.join(" ");
    let right = right
        .iter()
        .map(|word| word.word.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    strsim::normalized_levenshtein(&left, &right)
}

fn normalized_words(text: &str) -> Vec<String> {
    indexed_normalized_words(text)
        .into_iter()
        .map(|word| word.word)
        .collect()
}

fn indexed_normalized_words(text: &str) -> Vec<IndexedWord> {
    let mut words = Vec::new();
    let mut current = String::new();
    let mut current_start = 0;

    for (index, character) in text.char_indices() {
        if character.is_alphanumeric() {
            if current.is_empty() {
                current_start = index;
            }
            current.extend(character.to_lowercase());
        } else if !current.is_empty() {
            words.push(IndexedWord {
                word: std::mem::take(&mut current),
                start: current_start,
                end: index,
            });
        }
    }

    if !current.is_empty() {
        words.push(IndexedWord {
            word: current,
            start: current_start,
            end: text.len(),
        });
    }

    words
}

struct IndexedWord {
    word: String,
    #[allow(dead_code)]
    start: usize,
    end: usize,
}

#[cfg(test)]
mod tests {
    use super::new_text_after_overlap;

    #[test]
    fn emits_only_suffix_after_seen_prefix() {
        let emitted = "I don't know about you but to me there is something about this image";
        let text = "something about this image that's really creepy. It shows an undeveloped marsh with a temple";

        assert_eq!(
            new_text_after_overlap(emitted, text).trim(),
            "that's really creepy. It shows an undeveloped marsh with a temple"
        );
    }

    #[test]
    fn emits_suffix_after_noisy_fuzzy_overlap() {
        let emitted = "Why can't I write bytes? Why doesn't it let me write bytes?";
        let text = "? Bites? Why doesn't it let me write bites? Alright, fine. We'll do this then.";

        assert_eq!(
            new_text_after_overlap(emitted, text).trim(),
            "Alright, fine. We'll do this then."
        );
    }

    #[test]
    fn keeps_text_without_enough_overlap() {
        let text = "This is unrelated output";
        assert_eq!(new_text_after_overlap("already emitted words", text), text);
    }
}
