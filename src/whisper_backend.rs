use anyhow::{Context, Result};
use log::debug;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

const MODEL_PATH: &str = "ggml-base.bin";

pub(crate) struct WhisperTranscriber {
    context: WhisperContext,
}

impl WhisperTranscriber {
    pub(crate) fn load() -> Result<Self> {
        debug!("loading whisper.cpp model {MODEL_PATH}");
        let context =
            WhisperContext::new_with_params(MODEL_PATH, WhisperContextParameters::default())
                .context("failed to load whisper.cpp context")?;

        Ok(Self { context })
    }

    pub(crate) fn transcribe(&mut self, samples: &[f32]) -> Result<String> {
        let mut state = self
            .context
            .create_state()
            .context("failed to create whisper.cpp state")?;
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

        params.set_language(Some("en"));
        params.set_detect_language(false);
        params.set_translate(false);
        params.set_no_context(true);
        params.set_single_segment(true);
        params.set_no_timestamps(true);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_suppress_blank(true);
        params.set_suppress_nst(false);

        state
            .full(params, samples)
            .context("whisper.cpp inference failed")?;

        let segment_count = state.full_n_segments();
        debug!("whisper.cpp segments={segment_count}");

        Ok(state
            .as_iter()
            .map(|segment| {
                let text = segment.to_string();
                debug!(
                    "whisper.cpp segment {}: tokens={}, no_speech={:.3}, text={text:?}",
                    segment.segment_index(),
                    segment.n_tokens(),
                    segment.no_speech_probability()
                );
                text
            })
            .collect::<Vec<_>>()
            .join(" "))
    }
}
