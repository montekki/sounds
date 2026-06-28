use anyhow::{Context, Result, anyhow};
use candle::{Device, IndexOp, Tensor};
use candle_nn::{VarBuilder, ops::softmax};
use candle_transformers::models::whisper::{self as whisper, Config, audio};
use hf_hub::{Repo, RepoType, api::sync::Api};
use tokenizers::Tokenizer;

const MODEL_ID: &str = "openai/whisper-base";
const MODEL_REVISION: &str = "refs/pr/22";
const NO_REPEAT_NGRAM_SIZE: usize = 3;
const FALLBACK_TEMPERATURES: [f64; 6] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
const MAX_REPETITION_RATIO: f64 = 0.18;
const LANGUAGES: [(&str, &str); 99] = [
    ("en", "english"),
    ("zh", "chinese"),
    ("de", "german"),
    ("es", "spanish"),
    ("ru", "russian"),
    ("ko", "korean"),
    ("fr", "french"),
    ("ja", "japanese"),
    ("pt", "portuguese"),
    ("tr", "turkish"),
    ("pl", "polish"),
    ("ca", "catalan"),
    ("nl", "dutch"),
    ("ar", "arabic"),
    ("sv", "swedish"),
    ("it", "italian"),
    ("id", "indonesian"),
    ("hi", "hindi"),
    ("fi", "finnish"),
    ("vi", "vietnamese"),
    ("he", "hebrew"),
    ("uk", "ukrainian"),
    ("el", "greek"),
    ("ms", "malay"),
    ("cs", "czech"),
    ("ro", "romanian"),
    ("da", "danish"),
    ("hu", "hungarian"),
    ("ta", "tamil"),
    ("no", "norwegian"),
    ("th", "thai"),
    ("ur", "urdu"),
    ("hr", "croatian"),
    ("bg", "bulgarian"),
    ("lt", "lithuanian"),
    ("la", "latin"),
    ("mi", "maori"),
    ("ml", "malayalam"),
    ("cy", "welsh"),
    ("sk", "slovak"),
    ("te", "telugu"),
    ("fa", "persian"),
    ("lv", "latvian"),
    ("bn", "bengali"),
    ("sr", "serbian"),
    ("az", "azerbaijani"),
    ("sl", "slovenian"),
    ("kn", "kannada"),
    ("et", "estonian"),
    ("mk", "macedonian"),
    ("br", "breton"),
    ("eu", "basque"),
    ("is", "icelandic"),
    ("hy", "armenian"),
    ("ne", "nepali"),
    ("mn", "mongolian"),
    ("bs", "bosnian"),
    ("kk", "kazakh"),
    ("sq", "albanian"),
    ("sw", "swahili"),
    ("gl", "galician"),
    ("mr", "marathi"),
    ("pa", "punjabi"),
    ("si", "sinhala"),
    ("km", "khmer"),
    ("sn", "shona"),
    ("yo", "yoruba"),
    ("so", "somali"),
    ("af", "afrikaans"),
    ("oc", "occitan"),
    ("ka", "georgian"),
    ("be", "belarusian"),
    ("tg", "tajik"),
    ("sd", "sindhi"),
    ("gu", "gujarati"),
    ("am", "amharic"),
    ("yi", "yiddish"),
    ("lo", "lao"),
    ("uz", "uzbek"),
    ("fo", "faroese"),
    ("ht", "haitian creole"),
    ("ps", "pashto"),
    ("tk", "turkmen"),
    ("nn", "nynorsk"),
    ("mt", "maltese"),
    ("sa", "sanskrit"),
    ("lb", "luxembourgish"),
    ("my", "myanmar"),
    ("bo", "tibetan"),
    ("tl", "tagalog"),
    ("mg", "malagasy"),
    ("as", "assamese"),
    ("tt", "tatar"),
    ("haw", "hawaiian"),
    ("ln", "lingala"),
    ("ha", "hausa"),
    ("ba", "bashkir"),
    ("jw", "javanese"),
    ("su", "sundanese"),
];

pub(crate) struct CandleWhisper {
    model: whisper::model::Whisper,
    tokenizer: Tokenizer,
    device: Device,
    mel_filters: Vec<f32>,
    suppress_tokens: Tensor,
    sot_token: u32,
    transcribe_token: u32,
    eot_token: u32,
    no_speech_token: u32,
    no_timestamps_token: u32,
    language_tokens: Vec<(u32, &'static str)>,
}

impl CandleWhisper {
    pub(crate) fn load() -> Result<Self> {
        let device = candle_device()?;
        let api = Api::new().context("failed to create Hugging Face API client")?;
        let repo = api.repo(Repo::with_revision(
            MODEL_ID.to_owned(),
            RepoType::Model,
            MODEL_REVISION.to_owned(),
        ));

        eprintln!("loading Candle Whisper model {MODEL_ID}@{MODEL_REVISION}");
        let config_path = repo
            .get("config.json")
            .context("failed to fetch config.json")?;
        let tokenizer_path = repo
            .get("tokenizer.json")
            .context("failed to fetch tokenizer.json")?;
        let weights_path = repo
            .get("model.safetensors")
            .context("failed to fetch model.safetensors")?;

        let config: Config = serde_json::from_str(
            &std::fs::read_to_string(config_path).context("failed to read config.json")?,
        )
        .context("failed to parse config.json")?;
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;
        let weights = std::fs::read(weights_path).context("failed to read model.safetensors")?;
        let vb = VarBuilder::from_buffered_safetensors(weights, whisper::DTYPE, &device)
            .context("failed to load safetensors")?;
        let model = whisper::model::Whisper::load(&vb, config.clone())
            .context("failed to load Candle Whisper model")?;

        let no_timestamps_token = token_id(&tokenizer, whisper::NO_TIMESTAMPS_TOKEN)?;
        let suppress_tokens = config
            .suppress_tokens
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>();
        let suppress_tokens = (0..config.vocab_size as u32)
            .map(|token| {
                if suppress_tokens.contains(&token) {
                    f32::NEG_INFINITY
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>();
        let suppress_tokens = Tensor::new(suppress_tokens.as_slice(), &device)?;

        let no_speech_token = whisper::NO_SPEECH_TOKENS
            .iter()
            .find_map(|token| token_id(&tokenizer, token).ok())
            .ok_or_else(|| anyhow!("failed to find no-speech token"))?;
        let language_tokens = LANGUAGES
            .iter()
            .filter_map(|(code, name)| {
                token_id(&tokenizer, &format!("<|{code}|>"))
                    .ok()
                    .map(|token| (token, *name))
            })
            .collect::<Vec<_>>();

        eprintln!(
            "candle model: device={:?}, vocab={}, layers={}/{}, d_model={}, mels={}, audio_ctx={}, text_ctx={}",
            device,
            config.vocab_size,
            config.encoder_layers,
            config.decoder_layers,
            config.d_model,
            config.num_mel_bins,
            config.max_source_positions,
            config.max_target_positions
        );

        Ok(Self {
            model,
            sot_token: token_id(&tokenizer, whisper::SOT_TOKEN)?,
            transcribe_token: token_id(&tokenizer, whisper::TRANSCRIBE_TOKEN)?,
            eot_token: token_id(&tokenizer, whisper::EOT_TOKEN)?,
            no_speech_token,
            no_timestamps_token,
            language_tokens,
            tokenizer,
            device,
            mel_filters: mel_filters(config.num_mel_bins)?,
            suppress_tokens,
        })
    }

    pub(crate) fn transcribe(&mut self, samples: &[f32]) -> Result<String> {
        let mel = audio::pcm_to_mel(&self.model.config, samples, &self.mel_filters);
        let mel_len = mel.len() / self.model.config.num_mel_bins;
        let mel = Tensor::from_vec(
            mel,
            (1, self.model.config.num_mel_bins, mel_len),
            &self.device,
        )?;
        eprintln!("candle mel: {:?}", mel.dims());

        let (_, _, padded_frames) = mel.dims3()?;
        let content_frames = (samples.len() / whisper::HOP_LENGTH).min(padded_frames);
        let mut seek = 0;
        let mut texts = Vec::new();
        while seek < content_frames {
            let segment_size = (content_frames - seek).min(whisper::N_FRAMES);
            let segment = mel.narrow(2, seek, segment_size)?;
            let start = seek as f64 * whisper::HOP_LENGTH as f64 / whisper::SAMPLE_RATE as f64;
            let end = (seek + segment_size) as f64 * whisper::HOP_LENGTH as f64
                / whisper::SAMPLE_RATE as f64;
            let (language_token, language_name) = self.detect_language(&segment)?;
            let result = self.decode_segment_with_fallback(&segment, language_token)?;
            seek += segment_size;

            if result.no_speech_prob > whisper::NO_SPEECH_THRESHOLD {
                eprintln!(
                    "candle: skipping no-speech segment {:.1}s--{:.1}s, p={:.3}",
                    start, end, result.no_speech_prob
                );
                continue;
            }

            let text = result.text.trim();
            if !text.is_empty() {
                eprintln!(
                    "candle segment {:.1}s--{:.1}s, language={language_name}, no_speech={:.3}",
                    start, end, result.no_speech_prob
                );
                texts.push(text.to_owned());
            }
        }

        Ok(texts.join(" "))
    }

    fn detect_language(&mut self, mel: &Tensor) -> Result<(u32, &'static str)> {
        let audio_features = self.model.encoder.forward(mel, true)?;
        let tokens = Tensor::new(&[[self.sot_token]], &self.device)?;
        let language_token_ids = self
            .language_tokens
            .iter()
            .map(|(token, _)| *token)
            .collect::<Vec<_>>();
        let language_token_ids = Tensor::new(language_token_ids.as_slice(), &self.device)?;
        let ys = self.model.decoder.forward(&tokens, &audio_features, true)?;
        let logits = self.model.decoder.final_linear(&ys.i(..1)?)?.i(0)?.i(0)?;
        let logits = logits.index_select(&language_token_ids, 0)?;
        let probs = softmax(&logits, candle::D::Minus1)?.to_vec1::<f32>()?;
        let (index, probability) = probs
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .ok_or_else(|| anyhow!("no language probabilities"))?;
        let (token, name) = self.language_tokens[index];
        eprintln!("detected language={name}, p={probability:.3}");
        Ok((token, name))
    }

    fn decode_segment_with_fallback(
        &mut self,
        mel: &Tensor,
        language_token: u32,
    ) -> Result<DecodeResult> {
        let mut last_result = None;

        for temperature in FALLBACK_TEMPERATURES {
            let result = self.decode_segment(mel, language_token, temperature)?;
            let repetition_ratio = repetition_ratio(&result.tokens);
            if repetition_ratio <= MAX_REPETITION_RATIO
                || result.no_speech_prob > whisper::NO_SPEECH_THRESHOLD
            {
                return Ok(result);
            }

            eprintln!(
                "retrying decode: repetition_ratio={repetition_ratio:.3}, temperature={temperature:.1}"
            );
            last_result = Some(result);
        }

        last_result.ok_or_else(|| anyhow!("decode fallback did not run"))
    }

    fn decode_segment(
        &mut self,
        mel: &Tensor,
        language_token: u32,
        temperature: f64,
    ) -> Result<DecodeResult> {
        let audio_features = self.model.encoder.forward(mel, true)?;
        let sample_len = self.model.config.max_target_positions / 2;
        let mut tokens = vec![
            self.sot_token,
            language_token,
            self.transcribe_token,
            self.no_timestamps_token,
        ];
        let mut no_speech_prob = f64::NAN;

        for step in 0..sample_len {
            let tokens_t = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
            let ys = self
                .model
                .decoder
                .forward(&tokens_t, &audio_features, step == 0)?;

            if step == 0 {
                let logits = self.model.decoder.final_linear(&ys.i(..1)?)?.i(0)?.i(0)?;
                no_speech_prob = softmax(&logits, 0)?
                    .i(self.no_speech_token as usize)?
                    .to_scalar::<f32>()? as f64;
            }

            let (_, seq_len, _) = ys.dims3()?;
            let logits = self
                .model
                .decoder
                .final_linear(&ys.i((..1, seq_len - 1..))?)?
                .i(0)?
                .i(0)?
                .broadcast_add(&self.suppress_tokens)?;
            let mut logits = logits.to_vec1::<f32>()?;
            apply_no_repeat_ngram(&mut logits, &tokens, NO_REPEAT_NGRAM_SIZE);
            let next_token = next_token(&logits, temperature)?;

            if next_token == self.eot_token {
                break;
            }
            tokens.push(next_token);
        }

        let text = self
            .tokenizer
            .decode(&tokens, true)
            .map_err(anyhow::Error::msg)?;
        Ok(DecodeResult {
            tokens,
            text,
            no_speech_prob,
        })
    }
}

fn candle_device() -> Result<Device> {
    if candle::utils::metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        Ok(Device::Cpu)
    }
}

fn mel_filters(num_mel_bins: usize) -> Result<Vec<f32>> {
    let bytes = match num_mel_bins {
        80 => include_bytes!("assets/melfilters.bytes").as_slice(),
        128 => include_bytes!("assets/melfilters128.bytes").as_slice(),
        bins => return Err(anyhow!("unsupported mel bin count {bins}")),
    };

    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("chunk has 4 bytes")))
        .collect())
}

fn next_token(logits: &[f32], temperature: f64) -> Result<u32> {
    if logits.is_empty() {
        return Err(anyhow!("empty logits"));
    }

    if temperature == 0.0 {
        return logits
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| index as u32)
            .ok_or_else(|| anyhow!("empty logits"));
    }

    let scaled = logits
        .iter()
        .map(|logit| (*logit as f64 / temperature).exp())
        .collect::<Vec<_>>();
    let total = scaled.iter().sum::<f64>();
    if total == 0.0 || !total.is_finite() {
        return next_token(logits, 0.0);
    }

    let mut threshold = rand::random::<f64>() * total;
    for (index, weight) in scaled.iter().enumerate() {
        threshold -= weight;
        if threshold <= 0.0 {
            return Ok(index as u32);
        }
    }

    Ok((scaled.len() - 1) as u32)
}

fn apply_no_repeat_ngram(logits: &mut [f32], tokens: &[u32], ngram_size: usize) {
    if ngram_size == 0 || tokens.len() + 1 < ngram_size {
        return;
    }

    let prefix_len = ngram_size - 1;
    let prefix_start = tokens.len() - prefix_len;
    let prefix = &tokens[prefix_start..];

    for window in tokens.windows(ngram_size) {
        if &window[..prefix_len] == prefix {
            let token = window[prefix_len] as usize;
            if let Some(logit) = logits.get_mut(token) {
                *logit = f32::NEG_INFINITY;
            }
        }
    }
}

fn repetition_ratio(tokens: &[u32]) -> f64 {
    if tokens.len() < 2 {
        return 0.0;
    }

    let repeated = tokens
        .windows(2)
        .filter(|window| window[0] == window[1])
        .count();
    repeated as f64 / (tokens.len() - 1) as f64
}

struct DecodeResult {
    tokens: Vec<u32>,
    text: String,
    no_speech_prob: f64,
}

fn token_id(tokenizer: &Tokenizer, token: &str) -> Result<u32> {
    tokenizer
        .token_to_id(token)
        .ok_or_else(|| anyhow!("no token id for {token}"))
}
