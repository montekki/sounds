use std::sync::mpsc;

use anyhow::{Context, Result, anyhow};
use audio_blocks::{AudioBlockOps, InterleavedView, Mono};
use cpal::{
    SampleFormat, Stream, StreamConfig,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};

pub(crate) struct AudioCapture {
    sample_rate: u32,
    channels: usize,
    sample_format: SampleFormat,
    stream: Stream,
    rx: mpsc::Receiver<Vec<f32>>,
}

impl AudioCapture {
    pub(crate) fn new() -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| anyhow!("no device found"))?;
        let supported_configs = device
            .supported_input_configs()
            .context("failed to query supported input configs")?
            .collect::<Vec<_>>();

        for config_range in &supported_configs {
            println!(
                "{:?}: {}-{} Hz",
                config_range.sample_format(),
                config_range.min_sample_rate(),
                config_range.max_sample_rate(),
            );
        }

        let supported_config = supported_configs
            .iter()
            .find(|config| {
                config.sample_format() == SampleFormat::F32
                    && config.min_sample_rate() <= 48_000
                    && config.max_sample_rate() >= 48_000
            })
            .or_else(|| {
                supported_configs
                    .iter()
                    .find(|config| config.sample_format() == SampleFormat::F32)
            })
            .ok_or_else(|| anyhow!("no f32 input config found"))?;

        let sample_format = supported_config.sample_format();
        let selected_sample_rate = if supported_config.min_sample_rate() <= 48_000
            && supported_config.max_sample_rate() >= 48_000
        {
            48_000
        } else {
            supported_config.max_sample_rate()
        };

        println!("proceeding with config {supported_config:?} at {selected_sample_rate} Hz");
        let config: StreamConfig = supported_config
            .try_with_sample_rate(selected_sample_rate)
            .ok_or_else(|| anyhow!("failed to try with sample rate {selected_sample_rate}"))?
            .into();
        let sample_rate = config.sample_rate;
        let channels = config.channels as usize;
        let (tx, rx) = mpsc::channel::<Vec<f32>>();

        let stream = match sample_format {
            SampleFormat::F32 => device
                .build_input_stream(
                    config,
                    move |data: &[f32], _| {
                        let _ = tx.send(interleaved_to_mono_f32(data, channels));
                    },
                    |error| eprintln!("input stream error: {error}"),
                    None,
                )
                .context("failed to build input stream")?,
            sample_format => {
                return Err(anyhow!("unsupported input sample format {sample_format}"));
            }
        };

        Ok(Self {
            sample_rate,
            channels,
            sample_format,
            stream,
            rx,
        })
    }

    pub(crate) fn start(&self) -> Result<()> {
        self.stream.play().context("failed to start input stream")
    }

    pub(crate) fn recv(&self) -> Result<Vec<f32>, mpsc::RecvError> {
        self.rx.recv()
    }

    pub(crate) fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub(crate) fn channels(&self) -> usize {
        self.channels
    }

    pub(crate) fn sample_format(&self) -> SampleFormat {
        self.sample_format
    }
}

fn interleaved_to_mono_f32(data: &[f32], num_channels: usize) -> Vec<f32> {
    let frames = data.len() / num_channels;
    let interleaved = InterleavedView::from_slice(data, num_channels as u16);
    let mut mono = Mono::new(frames);
    interleaved.mix_to_mono_exact(&mut mono.view_mut());
    mono.samples().to_vec()
}
