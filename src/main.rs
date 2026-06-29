use std::{path::PathBuf, thread, time::Duration};

use anyhow::{Context, Result, anyhow};
use clap::{Parser, Subcommand};
use cpal::{
    FromSample, Sample, SampleFormat, SizedSample, Stream, StreamConfig,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};
use env_logger::Env;
use log::{debug, warn};
use wavers::Wav;

mod audio;
mod capture;
mod detect;
mod whisper_backend;

#[derive(Subcommand)]
enum Command {
    Play {
        #[arg(short)]
        wav_file: PathBuf,
    },
    Detect {
        #[arg(short, long, default_value = "ggml-medium.bin")]
        model: PathBuf,
    },
}

#[derive(Parser)]
#[command(version)]
struct Args {
    #[command(subcommand)]
    subcommand: Command,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();
    whisper_rs::install_logging_hooks();

    let args = Args::parse();
    match args.subcommand {
        Command::Play { wav_file } => run_play(wav_file),
        Command::Detect { model } => detect::run_detect(model),
    }
}

fn run_play(wav_file: PathBuf) -> Result<()> {
    let mut wav: Wav<f32> = Wav::from_path(wav_file)?;
    let wav_sample_rate = wav.sample_rate() as u32;
    let wav_channels = wav.n_channels() as usize;
    let samples = wav.read()?.to_vec();
    let host = cpal::default_host();

    let device = host
        .default_output_device()
        .ok_or(anyhow!("no output device found"))?;

    let device_id = device
        .id()
        .map(|id| id.to_string())
        .unwrap_or_else(|_| "unknown".to_owned());
    let supported_config = device
        .default_output_config()
        .context("failed to get default output config")?;
    let sample_format = supported_config.sample_format();

    let config: StreamConfig = supported_config.into();
    let output_channels = config.channels as usize;
    let output_sample_rate = config.sample_rate;
    let samples = convert_to_output_config(
        &samples,
        wav_channels,
        wav_sample_rate,
        output_channels,
        output_sample_rate,
    );

    let play_time = Duration::from_secs_f64(
        samples.len() as f64 / output_sample_rate as f64 / output_channels as f64,
    );
    debug!(
        "playing on {device_id}: {output_channels} channels, {output_sample_rate} Hz, {sample_format}"
    );
    let stream = match sample_format {
        SampleFormat::F32 => build_stream::<f32>(&device, config, samples)?,
        SampleFormat::I16 => build_stream::<i16>(&device, config, samples)?,
        SampleFormat::U16 => build_stream::<u16>(&device, config, samples)?,
        sample_format => return Err(anyhow!("unsupported output sample format: {sample_format}")),
    };

    stream.play().context("failed to start output stream")?;
    thread::sleep(play_time);

    Ok(())
}
fn build_stream<T>(device: &cpal::Device, config: StreamConfig, samples: Vec<f32>) -> Result<Stream>
where
    T: Sample + SizedSample + FromSample<f32>,
{
    let mut sample_index = 0;

    device
        .build_output_stream(
            config,
            move |data: &mut [T], _| {
                for (data, sample) in data.iter_mut().zip(
                    samples
                        .iter()
                        .skip(sample_index)
                        .chain(std::iter::repeat(&0.0)),
                ) {
                    *data = T::from_sample(*sample);
                    sample_index += 1;
                }
            },
            |error| warn!("stream error: {error}"),
            None,
        )
        .context("failed to build output stream")
}

fn convert_to_output_config(
    input: &[f32],
    input_channels: usize,
    input_sample_rate: u32,
    output_channels: usize,
    output_sample_rate: u32,
) -> Vec<f32> {
    let input_frames = input.len() / input_channels;
    let output_frames = (input_frames as f64 * output_sample_rate as f64 / input_sample_rate as f64)
        .ceil() as usize;
    let mut output = Vec::with_capacity(output_frames * output_channels);

    for output_frame in 0..output_frames {
        let input_frame_position =
            output_frame as f64 * input_sample_rate as f64 / output_sample_rate as f64;
        let previous_frame = input_frame_position.floor() as usize;
        let next_frame = (previous_frame + 1).min(input_frames.saturating_sub(1));
        let lerp = (input_frame_position - previous_frame as f64) as f32;

        for output_channel in 0..output_channels {
            let previous_sample =
                sample_for_channel(input, input_channels, previous_frame, output_channel);
            let next_sample = sample_for_channel(input, input_channels, next_frame, output_channel);
            output.push(previous_sample + (next_sample - previous_sample) * lerp);
        }
    }

    output
}

fn sample_for_channel(
    samples: &[f32],
    input_channels: usize,
    frame: usize,
    output_channel: usize,
) -> f32 {
    if input_channels == 1 {
        return samples[frame];
    }

    if output_channel < input_channels {
        return samples[frame * input_channels + output_channel];
    }

    let frame_start = frame * input_channels;
    let frame_end = frame_start + input_channels;
    samples[frame_start..frame_end].iter().sum::<f32>() / input_channels as f32
}
