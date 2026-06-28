use std::{io::Write, path::Path};

pub(crate) fn audio_stats(samples: &[f32]) -> (f32, f32) {
    if samples.is_empty() {
        return (0.0, 0.0);
    }

    let mean_square =
        samples.iter().map(|sample| sample * sample).sum::<f32>() / samples.len() as f32;
    let peak = samples
        .iter()
        .map(|sample| sample.abs())
        .fold(0.0, f32::max);

    (mean_square.sqrt(), peak)
}

pub(crate) fn resample_mono(
    input: &[f32],
    input_sample_rate: u32,
    output_sample_rate: u32,
) -> Vec<f32> {
    if input.is_empty() || input_sample_rate == output_sample_rate {
        return input.to_vec();
    }

    let output_len =
        (input.len() as f64 * output_sample_rate as f64 / input_sample_rate as f64).ceil() as usize;
    let mut output = Vec::with_capacity(output_len);

    for output_index in 0..output_len {
        let input_position =
            output_index as f64 * input_sample_rate as f64 / output_sample_rate as f64;
        let previous_index = input_position.floor() as usize;
        let next_index = (previous_index + 1).min(input.len() - 1);
        let lerp = (input_position - previous_index as f64) as f32;
        let previous_sample = input[previous_index];
        let next_sample = input[next_index];

        output.push(previous_sample + (next_sample - previous_sample) * lerp);
    }

    output
}

pub(crate) fn write_wav_i16(path: &Path, samples: &[f32], sample_rate: u32) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    let data_len = samples.len() as u32 * 2;
    let byte_rate = sample_rate * 2;

    file.write_all(b"RIFF")?;
    file.write_all(&(36 + data_len).to_le_bytes())?;
    file.write_all(b"WAVEfmt ")?;
    file.write_all(&16u32.to_le_bytes())?;
    file.write_all(&1u16.to_le_bytes())?;
    file.write_all(&1u16.to_le_bytes())?;
    file.write_all(&sample_rate.to_le_bytes())?;
    file.write_all(&byte_rate.to_le_bytes())?;
    file.write_all(&2u16.to_le_bytes())?;
    file.write_all(&16u16.to_le_bytes())?;
    file.write_all(b"data")?;
    file.write_all(&data_len.to_le_bytes())?;

    for sample in samples {
        let pcm = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        file.write_all(&pcm.to_le_bytes())?;
    }

    Ok(())
}
