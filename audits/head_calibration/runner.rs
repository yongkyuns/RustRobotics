#[path = "head.rs"]
mod head;
use std::{env, fs, io, path::Path};

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn execute(input: &Path, output: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = fs::read(input)?;
    if bytes.len() < 8 || &bytes[..4] != b"HC01" {
        return Err(invalid("invalid fixture header").into());
    }
    let rows = u32::from_le_bytes(bytes[4..8].try_into()?) as usize;
    if rows == 0 || rows > 100_000 {
        return Err(invalid("invalid fixture rows").into());
    }
    let expected = 8 + (head::COEFFICIENTS + rows * head::COEFFICIENTS) * 4;
    if bytes.len() != expected {
        return Err(invalid("invalid fixture length").into());
    }
    let (chunks, remainder) = bytes[8..].as_chunks::<4>();
    if !remainder.is_empty() {
        return Err(invalid("unaligned fixture").into());
    }
    let numbers: Vec<_> = chunks.iter().map(|x| f32::from_le_bytes(*x)).collect();
    let incoming: [f32; head::COEFFICIENTS] = numbers[..head::COEFFICIENTS].try_into()?;
    let (records, tail) = numbers[head::COEFFICIENTS..].as_chunks::<{ head::COEFFICIENTS }>();
    if !tail.is_empty() {
        return Err(invalid("partial record").into());
    }
    let mut features = Vec::with_capacity(rows);
    let mut targets = Vec::with_capacity(rows);
    for row in records {
        features.push(row[..head::FEATURES].try_into()?);
        targets.push(row[head::FEATURES]);
    }
    let fitted = head::fit(&features, &targets, &incoming)
        .map_err(|e| invalid(&format!("fit failed: {e:?}")))?;
    fs::create_dir_all(output)?;
    let coefficients: Vec<_> = fitted.coefficients.iter().flat_map(|x| x.to_le_bytes()).collect();
    fs::write(output.join("head.bin"), coefficients)?;
    let delta: Vec<_> = fitted.delta.iter().flat_map(|x| x.to_le_bytes()).collect();
    fs::write(output.join("delta.bin"), delta)?;
    let predictions: Result<Vec<_>, _> = features.iter().map(|h| fitted.predict(h)).collect();
    let predictions = predictions.map_err(|e| invalid(&format!("prediction failed: {e:?}")))?;
    let predictions: Vec<_> = predictions.iter().flat_map(|x| x.to_le_bytes()).collect();
    fs::write(output.join("predictions.bin"), predictions)?;
    fs::write(output.join("summary.json"), format!(
        "{{\"rows\":{},\"relative_residual\":{},\"objective_before\":{},\"objective_after\":{},\"solves\":1}}\n",
        rows, fitted.relative_residual, fitted.objective_before, fitted.objective_after
    ))?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args_os().collect();
    if args.len() != 3 {
        return Err(invalid("usage: head-runner input.bin output-directory").into());
    }
    execute(Path::new(&args[1]), Path::new(&args[2]))
}
