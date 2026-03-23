"""Pipeline position: AUDIO POST-PROCESSING — chunk smoothing utilities.

Role in pipeline:
  Applied between codec decode and WAV encode in the streaming path to
  reduce audible artefacts at chunk boundaries.

  crossfade(prev_pcm, next_pcm, n_samples)
    → linear overlap-add between two consecutive chunks.

  fade_out(pcm, n_samples)
    → linear fade applied to the tail of non-final chunks to suppress
      codec boundary noise.

  resample_audio(pcm, orig_sr, target_sr)
    → librosa-based resampling for sample-rate conversion.
"""

from __future__ import annotations

import numpy as np


def crossfade(prev_pcm: np.ndarray, next_pcm: np.ndarray, n_samples: int) -> np.ndarray:
    """Linear crossfade the tail of prev_pcm into the head of next_pcm.

    Returns next_pcm with its first n_samples blended with the last
    n_samples of prev_pcm.
    """
    if n_samples <= 0 or len(prev_pcm) < n_samples or len(next_pcm) < n_samples:
        return next_pcm

    fade_out_curve = np.linspace(1.0, 0.0, n_samples, dtype=np.float32)
    fade_in_curve  = np.linspace(0.0, 1.0, n_samples, dtype=np.float32)

    blended = prev_pcm[-n_samples:] * fade_out_curve + next_pcm[:n_samples] * fade_in_curve
    return np.concatenate([blended, next_pcm[n_samples:]])


def fade_out(pcm: np.ndarray, n_samples: int) -> np.ndarray:
    """Apply a linear fade-out to the last n_samples of pcm."""
    if n_samples <= 0 or len(pcm) < n_samples:
        return pcm
    result = pcm.copy()
    fade_curve = np.linspace(1.0, 0.0, n_samples, dtype=np.float32)
    result[-n_samples:] *= fade_curve
    return result


def resample_audio(pcm: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample PCM from orig_sr to target_sr using librosa."""
    if orig_sr == target_sr:
        return pcm
    import librosa
    return librosa.resample(pcm.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr)


def process_audio_pipeline(pcm: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Apply the full post-processing pipeline: resample → (future stages)."""
    pcm = resample_audio(pcm, sr, target_sr)
    return pcm
